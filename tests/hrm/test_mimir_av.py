"""Mimir (HRM-Text) as the verbalizer backbone: batch/padding invariance of generation, teacher-forced scoring and the AR
readout, the response-only LM head, and backbone bookkeeping. Tiny random models, CPU."""
import json

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import HrmTextConfig, HrmTextForCausalLM, Qwen2Config, Qwen2ForCausalLM

from nla.hrm.mimir_ar import mimir_ar_hidden
from nla.hrm.model import (
    LORA_TARGET_MODULES, critic_hidden, forward_kwargs, generate_kwargs, is_mimir, read_backbone,
    resolve_verbalizer_model, response_token_ce, write_backbone,
)


def _peft(base):
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=LORA_TARGET_MODULES, lora_dropout=0.0, task_type="CAUSAL_LM")
    model = get_peft_model(base.float().eval(), cfg, adapter_name="av")
    model.add_adapter("ar", cfg)
    model.set_adapter("av")
    with torch.no_grad():  # LoRA B starts at zero (a no-op): randomise so the adapters actually matter
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    return model


def _mimir():
    torch.manual_seed(0)
    cfg = HrmTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=8, num_attention_heads=2,
                        num_key_value_heads=2, head_dim=16, H_cycles=2, L_cycles=3, L_bp_cycles=[3, 3], embedding_scale=5.0,
                        pad_token_id=0, prefix_lm=True)
    return _peft(HrmTextForCausalLM(cfg))


def _qwen():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2)
    return _peft(Qwen2ForCausalLM(cfg))


def _left_pad(seqs, pad=0):
    w = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), w), pad)
    attn = torch.zeros((len(seqs), w), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, w - len(s):] = torch.tensor(s)
        attn[i, w - len(s):] = 1
    return ids, attn


def test_is_mimir():
    assert is_mimir(_mimir()) and not is_mimir(_qwen())
    assert generate_kwargs(_qwen(), torch.ones(1, 3, dtype=torch.long)) == {}
    assert forward_kwargs(_qwen(), torch.ones(1, 3, dtype=torch.long), n_prompt=2) == {}


def test_left_padded_generate_matches_single_rows():
    model = _mimir()
    emb = model.get_input_embeddings()
    a, b = [5, 6, 7], [11, 12, 13, 14, 15, 16]
    ids, attn = _left_pad([a, b])

    def gen(i, at):
        with torch.no_grad():
            return model.generate(inputs_embeds=emb(i), attention_mask=at, max_new_tokens=6, do_sample=False,
                                  pad_token_id=0, **generate_kwargs(model, at))

    batched = gen(ids, attn)
    for k, s in enumerate((a, b)):
        alone = gen(torch.tensor([s]), torch.ones(1, len(s), dtype=torch.long))
        assert torch.equal(batched[k], alone[0])


def test_teacher_forced_logprobs_left_padding_invariant():
    model = _mimir()
    emb = model.get_input_embeddings()
    prompts, resp = ([5, 6, 7], [11, 12, 13, 14, 15]), [20, 21, 22]
    pids, pattn = _left_pad(list(prompts))
    S = pids.shape[1]
    full = torch.cat([pids, torch.tensor([resp] * 2)], dim=1)
    fattn = torch.cat([pattn, torch.ones(2, len(resp), dtype=torch.long)], dim=1)
    k = full.shape[1] - S + 1
    with torch.no_grad():
        batched = model(inputs_embeds=emb(full), attention_mask=fattn, logits_to_keep=k,
                        **forward_kwargs(model, fattn, n_prompt=S)).logits
        for i, p in enumerate(prompts):
            one = torch.tensor([p + resp])
            oa = torch.ones_like(one)
            alone = model(inputs_embeds=emb(one), attention_mask=oa, logits_to_keep=len(resp) + 1,
                          **forward_kwargs(model, oa, n_prompt=len(p))).logits
            assert torch.allclose(batched[i], alone[0], atol=1e-4)


def test_teacher_forced_matches_generation_conditioning():
    """The logits that score the first generated token must be the ones generate used (prompt bidirectional)."""
    model = _mimir()
    emb = model.get_input_embeddings()
    p = [5, 6, 7, 8]
    ids = torch.tensor([p])
    at = torch.ones_like(ids)
    with torch.no_grad():
        first = model.generate(inputs_embeds=emb(ids), attention_mask=at, max_new_tokens=1, do_sample=False, pad_token_id=0,
                               **generate_kwargs(model, at))[0, 0]
        logits = model(inputs_embeds=emb(ids), attention_mask=at, logits_to_keep=1, **forward_kwargs(model, at)).logits[0, -1]
    assert int(logits.argmax()) == int(first)


def test_critic_hidden_pad_side_invariant_and_matches_mimir_ar():
    model = _mimir()
    model.set_adapter("ar")
    seqs = [[5, 6, 7], [11, 12, 13, 14, 15]]
    lids, lattn = _left_pad(seqs)
    rids = torch.zeros_like(lids)
    rattn = torch.zeros_like(lattn)
    for i, s in enumerate(seqs):
        rids[i, : len(s)] = torch.tensor(s)
        rattn[i, : len(s)] = 1
    with torch.no_grad():
        h_left = critic_hidden(model, lids, lattn)
        h_right = critic_hidden(model, rids, rattn)
        assert torch.allclose(h_left, h_right, atol=1e-4)
        assert torch.allclose(h_right, mimir_ar_hidden(model, rids, rattn), atol=1e-4)


def test_response_ce_mimir_matches_full_logits_and_is_padding_invariant():
    model = _mimir()
    emb = model.get_input_embeddings()
    rows = [([5, 6, 7], [20, 21, 22, 23]), ([11, 12, 13, 14, 15], [30, 31])]
    w = max(len(p) + len(r) for p, r in rows)
    ids = torch.zeros(2, w, dtype=torch.long)
    labels = torch.full((2, w), -100)
    attn = torch.zeros(2, w, dtype=torch.long)
    for i, (p, r) in enumerate(rows):
        n = len(p) + len(r)
        ids[i, :n] = torch.tensor(p + r)
        labels[i, len(p):n] = torch.tensor(r)
        attn[i, :n] = 1
    with torch.no_grad():
        ce = response_token_ce(model, emb(ids), attn, labels)
        kw = forward_kwargs(model, attn, prompt_mask=(labels == -100))
        logits = model(inputs_embeds=emb(ids), attention_mask=attn, **kw).logits[:, :-1].float()
        ref = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100,
                              reduction="none").view(2, -1)
        assert torch.allclose(ce, ref, atol=1e-4)
        assert (ce[labels[:, 1:] == -100] == 0).all()
        # a row alone gives the same per-token CE as inside the padded batch
        for i, (p, r) in enumerate(rows):
            n = len(p) + len(r)
            alone = response_token_ce(model, emb(ids[i : i + 1, :n]), attn[i : i + 1, :n], labels[i : i + 1, :n])
            assert torch.allclose(alone[0], ce[i, : n - 1], atol=1e-4)


def test_response_ce_qwen_unchanged_formula():
    model = _qwen()
    emb = model.get_input_embeddings()
    ids = torch.tensor([[5, 6, 7, 20, 21, 0], [11, 12, 13, 14, 30, 31]])
    attn = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    labels = torch.tensor([[-100, -100, -100, 20, 21, -100], [-100, -100, -100, -100, 30, 31]])
    with torch.no_grad():
        ce = response_token_ce(model, emb(ids), attn, labels)
        logits = model(inputs_embeds=emb(ids), attention_mask=attn).logits[:, :-1].float()
        old = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(ce.sum() / (labels[:, 1:] != -100).sum(), old, atol=1e-6)


def test_response_ce_trains_only_av_adapter():
    model = _mimir()
    for n, p in model.named_parameters():
        p.requires_grad_(".av." in n)
    emb = model.get_input_embeddings()
    ids = torch.tensor([[5, 6, 7, 20, 21]])
    labels = torch.tensor([[-100, -100, -100, 20, 21]])
    response_token_ce(model, emb(ids), torch.ones_like(ids), labels).sum().backward()
    grads = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads and all(".av." in n for n in grads)


def test_backbone_record(tmp_path):
    assert read_backbone(str(tmp_path)) is None
    write_backbone(str(tmp_path), "x/y")
    assert json.loads((tmp_path / "backbone.json").read_text()) == {"verbalizer_model": "x/y"}
    assert resolve_verbalizer_model(None, str(tmp_path)) == "x/y"
    assert resolve_verbalizer_model("a/b", str(tmp_path)) == "a/b"
    assert resolve_verbalizer_model(None, str(tmp_path / "none")) == "Qwen/Qwen2.5-1.5B-Instruct"
