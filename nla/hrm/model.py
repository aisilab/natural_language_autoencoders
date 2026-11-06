"""The HRM verbalizer/reconstructor: one frozen Qwen2.5-1.5B-Instruct base with
two PEFT LoRA adapters (`av`, `ar`; an optional frozen `av_ref` snapshot is
added for the RL KL penalty — see `train_rl.py`), plus small trainable
per-stream injection/reconstruction affine maps that are NOT part of any
LoRA adapter (they're new parameters, always trainable, saved/loaded
separately — see `save_extra_modules`/`load_extra_modules`).

Deviation from the original NLA (docs/hrm.md): AV and AR are Qwen2.5-1.5B,
not copies of the target model, and the AR is the FULL-DEPTH model (no layer
truncation) plus an affine head — the original NLA's `NLACriticModel`
truncates to K+1 layers, which has no well-defined analog for a recurrent
HRM target.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from nla.injection import inject_at_marked_positions

DEFAULT_VERBALIZER = "Qwen/Qwen2.5-1.5B-Instruct"

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def is_mimir(model_or_config) -> bool:
    """True if the (PEFT-wrapped) model or its config is an HRM-Text (Mimir) model, i.e. the verbalizer backbone is a copy of
    the target model instead of Qwen. Everything Mimir-specific below is gated on this, so Qwen behaviour is unchanged."""
    cfg = getattr(model_or_config, "config", model_or_config)
    return getattr(cfg, "model_type", None) == "hrm_text"


def load_verbalizer(
    model_name: str = DEFAULT_VERBALIZER,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
    lora_r: int = 32,
    lora_alpha: int = 64,
) -> tuple[PeftModel, AutoTokenizer]:
    """Frozen base + two LoRA adapters ('av', 'ar'). `model.set_adapter(name)`
    switches which one is active for a forward pass; the base's own weights
    are never updated (`get_peft_model` freezes everything not matched by
    `target_modules`). `model_name` may be Mimir itself (a Mimir-native AV/AR pair, `docs/hrm.md`): the same LoRA recipe
    applies (module names match), attention is forced to sdpa (the PrefixLM mask is not representable otherwise)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    extra = {"attn_implementation": "sdpa"} if is_mimir(AutoConfig.from_pretrained(model_name)) else {}
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, **extra).to(device)
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0, task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg, adapter_name="av")
    model.add_adapter("ar", lora_cfg)
    model.set_adapter("av")
    return model, tokenizer


def add_frozen_reference_adapter(model: PeftModel, source_adapter: str = "av", ref_name: str = "av_ref") -> None:
    """Clone `source_adapter`'s current weights into a new adapter that is
    never trained — the RL KL-penalty reference (`train_rl.py`), a frozen
    snapshot of the AV right after SFT. Must be called AFTER SFT, before RL."""
    cfg = model.peft_config[source_adapter]
    model.add_adapter(ref_name, cfg)
    src_sd = get_peft_model_state_dict_for(model, source_adapter)
    dst_sd = {k.replace(f".{source_adapter}.weight", f".{ref_name}.weight")
              .replace(f".{source_adapter}.", f".{ref_name}."): v for k, v in src_sd.items()}
    _missing, unexpected = model.load_state_dict(dst_sd, strict=False)
    assert not unexpected, f"unexpected keys loading {ref_name}: {unexpected}"
    for name, p in model.named_parameters():
        if f".{ref_name}." in name:
            p.requires_grad_(False)


def get_peft_model_state_dict_for(model: PeftModel, adapter_name: str) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if f".{adapter_name}." in k}


def _init_near_identity(linear: nn.Linear, d_in: int, d_out: int, init_scale: float, init_noise_std: float) -> None:
    """weight ~= init_scale * eye + noise. eye() only when square (d_in==d_out,
    the common case — Mimir's d_model happens to equal Qwen2.5-1.5B's hidden
    size); otherwise falls back to a scaled-orthogonal-ish init (eye of the
    non-square shape, i.e. the leading min(d_in,d_out)x min(d_in,d_out) block
    is identity) so the smoke tests can swap in a differently-sized verbalizer."""
    with torch.no_grad():
        eye = torch.zeros(d_out, d_in)
        k = min(d_in, d_out)
        eye[:k, :k] = torch.eye(k)
        linear.weight.copy_(init_scale * eye + init_noise_std * torch.randn(d_out, d_in))
        linear.bias.zero_()


class InjectionAdapter(nn.Module):
    """Per-stream affine map: Mimir's raw z (norm ~= sqrt(d_model) ~= 39.2,
    from HrmTextStack's un-weighted final RMSNorm), d_mimir-dim -> verbalizer
    embedding space, d_verbalizer-dim (equal for the default Qwen2.5-1.5B
    target, but kept as two separate dims for robustness). Near-identity
    init, scaled by `init_scale` (see docs/hrm.md — the original NLA's
    heuristic: init large, near the verbalizer's own ambient residual-stream
    scale, not its token-embedding scale)."""

    def __init__(self, d_mimir: int, d_verbalizer: int | None = None, init_scale: float = 1.0, init_noise_std: float = 0.01):
        super().__init__()
        d_verbalizer = d_mimir if d_verbalizer is None else d_verbalizer
        self.linear = nn.Linear(d_mimir, d_verbalizer, bias=True)
        _init_near_identity(self.linear, d_mimir, d_verbalizer, init_scale, init_noise_std)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z.float())


class ReconHeads(nn.Module):
    """head_L, head_H: verbalizer final hidden state (d_verbalizer-dim) ->
    raw z_L / z_H space (d_mimir-dim). Same near-identity-scaled-affine init
    as InjectionAdapter; `init_scale` here should be small (Qwen's hidden-
    state norm >> Mimir's ~39.2) — train scripts compute it from
    norm_stats.json (see train_av_sft.py's `--head-init-scale`, defaulted
    from the injection scale's reciprocal)."""

    def __init__(self, d_mimir: int, d_verbalizer: int | None = None, init_scale: float = 0.02, init_noise_std: float = 0.001):
        super().__init__()
        d_verbalizer = d_mimir if d_verbalizer is None else d_verbalizer
        self.head_L = nn.Linear(d_verbalizer, d_mimir, bias=True)
        self.head_H = nn.Linear(d_verbalizer, d_mimir, bias=True)
        for head in (self.head_L, self.head_H):
            _init_near_identity(head, d_verbalizer, d_mimir, init_scale, init_noise_std)

    def forward_L(self, h: torch.Tensor) -> torch.Tensor:
        return self.head_L(h.float())

    def forward_H(self, h: torch.Tensor) -> torch.Tensor:
        return self.head_H(h.float())


def render_av_prompt(tokenizer, user_content: str, *, add_generation_prompt: bool = True) -> torch.Tensor:
    """Render a single-turn Qwen chat prompt -> input_ids [1, S]. Explicit
    `["input_ids"]` indexing: `apply_chat_template(tokenize=True)` returns a
    `BatchEncoding` (dict-like) on this repo's transformers version, not a
    bare list/tensor — see `nla/hrm/injection.py`'s note on the same gotcha."""
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True, add_generation_prompt=add_generation_prompt,
        return_tensors="pt", return_dict=True,
    )
    return enc["input_ids"]


def build_inputs_embeds(
    model: PeftModel,
    input_ids: torch.Tensor,
    z_L: torch.Tensor,
    z_H: torch.Tensor,
    inj_L: InjectionAdapter,
    inj_H: InjectionAdapter,
    inj_l_id: int, left_l_id: int, right_l_id: int,
    inj_h_id: int, left_h_id: int, right_h_id: int,
) -> torch.Tensor:
    """Embed input_ids, then overwrite the [L]/[H] marker positions with the
    (adapter-mapped) activation vectors. TWO separate calls to
    `inject_at_marked_positions` (one per marker) — see `nla/hrm/injection.py`
    module docstring for why two distinct markers instead of one repeated
    marker. z_L, z_H: [B, d_model] (one vector per sample — the row's gold or
    reconstructed stream state)."""
    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(input_ids)
    vec_l = inj_L(z_L).to(embeds.dtype)
    vec_h = inj_H(z_H).to(embeds.dtype)
    embeds = inject_at_marked_positions(input_ids, embeds, vec_l, inj_l_id, left_l_id, right_l_id)
    embeds = inject_at_marked_positions(input_ids, embeds, vec_h, inj_h_id, left_h_id, right_h_id)
    return embeds


def _positions(attn: torch.Tensor) -> torch.Tensor:
    """Position ids that ignore padding (what `generate` derives from the attention mask), so a left-padded row sees the
    same positions as when it is alone."""
    return (attn.long().cumsum(dim=-1) - 1).clamp(min=0)


def generate_kwargs(model, attn: torch.Tensor) -> dict:
    """Extra `generate(inputs_embeds=...)` kwargs: for Mimir the whole prompt is ONE bidirectional block (as when Mimir reads
    a rendered prompt), the generated tokens are causal. {} for other backbones."""
    return {"token_type_ids": attn.long()} if is_mimir(model) else {}


def forward_kwargs(model, attn: torch.Tensor, prompt_mask: torch.Tensor | None = None, n_prompt: int | None = None) -> dict:
    """Extra kwargs for a teacher-forced forward over [prompt | response] that must match `generate_kwargs` decoding.
    `prompt_mask` [B, S] marks prompt tokens (e.g. labels == -100 with right padding); `n_prompt` marks the first n columns
    (left-padded batches). With neither, every attended token is prompt. {} for other backbones."""
    if not is_mimir(model):
        return {}
    if prompt_mask is not None:
        tt = prompt_mask.long() * attn.long()
    elif n_prompt is not None:
        tt = attn.long() * (torch.arange(attn.shape[1], device=attn.device) < n_prompt).long().unsqueeze(0)
    else:
        tt = attn.long()
    return {"token_type_ids": tt, "position_ids": _positions(attn)}


def critic_hidden(model, input_ids: torch.Tensor, attn: torch.Tensor, readout: str = "last") -> torch.Tensor:
    """[B, d] AR readout at the last real token under the active ('ar') adapter. Qwen: last hidden state; Mimir: final H
    output with the whole text bidirectional (`nla/hrm/mimir_ar.py`). Correct for left and right padding."""
    idx = last_real_index(attn, readout)
    if is_mimir(model):
        h = model.get_base_model().model(input_ids=input_ids, attention_mask=attn, token_type_ids=attn.long(),
                                         position_ids=_positions(attn)).last_hidden_state
    else:
        # logits_to_keep=1: only hidden states are needed; do not build [B, T, vocab] logits.
        h = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, logits_to_keep=1).hidden_states[-1]
    return h[torch.arange(input_ids.shape[0], device=h.device), idx]


def response_token_ce(model, embeds: torch.Tensor, attn: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token cross-entropy [B, S-1] of the shifted labels (0 where labels == -100) for an AV forward over
    [prompt | response] (right padded; prompt = labels == -100 among attended tokens). Mimir: the LM head is applied to the
    response positions only (vocab 262,144, so full logits are several GB per batch)."""
    shift = labels[:, 1:]
    if not is_mimir(model):
        logits = model(inputs_embeds=embeds, attention_mask=attn).logits[:, :-1]
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), shift.reshape(-1),
                               ignore_index=-100, reduction="none").view_as(shift)
    kw = forward_kwargs(model, attn, prompt_mask=(labels == -100))
    base = model.get_base_model()
    h = base.model(inputs_embeds=embeds, attention_mask=attn, **kw).last_hidden_state[:, :-1]
    sel = shift != -100
    ce = torch.zeros(shift.shape, device=h.device, dtype=torch.float32)
    ce[sel] = F.cross_entropy(base.lm_head(h[sel]).float(), shift[sel], reduction="none")
    return ce


def write_backbone(ckpt_dir: str, model_name: str) -> None:
    """Record which base model a checkpoint was trained on (`backbone.json`), so eval/infer need no extra flag."""
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    (Path(ckpt_dir) / "backbone.json").write_text(json.dumps({"verbalizer_model": model_name}))


def read_backbone(ckpt_dir: str | None) -> str | None:
    f = Path(ckpt_dir) / "backbone.json" if ckpt_dir else None
    return json.loads(f.read_text())["verbalizer_model"] if f is not None and f.exists() else None


def resolve_verbalizer_model(arg: str | None, *ckpt_dirs: str | None) -> str:
    """--verbalizer-model if given, else the backbone recorded in the first checkpoint that has one, else the Qwen default.
    A mismatch between the flag and a recorded backbone is only warned about (a local copy of the same model has another
    name); loading adapters into a different architecture fails on shapes."""
    recorded = next((b for b in (read_backbone(d) for d in ckpt_dirs) if b), None)
    if arg is not None and recorded is not None and arg != recorded:
        print(f"[warning] --verbalizer-model {arg} differs from the checkpoint's recorded backbone {recorded}")
    return arg or recorded or DEFAULT_VERBALIZER


def last_real_index(attn: torch.Tensor, readout: str = "last") -> torch.Tensor:
    """Index of the token the AR reads out, per row. "last": the last attended token, correct for LEFT and RIGHT padding.
    "legacy": attn.sum - 1, which is only correct for right padding; with a left-padded batch (the tokenizer is switched to
    left padding for generation in eval.py / train_rl.py) it points into the padding or the middle of the text for every row
    shorter than the longest one. All AR numbers logged before 2026-10-11 (RL rewards, the online AR update, eval FVE) used
    the legacy readout; AR-SFT always used right padding (correct)."""
    if readout == "legacy":
        return attn.sum(dim=1) - 1
    assert readout == "last", readout
    pos = torch.arange(attn.shape[1], device=attn.device).unsqueeze(0)
    return (attn.long() * (pos + 1)).argmax(dim=1)


def load_sft_adapter(model: PeftModel, ckpt_dir: str, adapter_name: str) -> None:
    """Load a `train_av_sft.py`/`train_ar_sft.py` checkpoint's LoRA weights
    INTO the model's existing `adapter_name` slot (`save_extra_modules`'s
    companion on the LoRA side). `PeftModel.load_adapter` can only load into
    a NEW adapter name it invents from the checkpoint — `ckpt_dir` is that
    script's `--output` (PEFT nests under `{output}/{adapter_name}/`, only an
    adapter literally named "default" saves flat — see docs/hrm.md), so we
    load into a throwaway name, copy the state dict across, then drop it."""
    tmp_name = f"_load_tmp_{adapter_name}"
    model.load_adapter(f"{ckpt_dir}/{adapter_name}", adapter_name=tmp_name)
    src_sd = {k: v for k, v in model.state_dict().items() if f".{tmp_name}." in k}
    model.load_state_dict({k.replace(f".{tmp_name}.", f".{adapter_name}."): v for k, v in src_sd.items()}, strict=False)
    model.delete_adapter(tmp_name)


def load_rl_checkpoint(
    model: PeftModel, av_sft_ckpt: str, ar_sft_ckpt: str, d_mimir: int, d_verbalizer: int, device: str,
) -> tuple[InjectionAdapter, InjectionAdapter, ReconHeads]:
    """Load BOTH the av and ar LoRA adapters plus their extra modules
    (injection adapters from av_sft_ckpt, heads from ar_sft_ckpt) into
    `model`'s existing 'av'/'ar' adapter slots. Used by `train_rl.py` (then
    followed by `add_frozen_reference_adapter`) and `eval.py`/`judge.py`
    callers that need a trained AV+AR pair without any further training."""
    load_sft_adapter(model, av_sft_ckpt, "av")
    inj_L = InjectionAdapter(d_mimir, d_verbalizer).to(device)
    inj_H = InjectionAdapter(d_mimir, d_verbalizer).to(device)
    load_extra_modules(f"{av_sft_ckpt}/extra_modules.safetensors", inj_L, inj_H, ReconHeads(d_mimir, d_verbalizer))

    load_sft_adapter(model, ar_sft_ckpt, "ar")
    heads = ReconHeads(d_mimir, d_verbalizer).to(device)
    load_extra_modules(f"{ar_sft_ckpt}/extra_modules.safetensors",
                         InjectionAdapter(d_mimir, d_verbalizer), InjectionAdapter(d_mimir, d_verbalizer), heads)
    return inj_L, inj_H, heads


def save_extra_modules(path: str, inj_L: InjectionAdapter, inj_H: InjectionAdapter, heads: ReconHeads) -> None:
    sd = {}
    sd.update({f"inj_L.{k}": v for k, v in inj_L.state_dict().items()})
    sd.update({f"inj_H.{k}": v for k, v in inj_H.state_dict().items()})
    sd.update({f"heads.{k}": v for k, v in heads.state_dict().items()})
    save_file(sd, path)


def load_extra_modules(path: str, inj_L: InjectionAdapter, inj_H: InjectionAdapter, heads: ReconHeads) -> None:
    sd = load_file(path)
    inj_L.load_state_dict({k.removeprefix("inj_L."): v for k, v in sd.items() if k.startswith("inj_L.")})
    inj_H.load_state_dict({k.removeprefix("inj_H."): v for k, v in sd.items() if k.startswith("inj_H.")})
    heads.load_state_dict({k.removeprefix("heads."): v for k, v in sd.items() if k.startswith("heads.")})
