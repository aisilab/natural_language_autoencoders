"""AV-SFT: train the `av` LoRA adapter + per-stream injection adapters
(inj_L, inj_H) with next-token loss on the response only. Format-only warm-up
— teaches the `L: ...\nH: ...` field format and that generation should
condition on the injected vectors at all; teaches nothing about what makes L
different from H (see `nla/hrm/build.py` — e_a/e_b are randomly assigned).
Stops once the format rate is high on eval (`--target-format-rate`, checked
every `--eval-every` steps) — further SFT beyond format-acquisition just
overfits the randomly-assigned L/H split.
"""

import argparse
import json

import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from nla.hrm.devices import default_device, default_dtype
from nla.datagen.storage import LocalStorage
from nla.hrm.build import _INJECT_H_PLACEHOLDER, _INJECT_L_PLACEHOLDER, prefix_token_mask
from nla.hrm.model import (
    InjectionAdapter, build_inputs_embeds, generate_kwargs, is_mimir, load_verbalizer, render_av_prompt,
    resolve_verbalizer_model, response_token_ce, save_extra_modules, write_backbone,
)
from nla.hrm.sidecar import read_sidecar
from nla.schema import extract_explanation

_D_MIMIR = 1536


def _load_rows(parquet_path: str) -> list[dict]:
    t = pq.read_table(parquet_path, columns=["prompt", "response", "z_L", "z_H"])
    return t.to_pylist()


def _build_example(tokenizer, row: dict, inj_l_char: str, inj_h_char: str, with_pmask: bool = False) -> dict:
    content = row["prompt"][0]["content"].replace(_INJECT_L_PLACEHOLDER, inj_l_char).replace(_INJECT_H_PLACEHOLDER, inj_h_char)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    input_ids = prompt_ids + response_ids + [eos]
    labels = [-100] * len(prompt_ids) + response_ids + [eos]
    ex = {"input_ids": input_ids, "labels": labels, "z_L": row["z_L"], "z_H": row["z_H"]}
    if with_pmask:
        ex["pmask"] = prefix_token_mask(tokenizer, row["response"], len(response_ids))
    return ex


def _token_weights(batch: list[dict], labels: torch.Tensor, weight: float) -> torch.Tensor:
    """Per-position loss weights aligned with `labels` (before the one-token shift): `weight` on tokens inside a
    `Marked token: "X".` span, 1 elsewhere, 0 where labels are ignored."""
    w = (labels != -100).float()
    for i, e in enumerate(batch):
        pm = e.get("pmask")
        if pm is not None:
            n_prompt = sum(1 for x in e["labels"] if x == -100)
            w[i, n_prompt : n_prompt + len(pm)] += (weight - 1.0) * torch.tensor(pm, dtype=torch.float32, device=w.device)
    return w


def _collate(examples: list[dict], pad_id: int, device: str):
    max_len = max(len(e["input_ids"]) for e in examples)
    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(examples), max_len), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        input_ids[i, :n] = torch.tensor(e["input_ids"])
        labels[i, :n] = torch.tensor(e["labels"])
        attn[i, :n] = 1
    z_L = torch.tensor([e["z_L"] for e in examples], dtype=torch.float32)
    z_H = torch.tensor([e["z_H"] for e in examples], dtype=torch.float32)
    return input_ids.to(device), labels.to(device), attn.to(device), z_L.to(device), z_H.to(device)


@torch.no_grad()
def generate_and_check(model, tokenizer, row, inj_l_char, inj_h_char, inj_L, inj_H, ids_meta, device, max_new_tokens=200):
    content = row["prompt"][0]["content"].replace(_INJECT_L_PLACEHOLDER, inj_l_char).replace(_INJECT_H_PLACEHOLDER, inj_h_char)
    prompt_ids = render_av_prompt(tokenizer, content).to(device)
    z_L = torch.tensor([row["z_L"]], dtype=torch.float32, device=device)
    z_H = torch.tensor([row["z_H"]], dtype=torch.float32, device=device)
    embeds = build_inputs_embeds(model, prompt_ids, z_L, z_H, inj_L, inj_H, *ids_meta)
    attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
    out_ids = model.generate(inputs_embeds=embeds, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id, **generate_kwargs(model, attn))
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    expl = extract_explanation(text)
    parsed = expl is not None and expl.count("L:") >= 1 and expl.count("H:") >= 1
    return parsed, text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--eval-parquet", default=None)
    p.add_argument("--verbalizer-model", default=None,
                    help="AV base model. Default: the backbone recorded in --init-from, else Qwen2.5-1.5B-Instruct. Pass "
                         "danish-foundation-models/DFM-Mimir-v1.5 for a Mimir-native AV (data built with build.py "
                         "--verbalizer-model of the same name)")
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default=default_dtype())
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--injection-scale-init", type=float, default=None,
                    help="defaults to norm_stats.json's injection_scale_p75, or 5.0 if unset")
    p.add_argument("--norm-stats-json", default=None)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--target-format-rate", type=float, default=0.99,
                    help="stop early once the greedy format rate on eval reaches this. NOTE: the format is learned in a few "
                         "dozen steps, but READING the injected vector (what makes explanations faithful) takes far longer; "
                         "pass a value >1 (e.g. 2) to disable early stopping and train for --epochs, and check progress with "
                         "`python -m nla.hrm.nll_check`")
    p.add_argument("--prefix-weight", type=float, default=1.0,
                    help="loss weight on the tokens of the `Marked token: \"X\".` span (data built with build.py "
                         "--prefix-token). The span is ~6 of ~250 tokens, so at weight 1 it barely moves the loss; try 5 to 10 "
                         "to push the verbalizer to read the token. 1.0 = unchanged behaviour")
    p.add_argument("--save-every", type=int, default=0,
                    help="also save the adapter to <output>/step_N every N steps (0 = only at the end)")
    p.add_argument("--max-steps", type=int, default=None, help="hard cap in addition to --epochs")
    p.add_argument("--init-from", default=None,
                    help="continue from an earlier train_av_sft checkpoint dir (av adapter + injection adapters): e.g. stage 2 "
                         "(teacher prose) after a facts-only stage 1 on many more rows; --injection-scale-init is then ignored")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    verbalizer_model = resolve_verbalizer_model(args.verbalizer_model, args.init_from)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    model, tokenizer = load_verbalizer(verbalizer_model, device=args.device, torch_dtype=dtype)
    tokenizer.padding_side = "right"

    in_meta = read_sidecar(LocalStorage(), args.train_parquet)
    tokens_meta = in_meta.tokens
    assert tokens_meta is not None, f"{args.train_parquet}'s sidecar has no tokens metadata — run build.py first"
    inj_l_char, inj_h_char = tokens_meta.injection_char_L, tokens_meta.injection_char_H
    ids_meta = (
        tokens_meta.injection_token_id_L, tokens_meta.injection_left_neighbor_id_L, tokens_meta.injection_right_neighbor_id_L,
        tokens_meta.injection_token_id_H, tokens_meta.injection_left_neighbor_id_H, tokens_meta.injection_right_neighbor_id_H,
    )

    init_scale = args.injection_scale_init
    if is_mimir(model):
        # Mimir multiplies input embeddings by embedding_scale, so 1/embedding_scale puts z (norm ~ sqrt(d)) at the model's
        # native input scale; the Qwen heuristic from norm_stats does not apply.
        init_scale = init_scale or 1.0 / model.config.embedding_scale
    elif init_scale is None and args.norm_stats_json:
        stats = json.load(open(args.norm_stats_json))
        init_scale = stats.get("injection_scale_p75")
    init_scale = init_scale or 5.0
    inj_L = InjectionAdapter(_D_MIMIR, model.config.hidden_size, init_scale=init_scale).to(args.device)
    inj_H = InjectionAdapter(_D_MIMIR, model.config.hidden_size, init_scale=init_scale).to(args.device)
    print(f"[train_av_sft] injection scale init = {init_scale}")
    if args.init_from:
        from nla.hrm.model import ReconHeads as _RH
        from nla.hrm.model import load_extra_modules, load_sft_adapter
        load_sft_adapter(model, args.init_from, "av")
        load_extra_modules(f"{args.init_from}/extra_modules.safetensors", inj_L, inj_H, _RH(_D_MIMIR, model.config.hidden_size))
        print(f"[train_av_sft] initialised the av adapter and the injection adapters from {args.init_from}")

    for name, param in model.named_parameters():
        param.requires_grad_(".av." in name and ".av_ref." not in name)
    trainable = [pr for pr in model.parameters() if pr.requires_grad] + list(inj_L.parameters()) + list(inj_H.parameters())
    print(f"[train_av_sft] {sum(p.numel() for p in trainable):,} trainable params (av LoRA + injection adapters)")
    model.set_adapter("av")
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    train_rows = _load_rows(args.train_parquet)
    eval_rows = _load_rows(args.eval_parquet) if args.eval_parquet else train_rows[: min(16, len(train_rows))]

    examples = [_build_example(tokenizer, r, inj_l_char, inj_h_char, with_pmask=args.prefix_weight != 1.0) for r in train_rows]
    step = 0
    model.train()
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        order = torch.randperm(len(examples)).tolist()
        for start in tqdm(range(0, len(examples), args.batch_size), desc=f"epoch {epoch}"):
            batch = [examples[i] for i in order[start : start + args.batch_size]]
            input_ids, labels, attn, z_L, z_H = _collate(batch, tokenizer.pad_token_id, args.device)
            embeds = build_inputs_embeds(model, input_ids, z_L, z_H, inj_L, inj_H, *ids_meta)
            ce = response_token_ce(model, embeds, attn, labels)  # [B, S-1], 0 on ignored positions
            if args.prefix_weight != 1.0:
                w = _token_weights(batch, labels, args.prefix_weight)[:, 1:]
                loss = (ce * w).sum() / w.sum().clamp_min(1.0)
            else:
                loss = ce.sum() / (labels[:, 1:] != -100).sum().clamp_min(1)

            optim.zero_grad()
            loss.backward()
            optim.step()
            step += 1
            if step % args.log_every == 0:
                print(f"  step={step} loss={loss.item():.4f}")
            if args.save_every and step % args.save_every == 0:
                from pathlib import Path as _P
                _P(f"{args.output}/step_{step}").mkdir(parents=True, exist_ok=True)
                model.save_pretrained(f"{args.output}/step_{step}", selected_adapters=["av"])
                write_backbone(f"{args.output}/step_{step}", verbalizer_model)
                from nla.hrm.model import ReconHeads as _RH
                save_extra_modules(f"{args.output}/step_{step}/extra_modules.safetensors", inj_L, inj_H,
                                   _RH(_D_MIMIR, model.config.hidden_size))
            if step % args.eval_every == 0 or (args.max_steps and step >= args.max_steps):
                n_ok = 0
                for r in eval_rows:
                    ok, _ = generate_and_check(model, tokenizer, r, inj_l_char, inj_h_char, inj_L, inj_H, ids_meta, args.device)
                    n_ok += ok
                rate = n_ok / max(len(eval_rows), 1)
                print(f"  [eval] step={step} format_rate={rate:.2%} (target {args.target_format_rate:.0%})")
                model.train()
                if rate >= args.target_format_rate:
                    print("  target format rate reached — stopping AV-SFT")
                    stop = True
                    break
            if args.max_steps and step >= args.max_steps:
                stop = True
                break

    from pathlib import Path
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, selected_adapters=["av"])
    write_backbone(args.output, verbalizer_model)
    from nla.hrm.model import ReconHeads
    save_extra_modules(f"{args.output}/extra_modules.safetensors", inj_L, inj_H,
                         ReconHeads(_D_MIMIR, model.config.hidden_size))
    print(f"saved 'av' adapter + injection adapters -> {args.output}")


if __name__ == "__main__":
    main()
