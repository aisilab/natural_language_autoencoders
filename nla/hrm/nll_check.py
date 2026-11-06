"""Does the verbalizer READ the injected vector? Teacher-forced likelihood test.

For each row we compute the AV's mean per-token NLL of the teacher explanation
(`L: <e0>\\nH: <e1>`) under three conditions:
  real      the row's own (z_L, z_H) injected
  shuffled  another row's vectors injected (rolled by one)
A verbalizer that uses the vector assigns the explanation a clearly LOWER NLL with
the real vector (gap = shuffled - real > 0). Gap ~ 0 means the explanations are
predicted from the prompt template and language prior alone: the vector is ignored.
Cheap (forward passes only, no generation) and independent of RL rewards.

Use rows the AV never trained on, e.g. the AR-SFT bucket's explained parquet:
    python -m nla.hrm.nll_check --parquet splits/ar_sft_explained.parquet \\
        --sidecar-from av_sft.parquet --av-ckpt ckpt/av_sft --limit 300
Compare checkpoints (ckpt/av_sft vs ckpt/rl_logr/final) by running it on each.
"""

import argparse

import pyarrow.parquet as pq
import torch

from nla.datagen.storage import LocalStorage
from nla.hrm.build import _INJECT_H_PLACEHOLDER, _INJECT_L_PLACEHOLDER, prefix_token_mask, token_prefix, wrap_lh_explanation
from nla.hrm.devices import default_device, default_dtype
from nla.hrm.model import (
    InjectionAdapter, ReconHeads, build_inputs_embeds, load_extra_modules,
    load_sft_adapter, load_verbalizer, resolve_verbalizer_model, response_token_ce,
)
from nla.hrm.sidecar import read_sidecar
from nla.hrm.train_av_sft import _build_example, _collate

_D_MIMIR = 1536


@torch.no_grad()
def mean_nll(model, examples, inj_L, inj_H, ids_meta, pad_id, device, z_override=None, batch_size=8):
    """(mean per-token NLL over all response tokens, mean NLL over the token-prefix span or nan).
    z_override: (zL_list, zH_list) to inject instead of each row's own vectors."""
    model.set_adapter("av")
    total, n_tok, p_total, p_tok = 0.0, 0, 0.0, 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        input_ids, labels, attn, z_L, z_H = _collate(batch, pad_id, device)
        if z_override is not None:
            z_L = torch.tensor(z_override[0][start : start + batch_size], dtype=torch.float32, device=device)
            z_H = torch.tensor(z_override[1][start : start + batch_size], dtype=torch.float32, device=device)
        pmask = torch.zeros_like(labels)
        for i, e in enumerate(batch):
            pm = e.get("pmask")
            if pm is not None:
                n_prompt = sum(1 for x in e["labels"] if x == -100)
                pmask[i, n_prompt : n_prompt + len(pm)] = torch.tensor(pm, device=device)
        embeds = build_inputs_embeds(model, input_ids, z_L, z_H, inj_L, inj_H, *ids_meta)
        tgt, pm_t = labels[:, 1:], pmask[:, 1:]
        nll = response_token_ce(model, embeds, attn, labels)
        valid = tgt != -100
        total += nll[valid].sum().item()
        n_tok += valid.sum().item()
        p_total += nll[pm_t.bool() & valid].sum().item()
        p_tok += (pm_t.bool() & valid).sum().item()
    return total / max(n_tok, 1), (p_total / p_tok if p_tok else float("nan"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True, help="explained parquet with z_L, z_H, api_explanation_0/1 "
                                                    "(e.g. splits/ar_sft_explained.parquet: rows the AV never trained on)")
    p.add_argument("--sidecar-from", required=True, help="a built av_sft/rl parquet: supplies markers and the actor template")
    p.add_argument("--av-ckpt", required=True, help="dir containing av/ and extra_modules.safetensors "
                                                    "(ckpt/av_sft, or ckpt/rl_logr/final)")
    p.add_argument("--verbalizer-model", default=None, help="default: the backbone recorded in --av-ckpt, else Qwen2.5-1.5B-Instruct")
    p.add_argument("--prefix-token", action="store_true",
                    help="score against targets with the 'Marked token: \"X\".' prefix (use for AV checkpoints trained "
                         "from data built with build.py --prefix-token; needs the context_marked column)")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default=default_dtype())
    args = p.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    meta = read_sidecar(LocalStorage(), args.sidecar_from)
    tm = meta.tokens
    assert tm is not None, f"{args.sidecar_from} has no token metadata"
    ids_meta = (tm.injection_token_id_L, tm.injection_left_neighbor_id_L, tm.injection_right_neighbor_id_L,
                tm.injection_token_id_H, tm.injection_left_neighbor_id_H, tm.injection_right_neighbor_id_H)
    actor_content = meta.prompt_templates["actor"].format(inj_L=_INJECT_L_PLACEHOLDER, inj_H=_INJECT_H_PLACEHOLDER)

    model, tok = load_verbalizer(resolve_verbalizer_model(args.verbalizer_model, args.av_ckpt), device=args.device, torch_dtype=dtype)
    d_verb = model.config.hidden_size
    load_sft_adapter(model, args.av_ckpt, "av")
    inj_L, inj_H = InjectionAdapter(_D_MIMIR, d_verb).to(args.device), InjectionAdapter(_D_MIMIR, d_verb).to(args.device)
    load_extra_modules(f"{args.av_ckpt}/extra_modules.safetensors", inj_L, inj_H, ReconHeads(_D_MIMIR, d_verb))
    model.eval()

    cols = ["z_L", "z_H", "api_explanation_0", "api_explanation_1"] + (["context_marked"] if args.prefix_token else [])
    t = pq.read_table(args.parquet, columns=cols)
    rows = t.to_pylist()[: args.limit]
    assert len(rows) >= 2, "need at least 2 rows for the shuffled condition"
    examples = [
        _build_example(tok, {"prompt": [{"content": actor_content}],
                             "response": wrap_lh_explanation(
                                 (token_prefix(r["context_marked"]) if args.prefix_token else "") + r["api_explanation_0"],
                                 (token_prefix(r["context_marked"]) if args.prefix_token else "") + r["api_explanation_1"]),
                             "z_L": r["z_L"], "z_H": r["z_H"]}, tm.injection_char_L, tm.injection_char_H)
        for r in rows
    ]
    if args.prefix_token:
        for e, r in zip(examples, rows, strict=True):
            resp = wrap_lh_explanation(token_prefix(r["context_marked"]) + r["api_explanation_0"],
                                       token_prefix(r["context_marked"]) + r["api_explanation_1"])
            e["pmask"] = prefix_token_mask(tok, resp, sum(1 for x in e["labels"] if x != -100) - 1)  # minus the EOS label
    rolled = (examples[1:] + examples[:1])
    shuf = ([e["z_L"] for e in rolled], [e["z_H"] for e in rolled])

    real, real_p = mean_nll(model, examples, inj_L, inj_H, ids_meta, tok.pad_token_id, args.device, None, args.batch_size)
    shuffled, shuf_p = mean_nll(model, examples, inj_L, inj_H, ids_meta, tok.pad_token_id, args.device, shuf, args.batch_size)
    print(f"checkpoint {args.av_ckpt}  rows={len(rows)}")
    print(f"  NLL/token  real vector:     {real:.4f}")
    print(f"  NLL/token  shuffled vector: {shuffled:.4f}")
    print(f"  gap (shuffled - real):      {shuffled - real:+.4f}   "
          f"({'vector is used' if shuffled - real > 0.01 else 'vector looks IGNORED'}; 0.01 nats/token = rule of thumb)")
    if args.prefix_token:
        print(f"  marked-token span only:  NLL/token real {real_p:.4f}  shuffled {shuf_p:.4f}  gap {shuf_p - real_p:+.4f}")
        print("    (the span is ~6 tokens per explanation, so it is diluted in the all-token numbers above; "
              "a verbalizer that reads the token shows a large gap HERE)")

if __name__ == "__main__":
    main()
