"""Run the trained verbalizer on a NEW prompt.

    python -m nla.hrm.infer --prompt "Hvad er hovedstaden i Danmark?" --positions -1 \\
        --av-ckpt ckpt/rl/final --ar-ckpt ckpt/rl/final --sidecar-from rl.parquet

Pipeline: Mimir reads the prompt (rendered exactly like at extraction time) and
we capture z_L/z_H at the requested token positions -> the verbalizer writes an
L:/H: explanation for each -> the reconstructor turns each field back into a
vector (per-row FVE if --norm-stats-json is given) -> optionally the Mimir
patch-back judge (--judge).

--positions are token indices into the RENDERED prompt (chat template included);
negative counts from the end (-1 = last prompt token, the "answer position").
Use --show-tokens to print the indexed tokens first. --sidecar-from is any built
parquet (e.g. rl.parquet): its sidecar supplies the injection markers and
templates the checkpoint was trained with — never hardcoded.
"""

import argparse
import json

import torch

from nla.datagen.storage import LocalStorage
from nla.hrm.build import _INJECT_H_PLACEHOLDER, _INJECT_L_PLACEHOLDER
from nla.hrm.devices import default_device, default_dtype
from nla.hrm.eval import generate_and_reconstruct
from nla.hrm.mimir import DEFAULT_MIMIR, HrmStreamCapture, context_marked, load_mimir, render_and_encode_batch
from nla.hrm.model import load_rl_checkpoint, load_verbalizer, resolve_verbalizer_model
from nla.hrm.recon import ReconWeights, recon_loss
from nla.hrm.sidecar import read_sidecar

_D_MIMIR = 1536


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt")
    src.add_argument("--prompt-file", help="text file whose whole content is the prompt")
    p.add_argument("--positions", type=int, nargs="+", default=[-1])
    p.add_argument("--show-tokens", action="store_true", help="print indexed tokens of the rendered prompt and exit")
    p.add_argument("--av-ckpt", help="required unless --show-tokens")
    p.add_argument("--ar-ckpt")
    p.add_argument("--sidecar-from", help="a built parquet (rl.parquet) — supplies markers/templates")
    p.add_argument("--base-model", default=DEFAULT_MIMIR)
    p.add_argument("--verbalizer-model", default=None, help="default: the backbone recorded in --av-ckpt, else Qwen2.5-1.5B-Instruct")
    p.add_argument("--norm-stats-json", default=None)
    p.add_argument("--judge", action="store_true", help="also run the Mimir patch-back judge (KL at the position)")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--w-sum", type=float, default=1.0)
    p.add_argument("--w-comp", type=float, default=0.25)
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default=default_dtype())
    p.add_argument("--output", default=None, help="also write the results as JSON")
    args = p.parse_args()

    prompt = args.prompt if args.prompt is not None else open(args.prompt_file).read().strip()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    mimir, mimir_tok = load_mimir(args.base_model, device=args.device, torch_dtype=dtype)
    batch = render_and_encode_batch(mimir_tok, [prompt], device=args.device)
    length = batch.lengths[0]
    if args.show_tokens:
        for i, t in enumerate(mimir_tok.convert_ids_to_tokens(batch.input_ids[0, :length])):
            print(f"{i:4d} {t!r}")
        return
    assert args.av_ckpt and args.ar_ckpt and args.sidecar_from, "--av-ckpt, --ar-ckpt and --sidecar-from are required"

    positions = [pos % length for pos in args.positions]
    with HrmStreamCapture(mimir) as cap, torch.no_grad():
        mimir(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
              token_type_ids=batch.token_type_ids, use_cache=False, logits_to_keep=1)
    cap.verify_call_counts()
    cap.verify_sum_identity(atol=1e-2 if dtype == torch.bfloat16 else 1e-3, rtol=1e-2 if dtype == torch.bfloat16 else 1e-3)
    assert cap.z_L is not None and cap.z_H is not None

    meta = read_sidecar(LocalStorage(), args.sidecar_from)
    tm = meta.tokens
    assert tm is not None, f"{args.sidecar_from} has no token metadata — pass a parquet produced by build.py"
    actor_content = meta.prompt_templates["actor"].format(inj_L=_INJECT_L_PLACEHOLDER, inj_H=_INJECT_H_PLACEHOLDER)
    prompt_ids = batch.input_ids[0, :length].tolist()
    rows = [{
        "prompt": [{"role": "user", "content": actor_content}],
        "z_L": cap.z_L[0, pos].tolist(), "z_H": cap.z_H[0, pos].tolist(),
        "position": pos, "prompt_len": length, "prompt_ids": prompt_ids,
        "context_marked": context_marked(mimir_tok, batch.input_ids[0], pos, length),
    } for pos in positions]
    if not args.judge:
        del mimir, cap
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    model, tok = load_verbalizer(resolve_verbalizer_model(args.verbalizer_model, args.av_ckpt), device=args.device, torch_dtype=dtype)
    inj_L, inj_H, heads = load_rl_checkpoint(model, args.av_ckpt, args.ar_ckpt, _D_MIMIR,
                                             model.config.hidden_size, args.device)
    ids_meta = (tm.injection_token_id_L, tm.injection_left_neighbor_id_L, tm.injection_right_neighbor_id_L,
                tm.injection_token_id_H, tm.injection_left_neighbor_id_H, tm.injection_right_neighbor_id_H)
    results = generate_and_reconstruct(model, tok, rows, tm.injection_char_L, tm.injection_char_H, inj_L, inj_H,
                                       heads, ids_meta, meta.prompt_templates["critic"], args.device,
                                       args.max_new_tokens)

    weights = ReconWeights(w_sum=args.w_sum, w_comp=args.w_comp)
    mean_mse = json.load(open(args.norm_stats_json))["mean_mse"] if args.norm_stats_json else None
    out = []
    for r in results:
        row = r["row"]
        rec = {"position": row["position"], "context_marked": row["context_marked"], "completion": r["text"],
               "L_field": r["parsed"][0] if r["parsed"] else None, "H_field": r["parsed"][1] if r["parsed"] else None}
        if r["parsed"]:
            loss = recon_loss(r["z_L_hat"].unsqueeze(0), r["z_H_hat"].unsqueeze(0),
                              torch.tensor(row["z_L"]).unsqueeze(0), torch.tensor(row["z_H"]).unsqueeze(0), weights)
            rec["mse"] = {"sum": loss.mse_sum.item(), "L": loss.mse_L.item(), "H": loss.mse_H.item()}
            if mean_mse:
                rec["fve"] = loss.fve(mean_mse["sum"], mean_mse["L"], mean_mse["H"])
        out.append(rec)

    if args.judge:
        from nla.hrm.judge import run_judge
        ok = [r for r in results if r["parsed"]]
        if ok:
            j = run_judge(mimir, mimir_tok, [r["row"] for r in ok],
                          torch.stack([r["z_L_hat"] for r in ok]).float().cpu(),
                          torch.stack([r["z_H_hat"] for r in ok]).float().cpu(), args.device)
            k = 0
            for rec, r in zip(out, results, strict=True):
                if r["parsed"]:
                    rec["judge_kl_patch_pos"] = j["primary"]["kl_pred_at_patch"][k]
                    rec["judge_kl_last_pos"] = j["primary"]["kl_pred_at_last"][k]
                    k += 1

    for rec in out:
        print("=" * 100)
        print(f"position {rec['position']}:  {rec['context_marked'][-300:]!r}")
        if rec["L_field"] is None:
            print("  [malformed completion]\n  " + rec["completion"][:600])
            continue
        print(f"  L: {rec['L_field']}\n  H: {rec['H_field']}")
        if "fve" in rec:
            print("  FVE  " + "  ".join(f"{k}={v:.3f}" for k, v in rec["fve"].items()))
        else:
            print("  mse  " + "  ".join(f"{k}={v:.2e}" for k, v in rec["mse"].items()))
        if "judge_kl_patch_pos" in rec:
            print(f"  judge KL: at position={rec['judge_kl_patch_pos']:.3f}  at last token={rec['judge_kl_last_pos']:.3f}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
