"""Sanity check for built AV/RL parquets before a long run (CPU, seconds).

    python -m nla.hrm.check_data av_split_facts_v2_mimir.parquet rl_v2_mimir.parquet eval_iid_v2_mimir.parquet \
        --verbalizer-model danish-foundation-models/DFM-Mimir-v1.5 [--compare av_split_facts_v2.parquet ...]

Per file: row count (and against the Qwen build of the same split with --compare, same order), sidecar verbalizer and
marker ids, exactly one [L] and one [H] marker in the chat-rendered prompt with the sidecar's neighbour ids, no NaN/inf in
z, the z norm (about sqrt(d) = 39.2), and for SFT files the response format and how many responses carry the programmatic
facts. Exit code 1 on any hard failure.
"""

import argparse
import sys

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from nla.datagen.storage import LocalStorage
from nla.hrm.build import _INJECT_H_PLACEHOLDER, _INJECT_L_PLACEHOLDER
from nla.hrm.sidecar import read_sidecar


def check_file(path: str, tokenizer, n_rows: int = 2000) -> tuple[list[str], int]:
    problems: list[str] = []
    meta = read_sidecar(LocalStorage(), path)
    tm = meta.tokens
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    names = pf.schema_arrow.names
    print(f"{path}: {total} rows (sidecar says {meta.row_count}), stage={meta.stage}, "
          f"verbalizer={meta.verbalizer_model}")
    if meta.row_count != total:
        problems.append("sidecar row_count != parquet rows")
    if tokenizer.name_or_path != meta.verbalizer_model:
        problems.append(f"built for {meta.verbalizer_model}, checking with {tokenizer.name_or_path}")
    if tm is None:
        return problems + ["no token metadata"], total
    print(f"  markers L={tm.injection_char_L!r} ({tm.injection_token_id_L}) H={tm.injection_char_H!r} ({tm.injection_token_id_H})")
    rows = next(pf.iter_batches(batch_size=n_rows)).to_pylist()
    bad_marker = bad_nb = 0
    for r in rows:
        content = r["prompt"][0]["content"].replace(_INJECT_L_PLACEHOLDER, tm.injection_char_L) \
            .replace(_INJECT_H_PLACEHOLDER, tm.injection_char_H)
        ids = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                            add_generation_prompt=True, return_dict=True)["input_ids"]
        pl, ph = [i for i, t in enumerate(ids) if t == tm.injection_token_id_L], [i for i, t in enumerate(ids) if t == tm.injection_token_id_H]
        if len(pl) != 1 or len(ph) != 1:
            bad_marker += 1
            continue
        if (ids[pl[0] - 1], ids[pl[0] + 1]) != (tm.injection_left_neighbor_id_L, tm.injection_right_neighbor_id_L) or \
           (ids[ph[0] - 1], ids[ph[0] + 1]) != (tm.injection_left_neighbor_id_H, tm.injection_right_neighbor_id_H):
            bad_nb += 1
    print(f"  prompts with != 1 marker per stream: {bad_marker}/{len(rows)}; neighbour mismatch: {bad_nb}/{len(rows)}")
    if bad_marker or bad_nb:
        problems.append("marker/neighbour check failed")
    if "z_L" in names:
        z = torch.tensor([[*r["z_L"]] for r in rows]), torch.tensor([[*r["z_H"]] for r in rows])
        for nm, v in zip(("z_L", "z_H"), z, strict=True):
            fin = bool(torch.isfinite(v).all())
            print(f"  {nm}: dim {v.shape[1]}, finite={fin}, norm mean {v.norm(dim=1).mean():.2f}")
            if not fin:
                problems.append(f"{nm} has NaN/inf")
    if "response" in names:
        resp = [r["response"] for r in rows]
        fmt = sum(r.startswith("<explanation>\n") and r.rstrip().endswith("</explanation>") for r in resp)
        facts = sum('Marked token: "' in r and "Position: " in r for r in resp)
        print(f"  responses in <explanation> format: {fmt}/{len(rows)}; with token+position facts: {facts}/{len(rows)}")
        print(f"  example: {resp[0][:200]!r}")
        if fmt < len(rows):
            problems.append("responses not all in <explanation> format")
    return problems, total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("parquets", nargs="+")
    p.add_argument("--verbalizer-model", required=True)
    p.add_argument("--compare", nargs="*", default=[], help="Qwen builds of the same splits, same order as the parquets: "
                   "row counts must match")
    p.add_argument("--rows", type=int, default=2000)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.verbalizer_model)
    failed = False
    counts = []
    for path in args.parquets:
        problems, total = check_file(path, tok, args.rows)
        counts.append(total)
        for pr in problems:
            print(f"  FAIL: {pr}")
        failed |= bool(problems)
    for path, n in zip(args.compare, counts, strict=False):
        other = pq.ParquetFile(path).metadata.num_rows
        print(f"{path}: {other} rows vs {n} -> {'OK' if other == n else 'MISMATCH'}")
        failed |= other != n
    print("RESULT:", "FAILED" if failed else "ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
