"""Full evaluation of a trained AV+AR pair (post-SFT or post-RL — pass the
same checkpoint dir for both --av-ckpt/--ar-ckpt when evaluating an RL
checkpoint, since train_rl.py saves both adapters into one directory).

Produces, per named eval split (repeat `--eval-parquet name=path`, e.g.
`iid=.../eval_iid.parquet ood=.../eval_ood.parquet`):
  - format_rate: fraction of completions that parse as L:/H: fields
  - per-term FVE (sum/L/H), relative to norm_stats.json's mean-predictor
    baseline if given, else relative to THIS split's own mean predictor
  - field monitoring: L/H field length ratio, empty-field rate (folded into
    format_rate — parse_fields rejects empty fields), L-vs-H token-overlap
    (duplication smell — near-1.0 overlap means the two fields are saying
    the same thing, i.e. no stream-specific differentiation)
  - cancellation stats for (ẑ_L, ẑ_H) vs the stored (z_L, z_H) — comparable
    to `diagnostics.py`'s numbers on the raw data
  - broken down by is_last_prompt_pos vs other

With --run-judge (+ --base-model), also runs the Mimir patch-back judge
(`judge.py`) on the SAME rollouts and merges its KL numbers into the report
— the two are naturally sequential (judge needs the ẑ_L/ẑ_H this script just
produced) so doing both in one process avoids a JSON round-trip.
"""

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from nla.hrm.devices import default_device, default_dtype
from nla.hrm.build import _INJECT_H_PLACEHOLDER, _INJECT_L_PLACEHOLDER
from nla.hrm.diagnostics import compute_stream_stats
from nla.hrm.model import critic_hidden, generate_kwargs, load_rl_checkpoint, load_verbalizer, resolve_verbalizer_model
from nla.hrm.recon import ReconWeights, parse_fields, recon_loss
from nla.hrm.sidecar import read_sidecar
from nla.datagen.storage import LocalStorage

_D_MIMIR = 1536


@torch.no_grad()
def generate_and_reconstruct(model, tokenizer, rows, inj_l_char, inj_h_char, inj_L, inj_H, heads, ids_meta,
                               critic_template, device, max_new_tokens=300, batch_size=16, shuffle_vectors=False, within_dataset=False,
                               swap_streams=False, best_of=1, sample_temperature=1.0, weights=None, split_av=False,
                               critic_readout="last"):
    from nla.hrm.model import build_inputs_embeds

    # Best-of-N: candidate 0 is the greedy completion, candidates 1..N-1 are temperature samples of the same prompt; the
    # one whose AR reconstruction is closest to the row's GOLD vectors (recon loss) is kept. Uses the true vector, so the
    # selected FVE is optimistic (selection on the metric); token accuracy and text checks are independent of it.
    assert not split_av or (best_of == 1 and not swap_streams and not shuffle_vectors), (
        "--split-av is not combined with best-of / swap / shuffle")
    assert best_of == 1 or (weights is not None and not swap_streams and not shuffle_vectors), (
        "best-of-N needs recon weights and is not combined with the shuffle / swap controls")

    # Swap control: each stream keeps its own adapter but is written into the OTHER stream's slot (z_L's adapted vector at
    # the [H] marker, z_H's at the [L] marker) by swapping the marker ids. If the L/H text difference follows the slot the
    # text is a position artefact; if it follows the vector it is stream-specific. Scored against the original gold.
    assert not (swap_streams and shuffle_vectors), "--swap-streams and --shuffle-vectors are separate controls"
    if swap_streams:
        ids_meta = ids_meta[3:] + ids_meta[:3]
    out = []
    # Control: inject the vectors of a DIFFERENT row (rolled by one) while still scoring against the
    # original row's gold vectors. If FVE does not drop, the verbalizer is ignoring the vector.
    inject_rows = rows[1:] + rows[:1] if shuffle_vectors else rows
    if shuffle_vectors and within_dataset:
        # roll within each dataset, so the injected vector still comes from the same SOURCE: removes the dataset-level
        # information that a plain shuffle leaves intact. Datasets with a single row cannot be shuffled (kept as is).
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(r.get("dataset"), []).append(i)
        inject_rows = list(rows)
        for idxs in groups.values():
            for a, b in zip(idxs, idxs[1:] + idxs[:1], strict=True):
                inject_rows[a] = rows[b]
        n_same = sum(len(v) == 1 for v in groups.values())
        if n_same:
            print(f"  [shuffle] {n_same} dataset(s) with a single row were not shuffled")
    for start in tqdm(range(0, len(rows), batch_size), desc="rollout"):
        batch_rows = rows[start : start + batch_size]
        inj_rows = inject_rows[start : start + batch_size]
        if split_av:
            cand_texts = [_split_greedy(model, tokenizer, batch_rows, inj_l_char, inj_h_char, inj_L, inj_H, ids_meta,
                                        device, max_new_tokens)]
        else:
            contents = [r["prompt"][0]["content"].replace(_INJECT_L_PLACEHOLDER, inj_l_char)
                        .replace(_INJECT_H_PLACEHOLDER, inj_h_char) for r in batch_rows]
            tokenizer.padding_side = "left"
            enc = tokenizer.apply_chat_template(
                [[{"role": "user", "content": c}] for c in contents],
                tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True,
            )
            input_ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            z_L = torch.tensor([r["z_L"] for r in inj_rows], dtype=torch.float32, device=device)
            z_H = torch.tensor([r["z_H"] for r in inj_rows], dtype=torch.float32, device=device)
            embeds = build_inputs_embeds(model, input_ids, z_L, z_H, inj_L, inj_H, *ids_meta)
            model.set_adapter("av")
            gen_ids = model.generate(inputs_embeds=embeds, attention_mask=attn, max_new_tokens=max_new_tokens,
                                       do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                       **generate_kwargs(model, attn))
            cand_texts = [tokenizer.batch_decode(gen_ids, skip_special_tokens=True)]
            for _ in range(best_of - 1):
                sampled = model.generate(inputs_embeds=embeds, attention_mask=attn, max_new_tokens=max_new_tokens,
                                           do_sample=True, temperature=sample_temperature, top_p=1.0, top_k=0,
                                           pad_token_id=tokenizer.pad_token_id, **generate_kwargs(model, attn))
                cand_texts.append(tokenizer.batch_decode(sampled, skip_special_tokens=True))


        def _reconstruct(texts):
            parsed = [parse_fields(t) for t in texts]
            ok_idx = [i for i, p in enumerate(parsed) if p is not None]
            zL_hat = [None] * len(batch_rows)
            zH_hat = [None] * len(batch_rows)
            if ok_idx:
                model.set_adapter("ar")
                l_texts = [critic_template.format(explanation=parsed[i][0]) for i in ok_idx]
                h_texts = [critic_template.format(explanation=parsed[i][1]) for i in ok_idx]
                c_enc = tokenizer(l_texts + h_texts, return_tensors="pt", padding=True,
                                    add_special_tokens=False, truncation=True, max_length=512)
                c_ids, c_attn = c_enc["input_ids"].to(device), c_enc["attention_mask"].to(device)
                h_last = critic_hidden(model, c_ids, c_attn, critic_readout)
                n = len(ok_idx)
                zL_pred, zH_pred = heads.forward_L(h_last[:n]), heads.forward_H(h_last[n:])
                for j, i in enumerate(ok_idx):
                    zL_hat[i] = zL_pred[j].float().cpu()  # CPU: gold vectors live on CPU, avoids device mixing
                    zH_hat[i] = zH_pred[j].float().cpu()
            return parsed, zL_hat, zH_hat

        cands = [(texts, *_reconstruct(texts)) for texts in cand_texts]
        for i, r in enumerate(batch_rows):
            chosen, bo = 0, None
            if best_of > 1:
                gold_L, gold_H = torch.tensor(r["z_L"]).unsqueeze(0), torch.tensor(r["z_H"]).unsqueeze(0)
                losses = [None if c[1][i] is None else
                          recon_loss(c[2][i].unsqueeze(0), c[3][i].unsqueeze(0), gold_L, gold_H, weights).total.item()
                          for c in cands]
                valid = [k for k, v in enumerate(losses) if v is not None]
                chosen = min(valid, key=lambda k: losses[k]) if valid else 0
                bo = {"chosen": chosen, "losses": losses,
                      "quote_ok": [quote_match(c[1][i], r.get("context_marked")) for c in cands]}
            texts_c, parsed_c, zl_c, zh_c = cands[chosen]
            out.append({
                "row": r, "text": texts_c[i], "parsed": parsed_c[i],
                "z_L_hat": zl_c[i], "z_H_hat": zh_c[i], **({"bo": bo} if bo else {}),
            })
    return out


@torch.no_grad()
def _split_greedy(model, tokenizer, batch_rows, inj_l_char, inj_h_char, inj_L, inj_H, ids_meta, device, max_new_tokens):
    """Split AV: greedy L call (z_H zeroed) and H call (z_L zeroed), joined into one completion per row."""
    from nla.hrm.model import build_inputs_embeds
    from nla.hrm.split_av import join_split, tag_content

    outs = {}
    for described, zeroed in (("L", "H"), ("H", "L")):
        contents = [tag_content(r["prompt"][0]["content"].replace(_INJECT_L_PLACEHOLDER, inj_l_char)
                                .replace(_INJECT_H_PLACEHOLDER, inj_h_char), described) for r in batch_rows]
        tokenizer.padding_side = "left"
        enc = tokenizer.apply_chat_template(
            [[{"role": "user", "content": c}] for c in contents],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True,
        )
        input_ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        z_L = torch.tensor([r["z_L"] for r in batch_rows], dtype=torch.float32, device=device)
        z_H = torch.tensor([r["z_H"] for r in batch_rows], dtype=torch.float32, device=device)
        if zeroed == "L":
            z_L = torch.zeros_like(z_L)
        else:
            z_H = torch.zeros_like(z_H)
        embeds = build_inputs_embeds(model, input_ids, z_L, z_H, inj_L, inj_H, *ids_meta)
        model.set_adapter("av")
        gen = model.generate(inputs_embeds=embeds, attention_mask=attn, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id, **generate_kwargs(model, attn))
        outs[described] = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [join_split(a, b) for a, b in zip(outs["L"], outs["H"], strict=True)]


def best_of_report(name: str, results: list[dict]) -> None:
    """Token accuracy (non-last rows) of the greedy candidate, the selected one, a random candidate and the oracle, plus the
    mean reconstruction loss of greedy vs selected: does reconstruction consistency pick the right explanation?"""
    rows = [r for r in results if r.get("bo") and not r["row"].get("is_last_prompt_pos")]
    if not rows:
        return
    n_c = len(rows[0]["bo"]["quote_ok"])
    ok = lambda v: v is True  # noqa: E731
    greedy = sum(ok(r["bo"]["quote_ok"][0]) for r in rows) / len(rows)
    chosen = sum(ok(r["bo"]["quote_ok"][r["bo"]["chosen"]]) for r in rows) / len(rows)
    rand = sum(sum(ok(v) for v in r["bo"]["quote_ok"]) / n_c for r in rows) / len(rows)
    oracle = sum(any(ok(v) for v in r["bo"]["quote_ok"]) for r in rows) / len(rows)
    lg = [r["bo"]["losses"][0] for r in rows if r["bo"]["losses"][0] is not None]
    lc = [r["bo"]["losses"][r["bo"]["chosen"]] for r in rows if r["bo"]["losses"][r["bo"]["chosen"]] is not None]
    print(f"[{name}] best-of-{n_c}, non-last n={len(rows)}: marked token correct: greedy {greedy:.0%}, selected {chosen:.0%}, "
          f"random candidate {rand:.0%}, oracle (any) {oracle:.0%}; mean recon loss greedy {sum(lg) / len(lg):.3e} -> "
          f"selected {sum(lc) / len(lc):.3e}")


def _dump_samples(path: str, split: str, results: list[dict], weights: ReconWeights, mean_mse: dict | None) -> None:
    with open(path, "a") as f:
        for r in results:
            row = r["row"]
            rec = {"split": split, "dataset": row.get("dataset"), "position": row.get("position"),
                   "prompt_len": row.get("prompt_len"),
                   "pos_correct": list(position_match_fields(r["parsed"], row)),
                   "is_last_prompt_pos": row.get("is_last_prompt_pos"),
                   "context_marked": row.get("context_marked"), "completion": r["text"],
                   "L_field": r["parsed"][0] if r["parsed"] else None,
                   "H_field": r["parsed"][1] if r["parsed"] else None,
                   "marked_token_quote_correct": quote_match(r["parsed"], row.get("context_marked")),
                   "grounding": grounding(r["parsed"], row.get("context_marked"))}
            if r.get("bo"):
                rec["bo"] = r["bo"]
            if r["parsed"] is not None:
                z_L = torch.tensor(row["z_L"]).unsqueeze(0)
                z_H = torch.tensor(row["z_H"]).unsqueeze(0)
                loss = recon_loss(r["z_L_hat"].unsqueeze(0), r["z_H_hat"].unsqueeze(0), z_L, z_H, weights)
                rec["mse"] = {"sum": loss.mse_sum.item(), "L": loss.mse_L.item(), "H": loss.mse_H.item()}
                if mean_mse:
                    rec["fve"] = {k: v for k, v in loss.fve(mean_mse["sum"], mean_mse["L"], mean_mse["H"]).items()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


_QUOTE_RE = re.compile(r"""marked token\s*[:\-]?\s*[“"'`‘]([^”"'`’]{1,40})[”"'`’]""", re.I)


def shuffle_permutation(rows: list[dict], within_dataset: bool) -> list[int]:
    """perm[i] = index of the row whose vector is used for row i (rolled by one; within each dataset if asked).
    Single-row datasets map to themselves; callers should report how many rows are unshuffled."""
    n = len(rows)
    if not within_dataset:
        return [(i + 1) % n for i in range(n)]
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r.get("dataset"), []).append(i)
    perm = list(range(n))
    for idxs in groups.values():
        for a, b in zip(idxs, idxs[1:] + idxs[:1], strict=True):
            perm[a] = b
    return perm


_POS_RE = re.compile(r"Position:\s*(\d)\s*of\s*5", re.I)


def position_match_fields(fields: tuple[str, str] | None, row: dict) -> tuple[bool | None, bool | None]:
    """Split-AV position fact: does each field's `Position: K of 5` equal the true fifth of the prompt? None = no fact."""
    if fields is None or not row.get("prompt_len") or row.get("position") is None:
        return None, None
    true = min(int(5 * row["position"] / row["prompt_len"]), 4) + 1
    out = []
    for f in fields:
        m = _POS_RE.search(f)
        out.append(int(m.group(1)) == true if m else None)
    return out[0], out[1]


def marked_token(context_marked: str | None) -> str | None:
    m = re.search(r"⟦(.*?)⟧", context_marked or "", re.S)
    return m.group(1) if m else None


def quote_match_fields(fields: tuple[str, str] | None, context_marked: str | None) -> tuple[bool | None, bool | None]:
    """Per-field version of `quote_match`: (L_result, H_result), each True/False/None (field has no quoted token)."""
    tok = marked_token(context_marked)
    if tok is None or fields is None:
        return None, None
    want = (tok.strip() or json.dumps(tok)[1:-1]).lower()
    out = []
    for f in fields:
        m = _QUOTE_RE.search(f)
        out.append(m.group(1).strip().lower() == want if m else None)
    return out[0], out[1]


def quote_match(fields: tuple[str, str] | None, context_marked: str | None) -> bool | None:
    """Text-level faithfulness check: SFT explanations tend to open with `The marked token "X" ...`.
    True/False = the first quoted marked token equals / differs from the real ⟦marked⟧ token (case and
    surrounding whitespace ignored); None = no quoted marked token (or no marker) in either field."""
    tok = marked_token(context_marked)
    if tok is None or fields is None:
        return None
    want = (tok.strip() or json.dumps(tok)[1:-1]).lower()  # whitespace-only tokens are shown JSON-escaped (build.py)
    for f in fields:
        m = _QUOTE_RE.search(f)
        if m:
            return m.group(1).strip().lower() == want
    return None


_GENERIC_CAPS = {"marked", "the", "key", "narrative", "story", "state", "task", "feature", "children", "danish", "english",
                 "question", "answer", "yes", "no", "token", "context", "position", "marked", "this", "that", "these", "both"}
_QUOTED_RE = re.compile(r"""[“"]([^”"\n]{3,80})[”"]""")


def _context_text(context_marked: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"⟦|⟧", "", context_marked or "")).lower()


def grounding(fields: tuple[str, str] | None, context_marked: str | None) -> dict | None:
    """Text-level hallucination check against the prompt: (a) quoted spans of an explanation that occur VERBATIM in the
    context (the marked-token quote is excluded), (b) capitalised words not at a sentence start (names like 'Leo') that
    occur in the context. Both are lower bounds on confabulation: a name can be correct without being in the text, and
    generic capitalised words ('Danish') are filtered by a short stoplist. Returns None without fields/context."""
    if fields is None or not context_marked:
        return None
    ctx = _context_text(context_marked)
    quoted_total = quoted_ok = caps_total = caps_ok = 0
    for f in fields:
        body = re.sub(r'^\s*Marked token: "[^"]*"\.\s*', "", f)
        for q in _QUOTED_RE.findall(body):
            quoted_total += 1
            quoted_ok += re.sub(r"\s+", " ", q).strip().lower() in ctx
        for m in re.finditer(r"(?<=[a-z,;:] )([A-Z][a-zæøå]{2,})\b", body):
            w = m.group(1)
            if w.lower() in _GENERIC_CAPS:
                continue
            caps_total += 1
            caps_ok += w.lower() in ctx
    return {"quoted_total": quoted_total, "quoted_ok": quoted_ok, "caps_total": caps_total, "caps_ok": caps_ok}


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def summarize(results: list[dict], weights: ReconWeights, mean_mse: dict | None) -> dict:
    n = len(results)
    ok = [r for r in results if r["parsed"] is not None]
    format_rate = len(ok) / n if n else 0.0

    fve = {"sum": [], "L": [], "H": []}
    overlaps, len_l, len_h = [], [], []
    cancel_hat, cos_hat = [], []
    geo = {"cos_s": [], "cos_L": [], "cos_H": [], "norm_ratio_s": []}
    quote = [quote_match(r["parsed"], r["row"].get("context_marked")) for r in ok]
    quoted = [q for q in quote if q is not None]
    pos_fields = [position_match_fields(r["parsed"], r["row"]) for r in ok]
    pL = [float(a) for a, _ in pos_fields if a is not None]
    pH = [float(b) for _, b in pos_fields if b is not None]
    per_field = [quote_match_fields(r["parsed"], r["row"].get("context_marked")) for r in ok]
    qL = [float(a) for a, _ in per_field if a is not None]
    qH = [float(b) for _, b in per_field if b is not None]
    gr = [g for g in (grounding(r["parsed"], r["row"].get("context_marked")) for r in ok) if g is not None]
    q_tot, q_ok = sum(g["quoted_total"] for g in gr), sum(g["quoted_ok"] for g in gr)
    c_tot, c_ok = sum(g["caps_total"] for g in gr), sum(g["caps_ok"] for g in gr)
    for r in ok:
        z_L = torch.tensor(r["row"]["z_L"], dtype=torch.float32).unsqueeze(0)
        z_H = torch.tensor(r["row"]["z_H"], dtype=torch.float32).unsqueeze(0)
        zL_hat, zH_hat = r["z_L_hat"].unsqueeze(0), r["z_H_hat"].unsqueeze(0)
        loss = recon_loss(zL_hat, zH_hat, z_L, z_H, weights)
        if mean_mse:
            f = loss.fve(mean_mse["sum"], mean_mse["L"], mean_mse["H"])
            fve["sum"].append(f["fve_sum"])
            fve["L"].append(f["fve_L"])
            fve["H"].append(f["fve_H"])
        l_field, h_field = r["parsed"]
        overlaps.append(_jaccard(l_field, h_field))
        len_l.append(len(l_field.split()))
        len_h.append(len(h_field.split()))
        cos = torch.nn.functional.cosine_similarity
        geo["cos_s"].append(cos(zL_hat + zH_hat, z_L + z_H).item())
        geo["cos_L"].append(cos(zL_hat, z_L).item())
        geo["cos_H"].append(cos(zH_hat, z_H).item())
        geo["norm_ratio_s"].append(((zL_hat + zH_hat).norm() / (z_L + z_H).norm()).item())
        stats = compute_stream_stats(zL_hat, zH_hat)
        cancel_hat.append(stats["cancellation"].item())
        cos_hat.append(stats["cos_LH"].item())

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {
        "n": n, "n_parsed": len(ok), "format_rate": format_rate,
        "fve_sum_mean": _mean(fve["sum"]), "fve_L_mean": _mean(fve["L"]), "fve_H_mean": _mean(fve["H"]),
        "lh_jaccard_mean": _mean(overlaps),  # near 1.0 = fields are near-duplicates (no differentiation)
        "l_field_words_mean": _mean(len_l), "h_field_words_mean": _mean(len_h),
        "quote_rate": len(quoted) / len(ok) if ok else float("nan"),  # share of explanations that quote a marked token
        "quote_match_rate": _mean([float(q) for q in quoted]),        # of those, share quoting the REAL token
        "pos_match_rate_L": _mean(pL), "pos_match_rate_H": _mean(pH), "pos_rate": len(pL) / len(ok) if ok else float("nan"),
        "quote_match_rate_L": _mean(qL), "quote_match_rate_H": _mean(qH),  # per field, over fields that quote a token
        "quote_rate_L": len(qL) / len(ok) if ok else float("nan"), "quote_rate_H": len(qH) / len(ok) if ok else float("nan"),
        "grounded_quote_rate": q_ok / q_tot if q_tot else float("nan"),   # quoted spans found verbatim in the prompt
        "grounded_quote_spans": q_tot,
        "grounded_name_rate": c_ok / c_tot if c_tot else float("nan"),    # capitalised names found in the prompt
        "grounded_name_words": c_tot,
        **{f"geo_{k}_mean": _mean(v) for k, v in geo.items()},  # cosine to gold / norm ratio of the sum
        "cancellation_hat_mean": _mean(cancel_hat), "cos_LH_hat_mean": _mean(cos_hat),
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-parquet", action="append", nargs="+", required=True,
                    help="name=path; several per flag (--eval-parquet a=x.parquet b=y.parquet) or repeat the flag")
    p.add_argument("--av-ckpt", required=True)
    p.add_argument("--ar-ckpt", required=True)
    p.add_argument("--verbalizer-model", default=None)
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default=default_dtype())
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--w-sum", type=float, default=1.0)
    p.add_argument("--w-comp", type=float, default=0.25)
    p.add_argument("--norm-stats-json", default=None)
    p.add_argument("--reuse-generations", action="store_true",
                    help="reuse <output>.gen_<split>.pt from an earlier run instead of regenerating "
                         "(generation is the slow part; it is always saved there right after it finishes)")
    p.add_argument("--shuffle-vectors", action="store_true",
                    help="CONTROL: inject another row's vectors (still score against the original gold). "
                         "Compare its FVE to the normal run: no drop = the verbalizer ignores the vector")
    p.add_argument("--shuffle-within-dataset", action="store_true",
                    help="with --shuffle-vectors: swap vectors only among rows of the SAME dataset (stronger control: "
                         "removes the dataset-level information a plain shuffle keeps)")
    p.add_argument("--limit", type=int, default=None,
                    help="evaluate only a random sample of N rows per split (quick check; seeded, reproducible)")
    p.add_argument("--dump-samples", default=None,
                    help="write one JSON line per eval row (context, completion, L/H fields, per-row FVE) to this path")
    p.add_argument("--mean-from", default=None,
                    help="parquet with z_L/z_H (e.g. rl.parquet) to compute the mean-ablation baselines for the judge "
                         "(adds 'fraction of KL recovered'); strongly recommended with --run-judge")
    p.add_argument("--swap-streams", action="store_true",
                   help="control: write z_L's (adapted) vector into the [H] slot and z_H's into the [L] slot; generations are "
                        "cached under a separate '_swap' name. Use with --dump-samples to see whether the L/H text follows "
                        "the slot or the vector")
    p.add_argument("--best-of", type=int, default=1,
                   help="sample N-1 extra explanations per row (greedy is candidate 0) and keep the one whose AR "
                        "reconstruction is closest to the gold vectors; FVE is then selection-biased, token accuracy "
                        "and text are not. Cached as <output>.gen_<split>_bo<N>.pt")
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--critic-readout", choices=["last", "legacy"], default="last",
                   help="token the AR reads out: 'last' = last attended token (correct); 'legacy' = attn.sum-1, wrong for the "
                        "left-padded batches used here: use it only to reproduce eval numbers logged before 2026-10-11")
    p.add_argument("--split-av", action="store_true",
                   help="the AV checkpoint is a split AV (nla/hrm/split_av.py): greedy L call (z_H zeroed) and H call "
                        "(z_L zeroed); cached as <output>.gen_<split>_split.pt")
    p.add_argument("--run-judge", action="store_true")
    p.add_argument("--judge-shuffle", choices=["roll", "within-dataset"], default=None,
                   help="with --run-judge: ALSO run the judge with each row patched by ANOTHER row's reconstruction "
                        "(rolled by one, or rolled within each dataset); stored as report['judge_shuffled']. Control for "
                        "'any plausible vector beats the mean vector'.")
    p.add_argument("--base-model", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    weights = ReconWeights(w_sum=args.w_sum, w_comp=args.w_comp)
    if args.best_of > 1:
        torch.manual_seed(0)  # reproducible candidate sampling
    mean_mse = json.load(open(args.norm_stats_json))["mean_mse"] if args.norm_stats_json else None

    verbalizer_model = resolve_verbalizer_model(args.verbalizer_model, args.av_ckpt)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    model, tokenizer = load_verbalizer(verbalizer_model, device=args.device, torch_dtype=dtype)
    d_verb = model.config.hidden_size
    inj_L, inj_H, heads = load_rl_checkpoint(model, args.av_ckpt, args.ar_ckpt, _D_MIMIR, d_verb, args.device)

    if args.dump_samples:
        open(args.dump_samples, "w").close()  # truncate once; each split appends
    report: dict = {"splits": {}}
    all_results: dict[str, list[dict]] = {}
    for spec in [spec for group in args.eval_parquet for spec in group]:
        name, path = spec.split("=", 1)
        rows = pq.read_table(path).to_pylist()
        if args.limit is not None and len(rows) > args.limit:
            import random
            rows = random.Random(0).sample(rows, args.limit)
        meta = read_sidecar(LocalStorage(), path)
        tm = meta.tokens
        assert tm is not None
        ids_meta = (tm.injection_token_id_L, tm.injection_left_neighbor_id_L, tm.injection_right_neighbor_id_L,
                    tm.injection_token_id_H, tm.injection_left_neighbor_id_H, tm.injection_right_neighbor_id_H)
        gen_path = f"{args.output}.gen_{name}{('_shufds' if args.shuffle_within_dataset else '_shuf') if args.shuffle_vectors else ''}{'_swap' if args.swap_streams else ''}{f'_bo{args.best_of}' if args.best_of > 1 else ''}{'_split' if args.split_av else ''}{'_legacyreadout' if args.critic_readout == 'legacy' else ''}.pt"
        if args.reuse_generations and Path(gen_path).exists():
            saved = torch.load(gen_path)
            assert len(saved) == len(rows), f"{gen_path} has {len(saved)} rows, split now has {len(rows)}; drop --reuse-generations"
            results = [{"row": r, **g} for r, g in zip(rows, saved, strict=True)]
            print(f"[{name}] reused {len(results)} saved generations from {gen_path}")
        else:
            results = generate_and_reconstruct(model, tokenizer, rows, tm.injection_char_L, tm.injection_char_H,
                                                 inj_L, inj_H, heads, ids_meta, meta.prompt_templates["critic"], args.device,
                                                 args.max_new_tokens, args.batch_size, args.shuffle_vectors, args.shuffle_within_dataset,
                                                 args.swap_streams, args.best_of, args.sample_temperature, weights, args.split_av,
                                                 args.critic_readout)
            torch.save([{k: v for k, v in r.items() if k != "row"} for r in results], gen_path)
        best_of_report(name, results)
        all_results[name] = results
        if args.dump_samples:
            _dump_samples(args.dump_samples, name, results, weights, mean_mse)
        summary = summarize(results, weights, mean_mse)
        last = [r for r in results if r["row"].get("is_last_prompt_pos")]
        other = [r for r in results if not r["row"].get("is_last_prompt_pos")]
        report["splits"][name] = {
            "overall": summary,
            "last_prompt_pos": summarize(last, weights, mean_mse) if last else None,
            "other_positions": summarize(other, weights, mean_mse) if other else None,
        }
        if other:
            o = report["splits"][name]["other_positions"]
            print(f"      non-last positions only (n={o['n']}): fve_sum={o['fve_sum_mean']:.3f} fve_L={o['fve_L_mean']:.3f} "
                  f"fve_H={o['fve_H_mean']:.3f}; quoted {o['quote_rate']:.0%}, correct {o['quote_match_rate']:.0%}  "
                  f"(the last prompt position is always the same newline token, so read quote accuracy HERE)")
        print(f"[{name}] n={summary['n']} format_rate={summary['format_rate']:.2%} "
              f"fve_sum={summary['fve_sum_mean']:.3f} fve_L={summary['fve_L_mean']:.3f} "
              f"fve_H={summary['fve_H_mean']:.3f} lh_jaccard={summary['lh_jaccard_mean']:.3f}\n"
              f"      geometry vs gold: cos(sum)={summary['geo_cos_s_mean']:.3f} cos(L)={summary['geo_cos_L_mean']:.3f} "
              f"cos(H)={summary['geo_cos_H_mean']:.3f} |sum_hat|/|sum|={summary['geo_norm_ratio_s_mean']:.2f}\n"
              f"      marked-token quote: quoted in {summary['quote_rate']:.0%} of explanations, "
              f"correct token in {summary['quote_match_rate']:.0%} of those\n"
              f"      per field: L correct {summary['quote_match_rate_L']:.0%} (quoted {summary['quote_rate_L']:.0%}), "
              f"H correct {summary['quote_match_rate_H']:.0%} (quoted {summary['quote_rate_H']:.0%})\n"
              + (f"      position fact (split AV, chance 20%): L {summary['pos_match_rate_L']:.0%}, "
                 f"H {summary['pos_match_rate_H']:.0%} (stated in {summary['pos_rate']:.0%})\n"
                 if summary["pos_rate"] > 0 else "") +
              f"      grounding in the prompt text: {summary['grounded_quote_rate']:.0%} of {summary['grounded_quote_spans']} quoted spans "
              f"verbatim in context; {summary['grounded_name_rate']:.0%} of {summary['grounded_name_words']} capitalised names in context "
              f"(low = confabulated details)")

    if args.run_judge:
        from nla.hrm.judge import run_judge
        from nla.hrm.mimir import DEFAULT_MIMIR, load_mimir
        mimir_model, mimir_tok = load_mimir(args.base_model or DEFAULT_MIMIR, device=args.device, torch_dtype=dtype)
        report["judge"] = {}
        mean_s = mean_zH = None
        if args.mean_from:
            t = pq.read_table(args.mean_from, columns=["z_L", "z_H"]).slice(0, 20000)
            zl = torch.tensor(t.column("z_L").to_pylist(), dtype=torch.float32)
            zh = torch.tensor(t.column("z_H").to_pylist(), dtype=torch.float32)
            mean_s, mean_zH = (zl + zh).mean(0), zh.mean(0)
        else:
            print("  [judge] no --mean-from: skipping mean-ablation baselines (no 'fraction of KL recovered')")
        for name, results in all_results.items():
            ok = [r for r in results if r["parsed"] is not None]
            if not ok:
                continue
            judge_rows = [r["row"] for r in ok]
            zL_hat = torch.stack([r["z_L_hat"] for r in ok])
            zH_hat = torch.stack([r["z_H_hat"] for r in ok])
            import statistics

            def _line(tag, j):
                pr = j["primary"]
                msg = (f"[{tag}:{name}] sum-patch KL at position mean={statistics.mean(pr['kl_pred_at_patch']):.4f} "
                       f"(noise floor, stored gold: {statistics.mean(pr['kl_true_at_patch']):.4f})")
                if "frac_kl_recovered" in pr:
                    fr = [v for v in pr["frac_kl_recovered"] if v == v]
                    msg += f"  fraction of KL recovered vs mean-ablation: {statistics.mean(fr):.3f}" if fr else ""
                    km = pr["kl_mean_ablation_at_patch"]
                    msg += f" (ratio of means {1 - statistics.mean(pr['kl_pred_at_patch']) / statistics.mean(km):+.3f})"
                sec = j["secondary_zH"]
                msg += f"  | z_H-only patch KL mean={statistics.mean(sec['kl_pred_at_patch']):.4f}"
                if "frac_kl_recovered" in sec:
                    fr = [v for v in sec["frac_kl_recovered"] if v == v]
                    msg += f" recovered={statistics.mean(fr):.3f}" if fr else ""
                print(msg)

            j = run_judge(mimir_model, mimir_tok, judge_rows, zL_hat, zH_hat, args.device, mean_s, mean_zH)
            report["judge"][name] = j
            _line("judge", j)
            if args.judge_shuffle:
                perm = shuffle_permutation(judge_rows, args.judge_shuffle == "within-dataset")
                n_same = sum(a == b for a, b in enumerate(perm))
                if n_same:
                    print(f"  [judge-shuffle] {n_same} row(s) have no other row to swap with and keep their own vector")
                idx = torch.tensor(perm)
                js = run_judge(mimir_model, mimir_tok, judge_rows, zL_hat[idx], zH_hat[idx], args.device, mean_s, mean_zH)
                js["permutation"] = perm
                report.setdefault("judge_shuffled", {})[name] = js
                _line(f"judge-shuffled/{args.judge_shuffle}", js)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
