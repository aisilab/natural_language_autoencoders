"""RL: a self-written GRPO-style loop training the `av` LoRA adapter + the
per-stream injection adapters (policy) SIMULTANEOUSLY with the `ar` LoRA
adapter + ReconHeads (online critic), on the same rollouts each step — the
AR is the reward model (reward = -reconstruction loss), so it must keep
learning alongside the AV or the AV would game a frozen, increasingly-stale
critic (see docs/hrm.md and the original NLA's rl.sh).

Per step:
  1. Sample B activation rows x G rollouts each (temperature=1, greedy-free).
  2. Score EACH rollout's parsed (L, H) fields with the CURRENT ar+heads
     (no_grad) -> reward = -loss (or -log(loss)); malformed -> failed_reward.
  3. Group-normalize advantages within each row's G rollouts (GRPO).
  4. Policy step: token-mean REINFORCE + advantage, plus a k2-KL penalty
     toward the frozen post-SFT `av_ref` snapshot. Updates av LoRA + inj_L/H.
  5. AR step: a fresh (WITH grad) AR forward + recon_loss on the well-formed
     rollouts from this batch -> supervised update of ar LoRA + heads. This
     is a SEPARATE forward pass from step 2's (no_grad) one — different
     optimizers, no shared graph.
  6. Every --eval-every steps: reconstruction FVE on a held-out parquet
     (cheap; the heavier Mimir patch-back judge is `judge.py`, run
     separately/periodically, never inside this loop — see docs/hrm.md).

Mimir itself is NEVER loaded here — this script only touches the verbalizer
and the RAW gold z_L/z_H already extracted into the RL parquet.
"""

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from nla.hrm.devices import default_device, default_dtype
from nla.datagen.storage import LocalStorage
from nla.hrm.build import _INJECT_H_PLACEHOLDER, _INJECT_L_PLACEHOLDER
from nla.hrm.model import (
    DEFAULT_VERBALIZER, add_frozen_reference_adapter, build_inputs_embeds,
    load_rl_checkpoint, load_verbalizer, save_extra_modules,
    last_real_index,
)
from nla.hrm.verifiable import split_call_rewards, verifiable_bonus, verifiable_terms
from nla.hrm.recon import ReconLoss, ReconWeights, failed_reward, loss_to_reward, parse_fields, recon_loss
from nla.hrm.sidecar import HrmDatasetMeta, read_sidecar
from nla.hrm.split_av import join_split, tag_content

_D_MIMIR = 1536
_CJK_RE = re.compile(r"[㈀-㏿一-鿿]")


def _load_rl_rows(parquet_path: str) -> tuple[list[dict], HrmDatasetMeta]:
    names = pq.ParquetFile(parquet_path).schema_arrow.names
    cols = ["prompt", "z_L", "z_H"] + [c for c in ("context_marked", "dataset", "position", "prompt_len") if c in names]
    t = pq.read_table(parquet_path, columns=cols)
    return t.to_pylist(), read_sidecar(LocalStorage(), parquet_path)


def _extra_content(row: dict, inj_l_char: str, inj_h_char: str) -> str:
    return row["prompt"][0]["content"].replace(_INJECT_L_PLACEHOLDER, inj_l_char).replace(_INJECT_H_PLACEHOLDER, inj_h_char)


def _mask_streams(z_L: torch.Tensor, z_H: torch.Tensor, mask_stream: str | None):
    """Split AV: zero the masked stream (the gold vectors used for the reward stay untouched)."""
    if mask_stream == "L":
        return torch.zeros_like(z_L), z_H
    if mask_stream == "H":
        return z_L, torch.zeros_like(z_H)
    return z_L, z_H


@torch.no_grad()
def _generate_batch(model, tokenizer, rows, inj_l_char, inj_h_char, inj_L, inj_H, ids_meta, device,
                     group_size: int, max_new_tokens: int, temperature: float, mask_stream: str | None = None):
    """rows: B activation rows. Returns per-(row,sample) generated text + the
    full token sequence (for logp scoring) + prompt length S (constant across
    the batch thanks to left-padding)."""
    contents = [_extra_content(r, inj_l_char, inj_h_char) for r in rows for _ in range(group_size)]
    if mask_stream:  # split AV: describe the OTHER stream; this one is zeroed in the embeddings below
        contents = [tag_content(c, "H" if mask_stream == "L" else "L") for c in contents]
    tokenizer.padding_side = "left"
    enc = tokenizer.apply_chat_template(
        [[{"role": "user", "content": c}] for c in contents],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    S = input_ids.shape[1]

    z_L = torch.tensor([r["z_L"] for r in rows for _ in range(group_size)], dtype=torch.float32, device=device)
    z_H = torch.tensor([r["z_H"] for r in rows for _ in range(group_size)], dtype=torch.float32, device=device)
    embeds = build_inputs_embeds(model, input_ids, *_mask_streams(z_L, z_H, mask_stream), inj_L, inj_H, *ids_meta)

    model.set_adapter("av")
    out_ids = model.generate(
        inputs_embeds=embeds, attention_mask=attn, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, top_p=1.0, pad_token_id=tokenizer.pad_token_id,
    )
    # generate() with inputs_embeds returns ONLY the newly generated tokens
    # (it never sees input_ids to prepend) — reconstruct the full sequence
    # ourselves for teacher-forced scoring.
    full_ids = torch.cat([input_ids, out_ids], dim=1)
    new_attn = (out_ids != tokenizer.pad_token_id).long()
    full_attn = torch.cat([attn, new_attn], dim=1)
    texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
    return full_ids, full_attn, S, texts, z_L, z_H


# Micro-batch size for every forward pass that needs a backward (policy step, AR update)
# and for no-grad scoring. Set from --micro-batch-size in main(). Full-vocab logits and
# activations for all B*G rollouts at once are what used to OOM a 95 GB GPU.
_MICRO_BATCH = 8
# Which token the AR reads out: "last" (the last attended token; correct under any padding side) or "legacy" (attn.sum - 1;
# wrong for left-padded batches, see model.last_real_index). Set from --critic-readout in main().
_CRITIC_READOUT = "last"


def fve_from_mses(mses: dict, baseline: dict | None) -> dict[str, float]:
    """Mean FVE per term over this batch's well-formed rollouts, relative to
    norm_stats.json's train-set mean-predictor MSE (FVE=0 means "no better than
    predicting the mean"). Cheap: the per-rollout MSEs are already computed for the reward."""
    if not mses or baseline is None:
        return {}
    n = len(mses)
    cols = {"sum": 0, "L": 1, "H": 2}
    return {f"fve_{k}": 1.0 - (sum(m[i] for m in mses.values()) / n) / baseline[k] for k, i in cols.items()}


def _critic_hidden(model, tokenizer, texts: list[str], device: str) -> torch.Tensor:
    model.set_adapter("ar")
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False,
                     truncation=True, max_length=512)
    ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    # logits_to_keep=1: we only need hidden states; don't build [B, T, vocab] logits.
    out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, logits_to_keep=1)
    return out.hidden_states[-1][torch.arange(ids.shape[0]), last_real_index(attn, _CRITIC_READOUT)]


def _critic_hidden_nograd(model, tokenizer, texts: list[str], device: str) -> torch.Tensor:
    return torch.cat([_critic_hidden(model, tokenizer, texts[i : i + _MICRO_BATCH], device)
                      for i in range(0, len(texts), _MICRO_BATCH)])


def compute_rewards(model, tokenizer, heads, critic_template, texts, z_L, z_H, weights, device, log_reward: bool):
    """no_grad AR forward -> reward per rollout (float list) + parsed fields
    (None for malformed) — the parsed fields are reused by the AR update step
    so its texts aren't re-tokenized twice for nothing."""
    parsed = [parse_fields(t) for t in texts]
    ok_idx = [i for i, p in enumerate(parsed) if p is not None]
    computed: dict[int, float] = {}
    mses: dict[int, tuple[float, float, float]] = {}  # rollout -> (mse_sum, mse_L, mse_H), for FVE logging
    if ok_idx:
        l_texts = [critic_template.format(explanation=parsed[i][0]) for i in ok_idx]
        h_texts = [critic_template.format(explanation=parsed[i][1]) for i in ok_idx]
        with torch.no_grad():
            h_l = _critic_hidden_nograd(model, tokenizer, l_texts, device)
            h_h = _critic_hidden_nograd(model, tokenizer, h_texts, device)
            zL_hat, zH_hat = heads.forward_L(h_l), heads.forward_H(h_h)
            gold_L, gold_H = z_L[ok_idx], z_H[ok_idx]
            for j, i in enumerate(ok_idx):
                loss = recon_loss(zL_hat[j : j + 1], zH_hat[j : j + 1], gold_L[j : j + 1], gold_H[j : j + 1], weights)
                computed[i] = loss_to_reward(loss.total, log_reward=log_reward)
                mses[i] = (loss.mse_sum.item(), loss.mse_L.item(), loss.mse_H.item())
    fallback = failed_reward(weights, log_reward=log_reward)
    rewards: list[float] = [computed[i] if i in computed else fallback for i in range(len(texts))]
    return rewards, parsed, mses


def ar_update_step(model, tokenizer, heads, heads_optim, critic_template, parsed, z_L, z_H, weights, device):
    ok_idx = [i for i, p in enumerate(parsed) if p is not None]
    if not ok_idx:
        return None
    heads_optim.zero_grad()
    n = len(ok_idx)
    tot = mse_sum = mse_L = mse_H = 0.0
    for start in range(0, n, _MICRO_BATCH):
        idx = ok_idx[start : start + _MICRO_BATCH]
        l_texts = [critic_template.format(explanation=parsed[i][0]) for i in idx]
        h_texts = [critic_template.format(explanation=parsed[i][1]) for i in idx]
        h_l = _critic_hidden(model, tokenizer, l_texts, device)
        h_h = _critic_hidden(model, tokenizer, h_texts, device)
        loss = recon_loss(heads.forward_L(h_l), heads.forward_H(h_h), z_L[idx], z_H[idx], weights)
        w = len(idx) / n  # chunk means -> full-batch mean
        (loss.total * w).backward()
        tot += loss.total.item() * w
        mse_sum += loss.mse_sum.item() * w
        mse_L += loss.mse_L.item() * w
        mse_H += loss.mse_H.item() * w
    heads_optim.step()
    return ReconLoss(total=torch.tensor(tot), mse_sum=torch.tensor(mse_sum),
                     mse_L=torch.tensor(mse_L), mse_H=torch.tensor(mse_H))


def _response_logprobs(model, adapter: str, embeds, attn, targets, S: int) -> torch.Tensor:
    """log p(token) for the response positions only. logits_to_keep restricts the LM head
    to positions S-1..end, so we never build [N, seq, vocab] logits for the prompt."""
    model.set_adapter(adapter)
    k = embeds.shape[1] - S + 1
    logits = model(inputs_embeds=embeds, attention_mask=attn, logits_to_keep=k).logits[:, :-1]
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def policy_step(model, full_ids, full_attn, S, advantages, inj_L, inj_H, ids_meta, z_L, z_H, policy_optim,
                 kl_beta: float, device, mask_stream: str | None = None):
    N = full_ids.shape[0]
    policy_optim.zero_grad()
    pg_total = kl_total = 0.0
    for start in range(0, N, _MICRO_BATCH):
        sl = slice(start, start + _MICRO_BATCH)
        ids, attn = full_ids[sl], full_attn[sl]
        targets, resp_mask = ids[:, S:], attn[:, S:].float()
        embeds = build_inputs_embeds(model, ids, *_mask_streams(z_L[sl], z_H[sl], mask_stream), inj_L, inj_H, *ids_meta)
        with torch.no_grad():
            logp_ref = _response_logprobs(model, "av_ref", embeds, attn, targets, S)
        logp = _response_logprobs(model, "av", embeds, attn, targets, S)

        denom = resp_mask.sum(-1).clamp_min(1.0)
        kl = ((0.5 * (logp - logp_ref).pow(2) * resp_mask).sum(-1) / denom)
        resp_logp_mean = (logp * resp_mask).sum(-1) / denom
        pg = -(advantages[sl].to(device) * resp_logp_mean)
        w = ids.shape[0] / N
        ((pg.mean() + kl_beta * kl.mean()) * w).backward()
        pg_total += pg.mean().item() * w
        kl_total += kl.mean().item() * w
    policy_optim.step()
    model.set_adapter("av")
    return pg_total, kl_total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rl-parquet", required=True)
    p.add_argument("--eval-parquet", default=None)
    p.add_argument("--av-sft-ckpt", required=True, help="train_av_sft.py's --output dir (contains av/adapter_config.json)")
    p.add_argument("--ar-sft-ckpt", required=True, help="train_ar_sft.py's --output dir (contains ar/adapter_config.json)")
    p.add_argument("--verbalizer-model", default=None)
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default=default_dtype())
    p.add_argument("--batch-size", type=int, default=8, help="B activation rows per step")
    p.add_argument("--group-size", type=int, default=8, help="G rollouts per row (GRPO group)")
    p.add_argument("--micro-batch-size", type=int, default=8,
                    help="rollouts per forward/backward pass (memory knob; results are identical)")
    p.add_argument("--norm-stats-json", default=None,
                    help="norm_stats.json for FVE logging (default: norm_stats.json next to --rl-parquet, if present)")
    p.add_argument("--samples-every", type=int, default=10,
                    help="every N steps append a few rollouts (text, reward) to <output>/samples.jsonl; 0=off")
    p.add_argument("--samples-per-dump", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="shuffle the RL rows with this seed before training. Default (None): the rows are used in FILE ORDER, "
                        "batch k = rows [16k, 16k+16): a run of N steps only ever sees the first N*batch_size rows of the "
                        "parquet (E18 saw the first 24k of 110k rows; the parquet order follows the length-sorted extraction "
                        "batches, to be verified), and two runs from the same SFT start see the SAME prompts")
    p.add_argument("--policy-lr", type=float, default=1e-5)
    p.add_argument("--ar-lr", type=float, default=1e-4)
    p.add_argument("--kl-beta", type=float, default=0.01)
    p.add_argument("--w-sum", type=float, default=1.0)
    p.add_argument("--w-comp", type=float, default=0.25)
    p.add_argument("--log-reward", action="store_true", help="reward = -log(loss) instead of -loss")
    p.add_argument("--critic-readout", choices=["last", "legacy"], default="last",
                   help="token the AR reads out: 'last' = last attended token (correct); 'legacy' = attn.sum-1, which is "
                        "wrong for the left-padded batches used here (reproduces runs before 2026-10-11)")
    p.add_argument("--split-av", action="store_true",
                   help="split AV (nla/hrm/split_av.py): each rollout pair is TWO calls, one with only z_L (z_H zeroed) "
                        "writing the L field and one with only z_H writing the H field; the AV checkpoint must come from "
                        "train_av_sft on split_av.py's parquet. Two policy updates per step; in-training eval/sanity skipped.")
    p.add_argument("--w-fact", type=float, default=0.0,
                   help="split AV only: bonus per correctly stated fact (marked token, position fifth) for EACH call, both facts "
                        "for both calls, added to that call's own reward (per-call advantages); last prompt positions get "
                        "no bonus; 0 = off. Needs position and prompt_len in the RL parquet")
    p.add_argument("--w-token", type=float, default=0.0,
                   help="verifiable reward: bonus for quoting the REAL marked token (see verifiable.py); 0 = off")
    p.add_argument("--w-ground", type=float, default=0.0,
                   help="verifiable reward: penalty (up to this value) for quoted spans/names not in the prompt; 0 = off")
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--sanity", action="store_true", help="run the real-vs-shuffled-vectors check before training")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    verbalizer_model = args.verbalizer_model or DEFAULT_VERBALIZER
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    weights = ReconWeights(w_sum=args.w_sum, w_comp=args.w_comp)
    norm_path = args.norm_stats_json or str(Path(args.rl_parquet).parent / "norm_stats.json")
    baseline = json.load(open(norm_path))["mean_mse"] if Path(norm_path).exists() else None
    print(f"[train_rl] FVE baseline: {norm_path if baseline else 'none (no norm_stats.json — FVE not logged)'}")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    global _MICRO_BATCH, _CRITIC_READOUT
    _MICRO_BATCH = args.micro_batch_size
    _CRITIC_READOUT = args.critic_readout
    if args.critic_readout == "legacy":
        print("[train_rl] WARNING: --critic-readout legacy: the AR reads a misaligned token for padded rows (reproduces "
              "runs before 2026-10-11 only)")

    model, tokenizer = load_verbalizer(verbalizer_model, device=args.device, torch_dtype=dtype)
    d_verb = model.config.hidden_size
    rows, meta = _load_rl_rows(args.rl_parquet)
    if args.shuffle_seed is not None:
        import random
        random.Random(args.shuffle_seed).shuffle(rows)
        print(f"[train_rl] shuffled {len(rows)} RL rows with seed {args.shuffle_seed}")
    if args.w_fact:
        assert args.split_av, "--w-fact is defined for the split AV (--split-av) only"
        assert all(r.get("context_marked") and r.get("position") is not None and r.get("prompt_len") for r in rows[:100]), (
            "--w-fact needs context_marked, position and prompt_len columns in the RL parquet")
    if args.w_token or args.w_ground:
        assert all(r.get("context_marked") for r in rows[:100]), (
            "--w-token/--w-ground need the context_marked column in the RL parquet (build with debug metadata)")
    tm = meta.tokens
    assert tm is not None
    inj_l_char, inj_h_char = tm.injection_char_L, tm.injection_char_H
    ids_meta = (tm.injection_token_id_L, tm.injection_left_neighbor_id_L, tm.injection_right_neighbor_id_L,
                tm.injection_token_id_H, tm.injection_left_neighbor_id_H, tm.injection_right_neighbor_id_H)
    critic_template = meta.prompt_templates["critic"]

    inj_L, inj_H, heads = load_rl_checkpoint(model, args.av_sft_ckpt, args.ar_sft_ckpt, _D_MIMIR, d_verb, args.device)
    add_frozen_reference_adapter(model, source_adapter="av", ref_name="av_ref")
    print(f"[train_rl] adapters: {list(model.peft_config.keys())}")

    for name, param in model.named_parameters():
        param.requires_grad_((".av." in name or ".ar." in name) and ".av_ref." not in name)
    policy_params = [pr for n, pr in model.named_parameters() if pr.requires_grad and ".av." in n]
    policy_params += list(inj_L.parameters()) + list(inj_H.parameters())
    ar_params = [pr for n, pr in model.named_parameters() if pr.requires_grad and ".ar." in n] + list(heads.parameters())
    policy_optim = torch.optim.AdamW(policy_params, lr=args.policy_lr)
    ar_optim = torch.optim.AdamW(ar_params, lr=args.ar_lr)
    print(f"[train_rl] policy params: {sum(p.numel() for p in policy_params):,}  "
          f"ar params: {sum(p.numel() for p in ar_params):,}")

    if args.sanity and not args.split_av:
        _run_sanity_check(model, tokenizer, rows[: args.batch_size], inj_l_char, inj_h_char, inj_L, inj_H,
                            ids_meta, heads, critic_template, weights, args.device, args.log_reward)

    step = 0
    with tqdm(total=args.steps) as pbar:
        while step < args.steps:
            batch_rows = [rows[i % len(rows)] for i in range(step * args.batch_size, (step + 1) * args.batch_size)]
            if args.split_av:
                full_ids, full_attn, S, texts_L, z_L, z_H = _generate_batch(
                    model, tokenizer, batch_rows, inj_l_char, inj_h_char, inj_L, inj_H, ids_meta, args.device,
                    args.group_size, args.max_new_tokens, args.temperature, mask_stream="H",  # the L call
                )
                full_ids_B, full_attn_B, S_B, texts_H, _, _ = _generate_batch(
                    model, tokenizer, batch_rows, inj_l_char, inj_h_char, inj_L, inj_H, ids_meta, args.device,
                    args.group_size, args.max_new_tokens, args.temperature, mask_stream="L",  # the H call
                )
                texts = [join_split(a, b) for a, b in zip(texts_L, texts_H, strict=True)]
                cjk_texts = texts_L + texts_H
            else:
                full_ids, full_attn, S, texts, z_L, z_H = _generate_batch(
                    model, tokenizer, batch_rows, inj_l_char, inj_h_char, inj_L, inj_H, ids_meta, args.device,
                    args.group_size, args.max_new_tokens, args.temperature,
                )
                cjk_texts = texts
            rewards, parsed, mses = compute_rewards(model, tokenizer, heads, critic_template, texts, z_L, z_H,
                                                weights, args.device, args.log_reward)
            fve = fve_from_mses(mses, baseline)
            verif = {}
            if args.w_token or args.w_ground:
                terms = [(i, verifiable_terms(p, batch_rows[i // args.group_size].get("context_marked")))
                         for i, p in enumerate(parsed) if p is not None]
                for i, t in terms:
                    rewards[i] += verifiable_bonus(t, args.w_token, args.w_ground)
                if terms:
                    verif = {"token_ok": sum(t["token_correct"] for _, t in terms) / len(terms),
                             "ungrounded": sum(t["ungrounded"] for _, t in terms) / len(terms)}
            def _advantages(rs: list[float]) -> torch.Tensor:
                rt = torch.tensor(rs, dtype=torch.float32).view(len(batch_rows), args.group_size)
                mean, std = rt.mean(dim=1, keepdim=True), rt.std(dim=1, keepdim=True).clamp_min(1e-4)
                return ((rt - mean) / std).view(-1)

            advantages = _advantages(rewards)
            adv_L = adv_H = advantages
            if args.w_fact:  # per-call rewards and advantages: each call is credited for its own stated facts
                rewards_L, rewards_H, fact_rates = split_call_rewards(rewards, texts_L, texts_H, batch_rows,
                                                                      args.group_size, args.w_fact)
                adv_L, adv_H = _advantages(rewards_L), _advantages(rewards_H)
                verif.update({f"fact_{k}": v for k, v in fact_rates.items()})

            if args.split_av:
                pg_a, kl_a = policy_step(model, full_ids, full_attn, S, adv_L, inj_L, inj_H, ids_meta,
                                           z_L, z_H, policy_optim, args.kl_beta, args.device, mask_stream="H")
                pg_b, kl_b = policy_step(model, full_ids_B, full_attn_B, S_B, adv_H, inj_L, inj_H, ids_meta,
                                           z_L, z_H, policy_optim, args.kl_beta, args.device, mask_stream="L")
                pg_loss, kl_loss = (pg_a + pg_b) / 2, (kl_a + kl_b) / 2
            else:
                pg_loss, kl_loss = policy_step(model, full_ids, full_attn, S, advantages, inj_L, inj_H, ids_meta,
                                                 z_L, z_H, policy_optim, args.kl_beta, args.device)
            ar_loss = ar_update_step(model, tokenizer, heads, ar_optim, critic_template, parsed, z_L, z_H,
                                       weights, args.device)

            n_malformed = sum(p is None for p in parsed)
            n_cjk_leak = sum(bool(_CJK_RE.search(t)) for t in cjk_texts)
            step += 1
            pbar.update(1)
            pbar.set_postfix(reward=f"{sum(rewards)/len(rewards):.3f}", pg=f"{pg_loss:.3f}", kl=f"{kl_loss:.4f}")
            if args.samples_every and (step % args.samples_every == 0 or step == 1):
                with open(f"{args.output}/samples.jsonl", "a") as f:
                    for k in range(min(args.samples_per_dump, len(texts))):
                        row = batch_rows[k // args.group_size]
                        f.write(json.dumps({
                            "step": step, "reward": rewards[k], "well_formed": parsed[k] is not None,
                            "dataset": row.get("dataset"), "position": row.get("position"),
                            "context_marked": row.get("context_marked"), "completion": texts[k],
                        }, ensure_ascii=False) + "\n")
            if step % 5 == 0 or step == 1:
                print(f"step={step} mean_reward={sum(rewards)/len(rewards):.4f} pg_loss={pg_loss:.4f} "
                      f"kl={kl_loss:.5f} ar_loss={ar_loss.total.item() if ar_loss else float('nan'):.4f} "
                      f"malformed={n_malformed}/{len(texts)} cjk_leak={n_cjk_leak}/{len(cjk_texts)}"
                      + "".join(f" {k}={v:.3f}" for k, v in {**fve, **verif}.items()))
                if n_cjk_leak > 0:
                    print(f"  WARNING: {n_cjk_leak} completions contain CJK chars — possible injection failure "
                          f"(the marker char leaking into generated text means it wasn't found/overwritten).")

            if step % args.eval_every == 0 and args.eval_parquet and not args.split_av:
                _run_eval(model, tokenizer, args.eval_parquet, inj_l_char, inj_h_char, inj_L, inj_H, ids_meta,
                            heads, critic_template, weights, args.device, args.log_reward, args.max_new_tokens)
            if step % args.save_every == 0 or step == args.steps:
                _save(model, inj_L, inj_H, heads, f"{args.output}/step_{step}")

    _save(model, inj_L, inj_H, heads, f"{args.output}/final")
    print(f"saved final checkpoint -> {args.output}/final")


def _save(model, inj_L, inj_H, heads, path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, selected_adapters=["av", "ar"])
    save_extra_modules(f"{path}/extra_modules.safetensors", inj_L, inj_H, heads)


@torch.no_grad()
def _run_eval(model, tokenizer, eval_parquet, inj_l_char, inj_h_char, inj_L, inj_H, ids_meta, heads,
               critic_template, weights, device, log_reward, max_new_tokens):
    rows, _ = _load_rl_rows(eval_parquet)
    _full_ids, _full_attn, _S, texts, z_L, z_H = _generate_batch(
        model, tokenizer, rows[: min(16, len(rows))], inj_l_char, inj_h_char, inj_L, inj_H, ids_meta, device,
        1, max_new_tokens, temperature=1.0,  # sampled, not greedy — matches training-time rollout distribution
    )
    rewards, parsed, _mses = compute_rewards(model, tokenizer, heads, critic_template, texts, z_L, z_H, weights, device, log_reward)
    print(f"  [eval] n={len(texts)} mean_reward={sum(rewards)/len(rewards):.4f} "
          f"malformed={sum(p is None for p in parsed)}/{len(texts)}")


@torch.no_grad()
def _run_sanity_check(model, tokenizer, rows, inj_l_char, inj_h_char, inj_L, inj_H, ids_meta, heads,
                        critic_template, weights, device, log_reward):
    """Reconstruction reward with REAL vectors should beat SHUFFLED vectors
    (activations permuted across the batch) — the loudest smoke test that
    injection + the AV/AR pipeline actually carries signal end-to-end."""
    _full_ids, _full_attn, _S, texts, z_L, z_H = _generate_batch(
        model, tokenizer, rows, inj_l_char, inj_h_char, inj_L, inj_H, ids_meta, device, 1, 200, 1.0
    )
    rewards_real, _, _m1 = compute_rewards(model, tokenizer, heads, critic_template, texts, z_L, z_H, weights, device, log_reward)
    # roll by one: a guaranteed non-identity shuffle (randperm is the identity with prob 1/N!,
    # which made tiny-batch sanity checks report real == shuffled)
    perm = torch.roll(torch.arange(z_L.shape[0]), 1)
    rewards_shuf, _, _m2 = compute_rewards(model, tokenizer, heads, critic_template, texts, z_L[perm], z_H[perm],
                                        weights, device, log_reward)
    r_real, r_shuf = sum(rewards_real) / len(rewards_real), sum(rewards_shuf) / len(rewards_shuf)
    print(f"[sanity] mean_reward real={r_real:.4f} shuffled={r_shuf:.4f}")
    if r_real <= r_shuf:
        print("  WARNING: shuffled vectors score >= real vectors — injection may not be wired correctly "
              "(check nla/hrm/model.py:build_inputs_embeds and the marker IDs in the sidecar).")


if __name__ == "__main__":
    main()
