set -e
M=danish-foundation-models/DFM-Mimir-v1.5
# 1) 30-step timing probe
python -u -m nla.hrm.train_av_sft --train-parquet av_split_facts_v2_mimir.parquet --verbalizer-model $M --target-format-rate 2 --epochs 1 --prefix-weight 8 --eval-every 100000 --max-steps 30 --batch-size 8 --output ckpt/mimir_timing
nvidia-smi --query-gpu=memory.used --format=csv
# 2) stage 1, full 110k facts rows (220k split calls)
python -u -m nla.hrm.train_av_sft --train-parquet av_split_facts_v2_mimir.parquet --verbalizer-model $M --target-format-rate 2 --epochs 1 --prefix-weight 8 --eval-every 100000 --save-every 2000 --batch-size 8 --output ckpt/av_split_facts_v2_mimir
# 3) eval on the clean v2 evals
python -m nla.hrm.eval --eval-parquet iid=eval_iid_v2_mimir.parquet ood=eval_ood_v2_mimir.parquet --av-ckpt ckpt/av_split_facts_v2_mimir --ar-ckpt ckpt/ar_mimir_facts_v2 --split-av --limit 300 --max-new-tokens 100 --norm-stats-json norm_stats.json --dump-samples samples_facts_mimir.jsonl --output eval_facts_mimir.json
