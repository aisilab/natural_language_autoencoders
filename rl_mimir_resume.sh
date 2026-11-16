set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=danish-foundation-models/DFM-Mimir-v1.5
RLARGS="--rl-parquet rl_mimir.parquet --verbalizer-model $M --split-av --log-reward --batch-size 16 --group-size 8 --micro-batch-size 4 --policy-lr 1e-5 --kl-beta 0.05 --w-comp 1.0 --max-new-tokens 300 --norm-stats-json norm_stats.json --shuffle-seed 0"
# AR stage 2: batch 8 (batch 16 ran out of memory on the long prose texts)
python -u -m nla.hrm.train_ar_sft --backbone mimir --verbalizer-model $M --train-parquet ar_sft_v2_mimir.parquet --init-from ckpt/ar_mimir_facts_v2 --epochs 3 --batch-size 8 --output ckpt/ar_mimir_stage2
python -u -m nla.hrm.train_rl $RLARGS --av-sft-ckpt ckpt/av_split_sft_v2_mimir --ar-sft-ckpt ckpt/ar_mimir_stage2 --steps 500 --save-every 50 --samples-every 10 --output ckpt/rl_mimir
