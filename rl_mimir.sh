set -e
M=danish-foundation-models/DFM-Mimir-v1.5
RLARGS="--rl-parquet rl_mimir.parquet --verbalizer-model $M --split-av --log-reward --batch-size 16 --group-size 8 --micro-batch-size 8 --policy-lr 1e-5 --kl-beta 0.05 --w-comp 1.0 --max-new-tokens 300 --norm-stats-json norm_stats.json --shuffle-seed 0"
# 0) RL probe from the stage-1 checkpoints: 2 steps, shows s/step and memory
python -u -m nla.hrm.train_rl $RLARGS --av-sft-ckpt ckpt/av_split_facts_v2_mimir --ar-sft-ckpt ckpt/ar_mimir_facts_v2 --steps 2 --output ckpt/rl_mimir_probe
nvidia-smi --query-gpu=memory.used --format=csv
# 1) stage-2 data with Mimir markers
python -m nla.hrm.build --input splits_v2/av_sft_explained.parquet --stage av_sft --prefix-token --position-fact --verbalizer-model $M --output av_sft_v2_mimir.parquet
python -m nla.hrm.split_av --in av_sft_v2_mimir.parquet --out av_split_sft_v2_mimir.parquet --facts position
python -m nla.hrm.build --input splits_v2/ar_sft_explained.parquet --stage ar_sft --prefix-token --position-fact --verbalizer-model $M --output ar_sft_v2_mimir.parquet
# 2) AV stage 2 (teacher prose + facts), init from stage 1
python -u -m nla.hrm.train_av_sft --train-parquet av_split_sft_v2_mimir.parquet --verbalizer-model $M --init-from ckpt/av_split_facts_v2_mimir --target-format-rate 2 --epochs 2 --prefix-weight 8 --eval-every 100000 --batch-size 8 --output ckpt/av_split_sft_v2_mimir
# 3) AR stage 2, init from the facts AR
python -u -m nla.hrm.train_ar_sft --backbone mimir --verbalizer-model $M --train-parquet ar_sft_v2_mimir.parquet --init-from ckpt/ar_mimir_facts_v2 --epochs 3 --output ckpt/ar_mimir_stage2
# 4) RL, 500 steps, shuffled prompts
python -u -m nla.hrm.train_rl $RLARGS --av-sft-ckpt ckpt/av_split_sft_v2_mimir --ar-sft-ckpt ckpt/ar_mimir_stage2 --steps 500 --save-every 50 --samples-every 10 --output ckpt/rl_mimir
