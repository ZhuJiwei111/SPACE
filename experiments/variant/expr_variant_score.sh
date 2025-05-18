#!/bin/bash  

export CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7
export NCCL_DEBUG=ERROR
export PYTHONWARNINGS="ignore"
cd ./experiments/variant

echo "获取expression difference"
# output_dir="./experiments/variant/outputs_multi_embedding_baseline"
# model_path="/home/jiwei_zhu/disk/Enformer/enformer_ckpt/"
output_dir="./results/outputs_28000"
model_path="../../results/Space_species_tracks/checkpoint-28000"

mkdir -p $output_dir
bash expr_variant.sh "$output_dir" "$model_path"> $output_dir.log 
bash expr_variant.sh "$output_dir" "$model_path"> $output_dir.log 

# echo $! > train.pid  # 保存 PID
# TASK_PID=$(cat train.pid)
# echo "训练任务已启动 (PID: $TASK_PID)"

# # 等待任务完成
# echo "等待训练任务完成..."
# wait $TASK_PID
echo "训练任务已完成"

echo "计算variant score"
python expr_variant_score.py --directory "$output_dir" >> $output_dir.log 
echo "计算完成"