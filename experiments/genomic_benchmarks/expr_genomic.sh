#!/bin/bash

# 设置分布式训练的环境变量
[ -z "${MASTER_PORT}" ] && MASTER_PORT=12346
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_LOCAL_RANK}" ] && OMPI_COMM_WORLD_LOCAL_RANK=0
[ -z "${GPUS}" ] && GPUS=$(nvidia-smi -L | wc -l)

if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]; then
  DISTRIBUTED_ARGS=""
else
  if (( $OMPI_COMM_WORLD_SIZE == 1 )); then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS \
                      --master_port $MASTER_PORT"
  else
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS \
                      --nnodes $OMPI_COMM_WORLD_SIZE \
                      --node_rank $OMPI_COMM_WORLD_RANK \
                      --master_addr $MASTER_ADDR"
  fi
fi

# 设置 WANDB 环境变量
unset WANDB_RUN_ID
export WANDB_API_KEY=""
export WANDB_PROJECT="genomic"
export WANDB_DIR="./wandb"
export WANDB_MODE=offline

# 设置默认超参数
[ -z "${learning_rate}" ] && learning_rate=5e-5
[ -z "${batch_size}" ] && batch_size=8
[ -z "${weight_decay}" ] && weight_decay=1e-2

# 任务列表
tasks=(
    'human_ocr_ensembl'
    'drosophila_enhancers_stark'
    'human_ensembl_regulatory'
    'demo_coding_vs_intergenomic_seqs'
    'human_enhancers_ensembl'
    'demo_human_or_worm'
    'human_enhancers_cohn'
    'human_nontata_promoters'
    'dummy_mouse_enhancers_ensembl'
)


# 定义一个函数，用于在指定 GPU 上运行任务
run_task_on_gpu() {
    local gpu_id=$1
    local task_queue=$2
    local seed=$3
    local model_name_or_path=$4
    local output_path=$5
    local report_to=$6
    local batch_size=$7
    local learning_rate=$8
    
    epochs=3

    while true; do
        # 使用 flock 加锁，确保同一时间只有一个进程读取任务队列
        flock 200
        task=$(head -n 1 "$task_queue")
        if [ -z "$task" ]; then
            flock -u 200
            break  # 任务队列为空，退出循环
        fi
        # 从任务队列中删除已读取的任务
        sed -i '1d' "$task_queue"
        flock -u 200

        export CUDA_VISIBLE_DEVICES=$gpu_id

        run_name="${task}"
        echo "On GPU ${gpu_id} (Seed: ${seed}) processing ${task}"
        output_dir="${output_path}/${run_name}"
        if [ ! -d "$output_dir" ]; then
            mkdir -p "$output_dir"
        fi

        





        python expr_genomic.py \
            --dataset_name ${task} \
            --num_train_epochs ${epochs}\
            --logging_steps 100 \
            --save_steps 200 \
            --eval_steps 200 \
            --learning_rate ${learning_rate} \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
            --seed ${seed} \
            --save_strategy steps \
            --eval_strategy steps \
            --run_name ${run_name} \
            --report_to ${report_to} \
            --output_dir ${output_dir} \
            --model_name_or_path ${model_name_or_path} \
            --remove_unused_columns False \
            --save_safetensors False \
            --load_best_model_at_end True \
            --save_total_limit 3 \
            > ${output_dir}/output.log 2>&1

    done 200>"$task_queue.lock"
}


# 主循环：遍历不同的随机种子
# for seed in $(seq 0 9); do
# batch_sizes=(8 16 32)
# learning_rates=(3e-6 8e-6 3e-5 5e-5 5e-4 8e-4)
batch_size=32
learning_rate=5e-5
seed=0
# for batch_size in "${batch_sizes[@]}"; do
#     for learning_rate in "${learning_rates[@]}"; do
        # 创建任务队列
        task_queue=$(mktemp)
        for task in "${tasks[@]}"; do
            echo "$task" >> "$task_queue"
        done

        # model_name_or_path="../../results/Space_species_tracks/checkpoint-28000"
        # output_path="./results/outputs_28000_bs${batch_size}_lr${learning_rate}_${seed}"
        model_name_or_path="/home/jiwei_zhu/disk/Enformer/enformer_ckpt"
        output_path="./results/outputs_baseline_${seed}"


        report_to="wandb"

        echo "Processing $output_path"
        if [ -f "$output_path/results.json" ]; then
            echo "文件 $output_path/results.json 已存在，跳过任务 (Seed: ${seed})"
            continue
        fi

        # 启动多个后台进程，每个 GPU 一个
        GPUS=3
        for ((gpu_id=0; gpu_id<GPUS; gpu_id++)); do
            run_task_on_gpu $gpu_id "$task_queue" $seed "$model_name_or_path" "$output_path" "$report_to" "$batch_size"  "$learning_rate" &
        done

        # 等待所有后台进程完成
        wait

        # 合并一次训练的不同数据集的结果
        python ./result_combine.py --output_dir "${output_path}"

        # 删除任务队列文件和锁文件
        rm "$task_queue" "$task_queue.lock"
#     done
# done
# 汇总所有结果
python ./result.py
