#!/bin/bash

cd ./experiments/GUE

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
export WANDB_PROJECT="GUE"
export WANDB_DIR="./wandb"

# 设置默认超参数
[ -z "${learning_rate}" ] && learning_rate=5e-5
[ -z "${batch_size}" ] && batch_size=32
[ -z "${weight_decay}" ] && weight_decay=1e-2

# 任务列表
tasks=(
    "virus/covid"
    "EMP/H3"
    "EMP/H3K4me1"
    "EMP/H3K4me2"
    "EMP/H3K4me3"
    "EMP/H3K9ac"
    "EMP/H3K14ac"
    "EMP/H3K36me3"
    "EMP/H3K79me3"
    "EMP/H4"
    "EMP/H4ac"
    "mouse/0"
    "mouse/1"
    "mouse/2"
    "mouse/3"
    "mouse/4"
    "prom/prom_300_all"
    "prom/prom_300_notata"
    "prom/prom_300_tata"
    "prom_core/prom_core_all"
    "prom_core/prom_core_notata"
    "prom_core/prom_core_tata"
    "splice/reconstructed"
    "tf/0"
    "tf/1"
    "tf/2"
    "tf/3"
    "tf/4"  
)


# 定义一个函数，用于在指定 GPU 上运行任务
run_task_on_gpu() {
    local gpu_id=$1
    local task_queue=$2
    local seed=$3
    local model_name_or_path=$4
    local output_path=$5
    local report_to=$6
    local learning_rate=$7
    local batch_size=$8
    local warmup_steps=$9

    while true; do
        # 使用 flock 加锁，确保同一时间只有一个进程读取任务
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
        echo "On GPU ${gpu_id} (Seed: ${seed} LR: ${learning_rate} Batch: ${batch_size}) processing ${task}"
        output_dir="${output_path}/${run_name}"
        if [ ! -d "$output_dir" ]; then
            mkdir -p "$output_dir"
        fi
        
        max_steps=10000
        eval_steps=200
        save_steps=200 
        
        #! virus/covid 
        if [[ "$task" =~ "virus/covid" ]]; then
            max_steps=14000
            learning_rate=8e-5
            warmup_steps=1000
            batch_size=16

        elif [[ "$task" =~ "tf/4" ]]; then
            learning_rate=5e-5
        elif [[ "$task" =~ "tf/3" ]]; then
            learning_rate=6e-5
        #! tf/2
        elif [[ "$task" =~ "tf/2" ]]; then
            learning_rate=3e-4
            batch_size=16
        #! tf/1
        elif [[ "$task" =~ "tf/1" ]]; then
            learning_rate=5e-5
            warmup_steps=600
        #! tf/0
        elif [[ "$task" =~ "tf/0" ]]; then
            learning_rate=8e-5  
            batch_size=16
        #! splice/reconstructed
        elif [[ "$task" =~ "splice/reconstructed" ]]; then
            learning_rate=6e-5
            batch_size=16
            max_steps=10000

        elif [[ "$task" =~ "prom_core/prom_core_tata" ]]; then
            learning_rate=3e-4
        #! prom_core/prom_core_notata
        elif [[ "$task" =~ "prom_core/prom_core_notata" ]]; then
            learning_rate=7e-5
            warmup_steps=400
        #! prom_core/prom_core_all
        elif [[ "$task" =~ "prom_core/prom_core_all" ]]; then
            learning_rate=3e-4
            warmup_steps=400
            batch_size=16

        elif [[ "$task" =~ "prom/prom_300_tata" ]]; then
            learning_rate=5e-5
            warmup_steps=400
        elif [[ "$task" =~ "prom/prom_300_notata" ]]; then
            learning_rate=5e-5
            warmup_steps=400
        elif [[ "$task" =~ "prom/prom_300_all" ]]; then
            learning_rate=4e-4
            warmup_steps=400
        #! mouse/4
        elif [[ "$task" =~ "mouse/4" ]]; then
            learning_rate=8e-5
            warmup_steps=400
            batch_size=16

        elif [[ "$task" =~ "mouse/3" ]]; then
            learning_rate=6e-5
        elif [[ "$task" =~ "mouse/2" ]]; then
            learning_rate=6e-5
        #! mouse/1
        elif [[ "$task" =~ "mouse/1" ]]; then
            learning_rate=1e-4
            warmup_steps=200
            batch_size=16
        elif [[ "$task" =~ "mouse/0" ]]; then
            learning_rate=5e-5
            warmup_steps=400
        elif [[ "$task" =~ "EMP/H4ac" ]]; then
            learning_rate=3e-4
        #! EMP/H4
        elif [[ "$task" =~ "EMP/H4" ]]; then
            learning_rate=1e-4
            warmup_steps=1000
        #! EMP/H3
        elif [[ "$task" =~ "EMP/H3" ]]; then
            learning_rate=8e-5
            warmup_steps=400
            batch_size=16
        elif [[ "$task" =~ "EMP/H3K9ac" ]]; then
            learning_rate=6e-5
        #! EMP/H3K79me3
        elif [[ "$task" =~ "EMP/H3K79me3" ]]; then
            learning_rate=3e-4
            warmup_steps=1000
        elif [[ "$task" =~ "EMP/H3K4me3" ]]; then
            learning_rate=3e-4
        elif [[ "$task" =~ "EMP/H3K4me2" ]]; then
            learning_rate=3e-4
        #! EMP/H3K4me1
        elif [[ "$task" =~ "EMP/H3K4me1" ]]; then
            learning_rate=3e-4
            warmup_steps=1000
        #! EMP/H3K36me3
        elif [[ "$task" =~ "EMP/H3K36me3" ]]; then
            learning_rate=6e-5
            warmup_steps=1000
        #! EMP/H3K14ac
        elif [[ "$task" =~ "EMP/H3K14ac" ]]; then
            learning_rate=9e-5
            warmup_steps=1000
        fi

        if [[ "$task" =~ "EMP" || "$task" =~ "mouse" ]]; then
            python expr_benchmark.py \
                --run_name ${run_name} \
                --max_steps $max_steps \
                --per_device_train_batch_size ${batch_size} \
                --per_device_eval_batch_size ${batch_size} \
                --dataset_name ${task} \
                --logging_steps 100 \
                --save_steps $save_steps \
                --eval_steps $eval_steps \
                --seed ${seed} \
                --save_strategy steps \
                --eval_strategy steps \
                --report_to ${report_to} \
                --output_dir ${output_dir} \
                --model_name_or_path ${model_name_or_path} \
                --remove_unused_columns False \
                --save_safetensors False \
                --save_total_limit 3 \
                --learning_rate ${learning_rate} \
                --weight_decay 1e-2 \
                --load_best_model_at_end \
                --metric_for_best_model "mcc" \
                --greater_is_better True \
                --warmup_steps ${warmup_steps} \
                --lr_scheduler_type cosine \
                > ${output_dir}/output.log 2>&1
        else
            python expr_benchmark.py \
                --run_name ${run_name} \
                --max_steps $max_steps \
                --per_device_train_batch_size ${batch_size} \
                --per_device_eval_batch_size ${batch_size} \
                --dataset_name ${task} \
                --logging_steps 100 \
                --save_steps $eval_steps \
                --eval_steps $save_steps \
                --seed ${seed} \
                --save_strategy steps \
                --eval_strategy steps \
                --report_to ${report_to} \
                --output_dir ${output_dir} \
                --model_name_or_path ${model_name_or_path} \
                --remove_unused_columns False \
                --save_safetensors False \
                --save_total_limit 3 \
                --learning_rate ${learning_rate} \
                --weight_decay 1e-2 \
                --load_best_model_at_end \
                --metric_for_best_model "mcc" \
                --greater_is_better True \
                > ${output_dir}/output.log 2>&1
        fi

    done 200>"$task_queue.lock"
}

learning_rates=(3e-4)
for learning_rate in "${learning_rates[@]}"; do
    seed=0
    warmup_steps=200
    # 创建任务队列
    task_queue=$(mktemp)
    for task in "${tasks[@]}"; do
        echo "$task" >> "$task_queue"
    done

    model_name_or_path="../../results/Space_species_tracks/checkpoint"
    output_path="./results/outputs_space_${seed}_${learning_rate}_${batch_size}_${warmup_steps}"
    # model_name_or_path="/home/jiwei_zhu/disk/Enformer/enformer_ckpt/"
    # output_path="./results/outputs_baseline_${seed}_${learning_rate}_${batch_size}"

    report_to="wandb"

    echo "Processing $output_path"
    if [ -f "$output_path/results.json" ]; then
        echo "文件 $output_path/results.json 已存在，跳过任务 (Seed: ${seed})"
        continue
    fi

    # 启动多个后台进程，每个 GPU 一个
    GPUS=2
    for ((gpu_id=0; gpu_id<GPUS; gpu_id++)); do 
        run_task_on_gpu $gpu_id "$task_queue" $seed "$model_name_or_path" "$output_path" "$report_to" "${learning_rate}" "${batch_size}" "${warmup_steps}" &
    done

    # 等待所有后台进程完成
    wait

    python ./result_combine.py --output_dir "${output_path}"

    # 删除任务队列文件和锁文件
    if [ -f "$task_queue" ]; then
        rm -f "$task_queue"
    fi
    if [ -f "$task_queue.lock" ]; then
        rm -f "$task_queue.lock"
    fi
done

# 汇总所有结果
python ./result.py
