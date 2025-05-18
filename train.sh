#!/bin/bash  

# env variables for DDP training
[ -z "${MASTER_PORT}" ] && MASTER_PORT=1111
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_LOCAL_RANK}" ] && OMPI_COMM_WORLD_LOCAL_RANK=0
[ -z "${GPUS}" ] && GPUS=$(nvidia-smi -L | wc -l)

GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
per_device_batch_size=1
gradient_accumulation_steps=4


if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]
then
  DISTRIBUTED_ARGS=""
else
  if (( $OMPI_COMM_WORLD_SIZE == 1))
  then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS \
                      --master_port $MASTER_PORT"
  else
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS \
                      --nnodes $OMPI_COMM_WORLD_SIZE \
                      --node_rank $OMPI_COMM_WORLD_RANK \
                      --master_addr $MASTER_ADDR"
    fi
fi

export OMP_NUM_THREADS=1

unset WANDB_RUN_ID
export WANDB_API_KEY="66a051adfe75ea4635ddf96f876e6d39e2e0b4b0"
export WANDB_PROJECT="temp"
export WANDB_DIR="./wandb"

[ -z "${learning_rate}" ] && learning_rate=5e-4
[ -z "${lr_scheduler_type}" ] && lr_scheduler_type="cosine"
[ -z "${shift_aug}" ] && shift_aug=True
[ -z "${rc_aug}" ] && rc_aug=True


max_steps=50_000
weight_decay=1e-2
dataloader_num_workers=16
dataloader_prefetch_factor=2
report_to="wandb"
# 使用MoE的方式有四种，baseline, species, tracks, species_tracks
moe="species_tracks"
species_num_experts=4
tracks_num_experts=8
top_k=3
tracks_topk=3

MIloss_lambda=0.01
zloss_lambda=0.001
cvloss_lambda=0.005
# dim=768
# seqlen=131072
dim=1536
seqlen=196608
run_name="Space_${moe}_(${top_k}_${species_num_experts})_(${tracks_topk}_${tracks_num_experts})_${dim}_${seqlen}"
echo "Run name: ${run_name}"
output_path="./outputs_temp/${run_name}"


# report_to="none"
# dataloader_num_workers=0
mkdir -p ${output_path}
nohup torchrun ${DISTRIBUTED_ARGS} train.py \
    --run_name ${run_name} \
    --save_steps 1000 \
    --eval_steps 1000\
    --logging_steps 100 \
    --metric_for_best_model "loss_total" \
    --max_steps ${max_steps} \
    --warmup_steps 5000 \
    --report_to ${report_to} \
    --output_dir ${output_path} \
    --weight_decay ${weight_decay} \
    --optim adamw_torch \
    --learning_rate ${learning_rate} \
    --lr_scheduler_type cosine \
    --dataloader_num_workers ${dataloader_num_workers} \
    --save_strategy steps \
    --eval_strategy steps \
    --load_best_model_at_end True\
    --save_total_limit 30 \
    --remove_unused_columns False \
    --per_device_train_batch_size ${per_device_batch_size} \
    --per_device_eval_batch_size ${per_device_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --save_safetensors False \
    --dataloader_prefetch_factor ${dataloader_prefetch_factor} \
    --max_grad_norm 0.2 \
    --shift_aug ${shift_aug} \
    --rc_aug ${rc_aug} \
    --moe ${moe} \
    --species_num_experts ${species_num_experts} \
    --tracks_num_experts ${tracks_num_experts} \
    --tracks_topk ${tracks_topk} \
    --top_k ${top_k} \
    --MIloss_lambda ${MIloss_lambda} \
    --zloss_lambda ${zloss_lambda} \
    --cvloss_lambda ${cvloss_lambda} \
    --dim ${dim} \
    --seqlen ${seqlen} \
    > ${output_path}/output.log &
