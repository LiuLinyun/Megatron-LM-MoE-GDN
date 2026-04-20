set -ex


export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=4,5,6,7
GPUS_PER_NODE=8
# export NVSHMEM_HCA_LIST=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_13
# export NVSHMEM_BOOTSTRAP=UID
# export NVSHMEM_IB_TRAFFIC_CLASS=130

# export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
# export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
# export NVSHMEM_IB_GID_INDEX=3

export NVTE_FWD_LAYERNORM_SM_MARGIN=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=24
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6657"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
  --nproc_per_node $GPUS_PER_NODE
  --nnodes $NNODES
  --node_rank $NODE_RANK
  --master_addr $MASTER_ADDR
  --master_port $MASTER_PORT
)

MEGATRON_LM_PATH=Megatron-LM/



# cd ${MEGATRON_LM_PATH}/megatron/core/datasets
# make -j32
# cd -

export PYTHONPATH=${MEGATRON_LM_PATH}:${PYTHONPATH}



export OUTPUT_BASE_DIR=z_outputs/moe_gdn/
export EXP_NAME=train_qwen3p5_moe_gdn



########### TRAIN SCRIPT #########

#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

export WANDB_MODE=offline

TOTAL_TOKENS=100000000000 # 大概100B！

WORLD_SIZE=$((8 * $NNODES))
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}

MICRO_BATCH=1
GLOBAL_BATCH=1024
TRAIN_ITER=$((TOTAL_TOKENS / GLOBAL_BATCH / 4096))

REPO_PATH=${OUTPUT_BASE_DIR}
JOB_NAME=${EXP_NAME}
TENSORBOARD_PATH="${REPO_PATH}/tensorboard/${JOB_NAME}"
CHECKPOINT_PATH="${REPO_PATH}/checkpoints/${JOB_NAME}"
WANDB_PATH="${REPO_PATH}/wandb/${JOB_NAME}"
LOG_DIR="${REPO_PATH}/logs/${JOB_NAME}"
LOG_FILE="${LOG_DIR}/$(date +"%Y%m%d_%H%M%S")_rank_${RANK}.log"

mkdir -p $TENSORBOARD_PATH
mkdir -p $CHECKPOINT_PATH
mkdir -p $WANDB_PATH
mkdir -p $LOG_DIR

cd ${MEGATRON_LM_PATH}
PRETRAIN_SCRIPT="${MEGATRON_LM_PATH}/pretrain_gpt.py"


BASE_PATH="${BASE_PATH:-data/merged_data}"
DATA_PATH="/your/path/to/tokenized/data/fineweb_edu_text_document"
DATA_PATH_CACHE="z_outputs/data/merged_data_cache"
TOKENIZER_PATH="z_ckpts/Qwen/Qwen3.5-2B-Base"

DATA_ARGS=(
    --tokenizer-model ${TOKENIZER_PATH}
    --tokenizer-type HuggingFaceTokenizer
    --data-path $DATA_PATH
    --data-cache-path ${DATA_PATH_CACHE}
    --train-iters $TRAIN_ITER
    --split 99,1,0
    --num-dataset-builder-threads 128
    --num-workers 16
    --no-mmap-bin-files
    --distributed-timeout-minutes 60
)

TRAINING_ARGS=(
    --lr 3e-4
    --lr-warmup-iters 100
    --lr-decay-style constant
    --min-lr 3e-4
    --lr-decay-iters $TRAIN_ITER
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
    --weight-decay 0.1
    --optimizer adam
)


MODEL_ARGS=(
    --num-layers 24
    --hidden-size 2048
    --ffn-hidden-size 6144
    --group-query-attention
    --num-attention-heads 8
    --num-query-groups 2
    --norm-epsilon 1e-6
    --kv-channels 128
    --seq-length 4096
    --max-position-embeddings 40960
    --attention-dropout 0
    --hidden-dropout 0
    --bf16
    --use-rotary-position-embeddings
    --rotary-base 1000000
    --swiglu
    --untie-embeddings-and-output-weights
    --normalization RMSNorm
    --qk-layernorm
    --disable-bias-linear
    --transformer-impl transformer_engine
    --attention-backend fused
    --init-method-std 0.02
    --use-cpu-initialization
)

LINEAR_ATTN_ARGS=(
    --experimental-attention-variant moe_gated_delta_net # compare to moe_gated_delta_net
    --linear-attention-freq 4
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    
    --linear-conv-kernel-dim 4
    # --linear-num-key-heads 16
    # --linear-num-value-heads 16

    --linear-num-shared-heads 8
    --linear-num-routed-heads 64
    --linear-write-topk 8
    --linear-read-topk 16
    --linear-write-coeff-for-read 0.5
    --linear-moe-router-enable-expert-bias

    --moe-per-layer-logging
)

EFFICIENCY_ARGS=(
    # --use-te-rng-tracker
    --use-flash-attn
    # --apply-rope-fusion
    # --bias-swiglu-fusion
    # --cross-entropy-loss-fusion
    # --overlap-grad-reduce
    # --overlap-param-gather
)

CKPT_ARGS=(
    --ckpt-format "torch_dist"
    --save-interval 200
    # --no-save-optim
    # --no-load-optim
    --async-save
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --use-distributed-optimizer
    --micro-batch-size ${MICRO_BATCH}
    --global-batch-size ${GLOBAL_BATCH}
)

LOGGER_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
)


WANDB_ARGS=(
    --wandb-project moe_gdn
    --wandb-exp-name $JOB_NAME
    --wandb-save-dir ${WANDB_PATH}
)


{
    torchrun \
        ${DISTRIBUTED_ARGS[@]} \
        $PRETRAIN_SCRIPT \
        ${DATA_ARGS[@]} \
        ${MODEL_ARGS[@]} \
        ${LINEAR_ATTN_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${PARALLEL_ARGS[@]} \
        ${EFFICIENCY_ARGS[@]} \
        ${CKPT_ARGS[@]} \
        ${LOGGER_ARGS[@]} \
        ${WANDB_ARGS[@]} \
        --eval-interval 500 \
        --eval-iters 25
} 2>&1 | tee -a "${LOG_FILE}"
