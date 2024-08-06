gpu=$1
config=$2
mp=${3:-23456}

# CUDA_VISIBLE_DEVICES=$gpu \
#     python -u train_arti.py \
#     --config $config

gpu_count=$(echo "$gpu" | tr -cd ',' | wc -c)
gpu_count=$((gpu_count + 1))

CUDA_VISIBLE_DEVICES=$gpu \
    torchrun --nproc_per_node=$gpu_count \
    --master_port=$mp \
    train.py --config $config