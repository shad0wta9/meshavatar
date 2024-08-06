gpu=$1
config=$2

CUDA_VISIBLE_DEVICES=$gpu \
    python -u train.py \
    --config $config

# gpu_count=$(echo "$gpu" | tr -cd ',' | wc -c)
# gpu_count=$((gpu_count + 1))

# CUDA_VISIBLE_DEVICES=$gpu \
#     torchrun --nproc_per_node=$gpu_count \
#     train_arti.py --config $config