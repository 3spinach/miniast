#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="mini-ast-esc50"
#SBATCH --output=./log_%j.txt

set -x

export TORCH_HOME=../../pretrained_models

# Model configuration
model=miniast
dataset=esc50
imagenetpretrain=True
audiosetpretrain=True

# Learning rate
if [ $audiosetpretrain == True ]; then
    lr=1e-5
else
    lr=1e-4
fi

# Data augmentation
freqm=24
timem=96
mixup=0

# Training settings
epoch=25
batch_size=24
fstride=10
tstride=10

# Dataset statistics for ESC-50
dataset_mean=-6.6268077
dataset_std=5.358466
audio_length=512
noise=False

# Evaluation settings
metrics=acc
loss=CE
warmup=False
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.85

# ========== MiniViT-specific settings ==========
num_shared_layers=2        # Share every 2 consecutive layers
use_attn_transform=True    # Enable attention transformation
use_mlp_transform=True     # Enable MLP transformation
mlp_kernel_size=7          # Kernel size for depth-wise conv

# Experiment directory
base_exp_dir=./exp/mini-${dataset}-f$fstride-t$tstride-imp$imagenetpretrain-asp$audiosetpretrain-b$batch_size-lr${lr}-shared${num_shared_layers}

# Prepare ESC-50 dataset
python ./prep_esc50.py

if [ -d $base_exp_dir ]; then
    echo 'Experiment directory exists'
    exit
fi
mkdir -p $base_exp_dir

# 5-fold cross validation
for((fold=1;fold<=5;fold++));
do
    echo "Processing fold ${fold}"
    
    exp_dir=${base_exp_dir}/fold${fold}
    
    tr_data=./data/datafiles/esc_train_data_${fold}.json
    te_data=./data/datafiles/esc_eval_data_${fold}.json
    
    CUDA_CACHE_DISABLE=1 python -W ignore ../src/run.py \
        --model ${model} \
        --dataset ${dataset} \
        --data-train ${tr_data} \
        --data-val ${te_data} \
        --exp-dir $exp_dir \
        --label-csv ./data/esc_class_labels_indices.csv \
        --n_class 50 \
        --lr $lr \
        --n-epochs ${epoch} \
        --batch-size $batch_size \
        --save_model False \
        --freqm $freqm \
        --timem $timem \
        --mixup ${mixup} \
        --bal none \
        --tstride $tstride \
        --fstride $fstride \
        --imagenet_pretrain $imagenetpretrain \
        --audioset_pretrain $audiosetpretrain \
        --metrics ${metrics} \
        --loss ${loss} \
        --warmup ${warmup} \
        --lrscheduler_start ${lrscheduler_start} \
        --lrscheduler_step ${lrscheduler_step} \
        --lrscheduler_decay ${lrscheduler_decay} \
        --dataset_mean ${dataset_mean} \
        --dataset_std ${dataset_std} \
        --audio_length ${audio_length} \
        --noise ${noise} \
        --num_shared_layers ${num_shared_layers} \
        --use_attn_transform ${use_attn_transform} \
        --use_mlp_transform ${use_mlp_transform} \
        --mlp_kernel_size ${mlp_kernel_size}
done

python ./get_esc_result.py --exp_path ${base_exp_dir}

echo "MiniAST ESC-50 experiment completed!"