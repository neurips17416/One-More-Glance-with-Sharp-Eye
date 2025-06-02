#!/bin/bash

export PATH="/home/work/.workspace/junhasong/miniconda3/envs/llava/bin:$PATH"

export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="opt125-T-L2H1024"

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}" \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --version opt

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-5000" \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --version opt

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-10000" \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --version opt

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-15000" \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --version opt

: '
MODEL_NAME="opt125-ViT-4MLP4E"

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-20000" \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --version opt

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-15000" \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --version opt

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-10000" \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --version opt
'