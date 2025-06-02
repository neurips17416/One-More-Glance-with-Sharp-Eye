#!/bin/bash

export PATH="/home/work/.workspace/junhasong/miniconda3/envs/llava/bin:$PATH"

export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="opt125-T-FP-L2H1024-CLIPDINOb"

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}" \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --batch_size 16 \
    --version opt

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-5000" \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --batch_size 16 \
    --version opt

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-10000" \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --batch_size 16 \
    --version opt

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-15000" \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --batch_size 16 \
    --version opt

MODEL_NAME="opt125-4MLP4E-s2"

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
MODEL_NAME="opt125-4MLP4E"

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-20000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-15000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-10000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt


MODEL_NAME="opt125-4MLP4E-s2"

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-20000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-15000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/sharegpt/${MODEL_NAME}/checkpoint-10000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt



python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-13b \
    --version llava_v1 \
    --batch_size 2 \
    --query "Provide a brief description of the given image.\n" \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --max_new_tokens 128

python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 2 \
    --query "Provide a brief description of the given image.\n" \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --max_new_tokens 128
'