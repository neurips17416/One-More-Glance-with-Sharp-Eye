#!/bin/bash

# region Templete
# adapter only finetuned
# python llava/eval/model_caption.py \
#     --model_path ./checkpoints/adapter-2e4/opt125-MLP2-ft/checkpoint-25000 \
#     --model_base facebook/opt-125m \
#     --version opt

# adapter and LM finetuned
# python llava/eval/model_caption.py \
#     --model_path ./checkpoints/LM-2e5/opt125-4MLP4E/checkpoint-25000 \
#     --version opt

# python llava/eval/model_caption.py \
#     --model_path liuhaotian/llava-v1.5-7b \
#     --version llava_v1 \
#     --batch_size 2 \
#     --query "Provide a brief description of the given image.\n" \
#     --max_new_tokens 100
# endregion


MODEL_NAME="opt-125m-CrossAttR1"
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt

MODEL_NAME="opt-125m-percR3H768"
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-3500" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-3000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-2500" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt

MODEL_NAME="opt125-4MLP4E"
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-3500" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-3000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-2500" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt

MODEL_NAME="opt125-4MLP4E-rQ"
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-3500" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-3000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-2500" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt


MODEL_NAME="opt125-4MLP4E-rQ-s2"
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-3500" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-3000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-2500" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt


MODEL_NAME="opt125-Q-L2H1024"
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt

MODEL_NAME="opt125-T-L2H1024"
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
MODEL_NAME="opt125-T-L2H1024-rQ"
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt


MODEL_NAME="opt125-coco-4MLP4E"
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-15000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-20000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-25000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt

MODEL_NAME="opt125-coco-4MLP4E-rQ"
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-15000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-20000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-25000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt

MODEL_NAME="opt125-coco-4MLP4E-rQ-s2"
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-15000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-20000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/dci/${MODEL_NAME}/checkpoint-25000" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --version opt
