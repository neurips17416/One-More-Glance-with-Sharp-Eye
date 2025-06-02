#!/bin/bash

# region Templete
# adapter only finetuned
# python llava/eval/model_caption.py \
#     --model_path ./checkpoints/adapter-2e4/opt125-MLP2-ft/checkpoint-25000 \
#     --model_base facebook/opt-125m \
#     --version opt

# adapter and LM finetuned
# MODEL_NAME="opt125-4MLP4E"
# python llava/eval/model_caption.py \
#     --model_path "./checkpoints/LM-2e5/${MODEL_NAME}/checkpoint-3500" \
#     --version opt

# python llava/eval/model_caption.py \
#     --model_path liuhaotian/llava-v1.5-7b \
#     --version llava_v1 \
#     --batch_size 2 \
#     --query "Provide a brief description of the given image.\n" \
#     --max_new_tokens 100

#     --data_path "./playground/data/DCI/dci_test_datalist.json" \
#     --image_folder "./playground/data/DCI/images" \
# endregion

MODEL_NAME="opt125-T-FP-L2H1024-CLIPDINOb"
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/${MODEL_NAME}/checkpoint-20000" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/${MODEL_NAME}/checkpoint-15000" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/${MODEL_NAME}/checkpoint-10000" \
    --version opt

MODEL_NAME="opt125-TB-15L-L2H1024"
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/${MODEL_NAME}/checkpoint-25000" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/${MODEL_NAME}/checkpoint-20000" \
    --version opt
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/${MODEL_NAME}/checkpoint-15000" \
    --version opt

MODEL_NAME="opt125-TB-FP-L2H1024"
python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/${MODEL_NAME}/checkpoint-10000" \
    --version opt

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/${MODEL_NAME}/checkpoint-15000" \
    --version opt

python llava/eval/model_caption.py \
    --model_path "./checkpoints/LM-2e5/${MODEL_NAME}/checkpoint-20000" \
    --version opt

    

    