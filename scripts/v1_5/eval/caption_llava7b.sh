# bash scripts/v1_5/eval/coco_240609.sh

# region LLaVA-v1.5-7b Generalist on COCO

## BRIEF DESCRIPTION
export CUDA_VISIBLE_DEVICES=3
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 8 \
    --query "Provide a very brief description of the given image.\n" \
    --max_new_tokens 50
# Bleu_4: 35.69
# METEOR: 28.88
# CIDEr: 123.08
# SPICE: 22.91

export CUDA_VISIBLE_DEVICES=3
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 6 \
    --query "Provide a very brief description of the given image.\n" \
    --max_new_tokens 128
# Bleu_4: 34.76
# METEOR: 28.79
# CIDEr: 122.95
# SPICE: 22.94

export CUDA_VISIBLE_DEVICES=5
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 6 \
    --query "Provide a very brief description of the given image.\n" \
    --max_new_tokens 256
# Bleu_4: 34.76
# METEOR: 28.79
# CIDEr: 122.95
# SPICE: 22.94

export CUDA_VISIBLE_DEVICES=4
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 6 \
    --query "Provide a brief description of the given image.\n" \
    --max_new_tokens 128
# Bleu_4: 0.072
# METEOR: 0.220
# CIDEr: 0.098
# SIDER: N/A

export CUDA_VISIBLE_DEVICES=6
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 6 \
    --query "Provide a brief description of the given image.\n" \
    --max_new_tokens 256
# Bleu_4: 0.073
# METEOR: 0.221
# CIDEr: 0.098
# SIDER: N/A



## SINGLE SENTENCE
export CUDA_VISIBLE_DEVICES=4
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 8 \
    --query "Provide a single-sentence description of the image within 25 words.\n"
# Bleu_4: 36.04
# METEOR: 28.83
# CIDEr: 124.57
# SPICE: 23.27

export CUDA_VISIBLE_DEVICES=5
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 8 \
    --query "Provide a single-sentence description of the image, using 25~30 words.\n"
# Bleu_4: 35.17
# METEOR: 29.33
# CIDEr: 124.49
# SPICE: 23.74

export CUDA_VISIBLE_DEVICES=4
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 6 \
    --query "Provide a single-sentence description of the image within 40 words.\n"
# Bleu_4: 35.55
# METEOR: 29.15
# CIDEr: 124.38
# SPICE: 23.49


export CUDA_VISIBLE_DEVICES=4
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 4 \
    --query "Provide a single-sentence description of the image within 30 words.\n" \
    --num_beams 5 \
# Bleu_4: 36.81
# METEOR: 28.86
# CIDEr: 126.73
# SPICE: 23.26

export CUDA_VISIBLE_DEVICES=4
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 8 \
    --query "Provide a single-sentence description of the image within 30 words.\n" \
    --do_sample True \
    --temperature 0.2 \
    --num_beams 1 \
# Bleu_4: 32.13
# METEOR: 28.70
# CIDEr: 116.41
# SPICE: 23.07

export CUDA_VISIBLE_DEVICES=4
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 8 \
    --query "Provide a single-sentence description of the image within 30 words.\n" \
    --do_sample True \
    --temperature 0.7 \
    --num_beams 1 \
# Bleu_4: 32.13
# METEOR: 28.70
# CIDEr: 116.41
# SPICE: 23.07


export CUDA_VISIBLE_DEVICES=3
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 6 \
    --query "Provide a single-sentence description of the image.\n"\
    --max_new_tokens 50
# --max_new_tokens 50, 126, 256 모두 동일한 결과 나옴
# Bleu_4: 36.25
# METEOR: 29.09
# CIDEr: 125.51
# SPICE: 23.36

export CUDA_VISIBLE_DEVICES=3
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 6 \
    --query "Provide a single-sentence description of the image using over 50 words.\n"\
    --max_new_tokens 126
# Bleu_4: 33.16
# METEOR: 29.66
# CIDEr: 121.13
# SPICE: 24.05

export CUDA_VISIBLE_DEVICES=4
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 6 \
    --query "Provide a single-sentence description of the image that is over 50 words long.\n"\
    --max_new_tokens 126
# METEOR: 29.42
# ROUGE_L: 57.06
# CIDEr: 120.49
# SPICE: 23.82

# endregion


# region LLaVA-v1.5-7b Generalist on DCI

python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --version llava_v1 \
    --batch_size 2 \
    --query "Provide a brief description of the given image.\n" \
    --data_path "./playground/data/DCI/dci_test_datalist.json" \
    --image_folder "./playground/data/DCI/images" \
    --max_new_tokens 128

# endregion


# region LLaVA-v1.5-7b Generalist on ShareGPT

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

# endregion





# region LLaVA-v1.5-13b Specialist on COCO

export CUDA_VISIBLE_DEVICES=4
python llava/eval/model_caption.py \
    --model_path "./checkpoints/llava-v1.5-7b-COCO/checkpoint-5000" \
    --version v1 \
    --batch_size 8 \
    --force_pred True \  
    --max_new_tokens 50

export CUDA_VISIBLE_DEVICES=5
python llava/eval/model_caption.py \
    --model_path "./checkpoints/llava-v1.5-7b-COCO/checkpoint-10000" \
    --version v1 \
    --batch_size 8 \
    --force_pred True \
    --max_new_tokens 50

# endregion

# region LLaVA-v1.5-13b Specialist on ShareGPT


export CUDA_VISIBLE_DEVICES=6
python llava/eval/model_caption.py \
    --model_path "./checkpoints/llava-v1.5-7b-ShareGPT/checkpoint-10000" \
    --version v1 \
    --batch_size 8 \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --force_pred True \
    --max_new_tokens 128

export CUDA_VISIBLE_DEVICES=7
python llava/eval/model_caption.py \
    --model_path "./checkpoints/llava-v1.5-7b-ShareGPT/checkpoint-15000" \
    --version v1 \
    --batch_size 8 \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --force_pred True \
    --max_new_tokens 128

export CUDA_VISIBLE_DEVICES=7
python llava/eval/model_caption.py \
    --model_path "./checkpoints/llava-v1.5-7b-ShareGPT/checkpoint-20000" \
    --version v1 \
    --batch_size 8 \
    --data_path "./playground/data/DCI/sharegpt_dci_test_datalist.json" \
    --image_folder "./playground/data/" \
    --force_pred True \
    --max_new_tokens 128

# endregion