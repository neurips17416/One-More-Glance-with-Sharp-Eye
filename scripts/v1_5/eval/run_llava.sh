export CUDA_VISIBLE_DEVICES=4

python llava/eval/run_llava.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file ./playground/data/coco/val2014/COCO_val2014_000000034180.jpg \
    --query "Provide a very brief description of the given image."