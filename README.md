# One More Glance with Sharp Eye

Official implementation of **One More Glance with Sharp Eye**. The finetuned weights and the conversation-formatted JSON file are shared via a [Google Drive link](https://drive.google.com/drive/folders/1s6U1bldf3YCkrmCbbbJ5Sx4JHINqECxu?usp=sharing).

## 📦 Installation

Requires **Python 3.10+**.

Install dependencies with:

```bash
pip install .[train]
````

> Make sure you have [PyTorch](https://pytorch.org/get-started/locally/) and [Deepspeed](https://www.deepspeed.ai/) installed properly.

## 📂 Datasets

### Pretraining

1. Download the LLaVA-Pretrain dataset from HuggingFace: [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main)

### Finetuning

1. **COCO Captioning**    
   Download [train2014](http://images.cocodataset.org/zips/train2014.zip) and [val2014](http://images.cocodataset.org/zips/val2014.zip) from [COCO official site](https://cocodataset.org/#home)

2. **ShareGPT-4V**    
   Follow instructions from [ShareGPT4V Data Guide](https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/docs/Data.md)

3. **DCI**    
   Follow instructions from [DCI Data Guide](https://github.com/facebookresearch/DCI?tab=readme-ov-file#setup).

```bash
data_root/
├── LLaVA-Pretrain/
│   ├── images/
│   └── blip_laion_cc_sbu_558k.json
├── COCO/
│   ├── train2014/
│   ├── val2014/
│   ├── coco_karpathy_test_datalist.json
│   ├── coco_karpathy_test_gt.json
│   └── coco_karpathy_train_convs.json
├── sharegpt/
│   ├── coco/
│   ├── llava/
│   ├── sam/
│   ├── share_textvqa/
│   ├── web-celebrity/
│   ├── web-landmark/
│   ├── wikiart/
│   ├── sharegpt_dci_test_datalist.j
│   ├── sharegpt_dci_test_gt.json
│   └── sharegpt_dci_train_conv.json
└── DCI/
    └── images/
```

## 🚀 Training

### Pretraining

Run the MLP-based pretraining script:

```bash
bash scripts/v1_5/mlp/pretrain_opt125m.sh
```

### Finetuning (ShareGPT)

Run the finetuning script for MS-COCO:

```bash
bash scripts/v1_5/mlp/finetune_opt125m.sh
```

Run the finetuning script for ShareGPT:

```bash
bash scripts/v1_5/mlp/sharegpt/finetune_opt125m.sh
```


## 📊 Evaluation

### Single-sentence captioning

This evaluates LLaVA-1.5 7B and our specialist model. The finetuned weights of our model are available [here](https://drive.google.com/drive/folders/1s6U1bldf3YCkrmCbbbJ5Sx4JHINqECxu?usp=sharing).
```
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --query "Provide a single-sentence description of the image." \
    --data_path "./data_root/COCO/coco_test_datalist.json" \
    --image_folder "./data_root/COCO" \
    --batch_size 6 \
    --version llava_v1 

python llava/eval/model_caption.py \
    --model_path "<finetuned_model_path>" \
    --data_path "./data_root/COCO/coco_test_datalist.json" \
    --image_folder "./data_root/COCO" \
    --version opt 
```

### Detailed captioning
```
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --query "Provide a brief description of the given image.\n" \
    --data_path "./data_root/sharegpt/sharegpt_dci_test_datalist.json" \
    --image_folder "./data_root" \
    --batch_size 2 \
    --version llava_v1

python llava/eval/model_caption.py \
    --model_path "<finetuned_model_path>" \
    --data_path "./data_root/sharegpt/sharegpt_dci_test_datalist.json" \
    --image_folder "./data_root" \
    --batch_size 16 \
    --version opt
```

## 📄 License

This project is licensed under the terms of the license in the [LICENSE](./LICENSE) file.