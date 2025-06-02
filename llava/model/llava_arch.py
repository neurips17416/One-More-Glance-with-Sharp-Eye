#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape, print_unfreezed_parameters

from transformers.trainer import logger


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            # train.py usually skips this part, since OPTConfig and LlavaConfig do not have mm_vision_tower.
            # raise ValueError("I always wonder when this will be called.") --> Evaluation! (ex, llava/eval/model_caption.py)
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size_projector, dtype=self.dtype)
                )


    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        # train.py usually calls this part
        vision_tower = model_args.vision_tower

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config = self.add_mm_config(self.config, model_args)

        if getattr(self, 'mm_projector', None) is None: # whether L37 was called
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in self.config.mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size_projector, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size_projector, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
            if 'crossatt' in self.config.mm_projector_type:
                self.unfreeze_crossattention_layers()

        print_unfreezed_parameters(self, print_fun=print, print_names=False) # FIXME: consider local_rank

        if self.config.pretrain_mm_adapter is not None:
            def load_weight_from_path(path):
                loaded_weights = torch.load(path, map_location='cpu')
                def get_w(weights, keywords: list):
                    return {k.split('model.')[1]: v for k, v in weights.items() if any(keyword in k for keyword in keywords)}
                keys_to_match = ['mm_projector', 'vision_resampler', 'crossattention']
                loaded_weights = get_w(loaded_weights, keys_to_match)
                return loaded_weights
            
            if ',' not in self.config.pretrain_mm_adapter:
                loaded_weights = load_weight_from_path(self.config.pretrain_mm_adapter)
            else:
                pretrain_mm_adapter_paths = self.config.pretrain_mm_adapter.split(',')
                loaded_weights = {}
                for i, path in enumerate(pretrain_mm_adapter_paths):
                    connector_weight = load_weight_from_path(path)
                    for k, v in connector_weight.items():
                        new_k = k.replace("mm_projector.", f"mm_projector.connectors.{i}.", 1)
                        loaded_weights.update({new_k: v})
            
            result = self.load_state_dict(loaded_weights, strict=False)
            print(f"Unexpected_keys keys: {result.unexpected_keys}" if len(result.unexpected_keys) > 0 \
                else "Successfully loaded pre-trained multimodal adapter (num: {}).".format(len(loaded_weights)))
    
    def add_mm_config(self, config, model_args):
        """
        Add multimodal configuration to the model configuration.
        
        Args:
        config (dict): The transforemer model configuration, ex) OPTConfig, LlamaConfig.
        model_args (dict): The model arguments in `train.py`, which can be changed by the arguments
        """
        config.use_mm_proj = True
        config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        config.mm_hidden_size = self.get_vision_tower().hidden_size
        config.mm_vision_select_layer = model_args.mm_vision_select_layer
        config.mm_vision_select_feature = model_args.mm_vision_select_feature
        config.mm_patch_merge_type = model_args.mm_patch_merge_type
        config.pretrain_mm_adapter = model_args.pretrain_mm_adapter
        config.mm_vision_s2 = model_args.mm_vision_s2
        config.mm_vision_s2_scales = model_args.mm_vision_s2_scales
        config.tokenizer_pad_token_id = model_args.tokenizer_pad_token_id
        config.tokenizer_bos_token_id = model_args.tokenizer_bos_token_id
        config.tokenizer_eos_token_id = model_args.tokenizer_eos_token_id

        if 'mlp' in config.mm_projector_type:
            config.vision_width = self.get_vision_tower().hidden_size
        
        if config.mm_projector_type == 'abstractor':
            config.num_learnable_queries = model_args.num_learnable_queries
            config.ab_num_hidden_layers = model_args.ab_num_hidden_layers
            config.ab_num_attention_heads = model_args.ab_num_attention_heads
            config.ab_intermediate_size = model_args.ab_intermediate_size
            config.attention_probs_dropout_prob = getattr(model_args, 'attention_probs_dropout_prob', 0.)
            config.grid_size = getattr(model_args, 'grid_size', 32)
            config.add_v2t_pos_emb = getattr(model_args, 'add_v2t_pos_emb', False)
            config.use_cls_token = getattr(model_args, 'use_cls_token', False)
            config.layer_norm_eps =getattr(model_args, 'layer_norm_eps', 1e-6)

        if config.mm_projector_type == 'qformer' or config.mm_projector_type == 'transformer' :
            config.vision_width = self.get_vision_tower().hidden_size
            config.bert_model_name = model_args.bert_model_name
            config.cross_attention_freq = model_args.cross_attention_freq 
            config.num_query_token = model_args.num_query_token
            if model_args.bert_rebase or model_args.transformer_rebase:
                config.bert_num_hidden_layers = model_args.bert_num_hidden_layers
                config.bert_hidden_size = model_args.bert_hidden_size
                config.bert_num_attention_heads = model_args.bert_num_attention_heads
                config.bert_intermediate_size = model_args.bert_intermediate_size

        if config.mm_projector_type == 'crossattention_perceiver':
            config.num_learnable_latents = model_args.num_learnable_latents
            config.perceiver_depth = model_args.perceiver_depth
            config.perceiver_hidden_size = model_args.perceiver_hidden_size
            config.perceiver_num_heads = model_args.perceiver_num_heads
            config.num_media_embeds = getattr(model_args, 'num_media_embeds', 1)
            config.perceiver_ff_mult = getattr(model_args, 'perceiver_ff_mult', 4)

        if config.mm_projector_type == 'detr':
            config.vision_width = self.get_vision_tower().hidden_size
            config.detr_num_hidden_layers = model_args.detr_num_hidden_layers
            config.detr_hidden_size = model_args.detr_hidden_size
            config.detr_num_attention_heads = model_args.detr_num_attention_heads
            config.detr_intermediate_size = model_args.detr_intermediate_size
            config.detr_num_queries = model_args.detr_num_queries
            config.detr_pre_caption_input = model_args.detr_pre_caption_input
            config.detr_visual_connector = model_args.detr_visual_connector

        return config


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, pre_captions=None):
        image_features = self.get_model().get_vision_tower()(images)
        
        if pre_captions is not None:
            return self.get_model().mm_projector(image_features, pre_captions)
        return self.get_model().mm_projector(image_features)

    def encode_images_with_pre_caption(self, images, pre_captions):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features, pre_captions)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, task_type=None, pre_captions=None
    ):
        vision_tower = self.get_vision_tower()
        mm_uses_cross_attention = 'crossatt' in self.config.mm_projector_type
        
        # For 'class GenerationMixin >> fun generate'
        # second or later forward, where 'input_ids' is a single generated word, and 'images' are None.
        if (vision_tower is None or images is None or input_ids.shape[1] == 1): #  and (not mm_uses_cross_attention)
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # if detr_pre_caption_input is True, 
        pre_captions_enabled = getattr(self.config, 'detr_pre_caption_input', False)
        # pre_captions shape = (batch size x max caption length)
        # attention_mask for pre_captions can be calculated self.get_model().config.tokenizer_pad_token_id
        if pre_captions_enabled:
            pre_captions_attention_mask = (pre_captions != self.config.tokenizer_pad_token_id).to(pre_captions.device)
            if self.config.tokenizer_bos_token_id == self.config.tokenizer_pad_token_id:
                pre_captions_attention_mask[:, 0] = True
            pre_captions_embeds = self.get_model().embed_tokens(pre_captions)
            pre_captions_tuple = (pre_captions_embeds, pre_captions_attention_mask)
            # pre_captions = [cur_pre_captions[cur_pre_captions_attention_mask] for cur_pre_captions, cur_pre_captions_attention_mask in zip(pre_captions, pre_captions_attention)]
        else:
            pre_captions_tuple = None

        # first forward, where natural questions and images are processed.
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images, pre_captions_tuple if pre_captions_enabled else None)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # in case that [llava/mm_utils.py >> fun process_images] is used.
            image_features = self.encode_images(images, pre_captions_tuple if pre_captions_enabled else None)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0: # For 'class GenerationMixin >> fun generate'
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = [] # current text_inputs_indices without image tokens
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]                              # Size[35], [18]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))  # Size[53, 4096]
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)       # Size[35, 4096], [18, 4096]
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1): # num of splited sqeuences
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images and (not mm_uses_cross_attention): # except the last sequence
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds] # Size[35, 4096], [576(24^2), 4096], [18, 4096]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)                   # Size[35, 4096], [576(24^2), 4096], [18, 4096]
            cur_new_labels = torch.cat(cur_new_labels)                               # Size[35, 4096], [576(24^2), 4096], [18, 4096]

            new_input_embeds.append(cur_new_input_embeds)                            # Size[629, 4096]
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        if mm_uses_cross_attention: 
            new_input_embeds = (new_input_embeds, image_features)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
