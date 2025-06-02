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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                        OPTConfig, OPTModel, OPTForCausalLM
from .opt_crossattn import OPTcrossattnModel, OPTcrossattnForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaOPTConfig(OPTConfig):
    model_type = "llava_opt"

class LlavaOPTcrossattnConfig(OPTConfig):
    model_type = "llava_opt_crossattn"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_cross_attention: bool = True



class LlavaOPTBase(LlavaMetaModel):
    config_class = LlavaOPTConfig

    def __init__(self, config: OPTConfig):
        config.hidden_size_projector = config.word_embed_proj_dim
        super().__init__(config)

    @property
    def embed_tokens(self):
        return self.decoder.embed_tokens

class LlavaOPTModel(LlavaOPTBase, OPTModel):
    config_class = LlavaOPTConfig

class LlavaOPTcrossattnModel(LlavaOPTBase, OPTcrossattnModel):
    config_class = LlavaOPTcrossattnConfig
    
    def unfreeze_crossattention_layers(self):
        updated_layers = []
        for name, param in self.named_parameters():
            if 'crossattention' in name:
                updated_layers.append(name)
                param.requires_grad = True
        return updated_layers



class LlavaOPTForCausalLM_Base(LlavaMetaForCausalLM):
    def get_model(self):
        return self.model

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        # FIXME: to solve an error in transfomers/generation/utils.py >> GenerationMixin >> sample 
        kwargs['attention_mask'] = None

        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        images_embeds = kwargs.pop("images_embeds", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if images_embeds is not None:
            inputs['images_embeds'] = images_embeds
        return inputs

class LlavaOPTForCausalLM(LlavaOPTForCausalLM_Base, OPTForCausalLM):
    config_class = LlavaOPTConfig

    def __init__(self, config):
        super(OPTForCausalLM, self).__init__(config)
        self.model = LlavaOPTModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        task_type: Optional[torch.Tensor] = None,
        pre_captions: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                task_type,
                pre_captions
            )

        # TEACHME: OPT does not use position_ids. But, hope that fun-generate() uses it.
        # TEACHME: llava_mpt.py does also not use position_ids and do not define fun-generate(). Why?
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
class LlavaOPTcrossattnForCausalLM(LlavaOPTForCausalLM_Base, OPTcrossattnForCausalLM):
    config_class = LlavaOPTcrossattnConfig

    def __init__(self, config):
        super(OPTcrossattnForCausalLM, self).__init__(config)
        self.model = LlavaOPTcrossattnModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_embeds: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        task_type = None,
        ## for cross-attention
        encoder_hidden_states=None,
        encoder_attention_mask = None,
        cross_attn_head_mask = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        if inputs_embeds is None: # during generating
            assert images_embeds is not None, "images_embeds should not be None."
            text_inputs_embeds, image_inputs_embeds = None, images_embeds
        else: # during training
            assert type(inputs_embeds) == tuple, f"type of inputs_embeds is {type(inputs_embeds)}. It should be tuple when mm_projector_type is crossattention."
            text_inputs_embeds, image_inputs_embeds = inputs_embeds[0], inputs_embeds[1]

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=text_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # for cross-attention
            encoder_hidden_states=image_inputs_embeds,
            encoder_attention_mask=encoder_attention_mask,
            cross_attn_head_mask=cross_attn_head_mask,
        )
    


AutoConfig.register("llava_opt", LlavaOPTConfig)
AutoModelForCausalLM.register(LlavaOPTConfig, LlavaOPTForCausalLM)

AutoConfig.register("llava_opt_crossattn", LlavaOPTcrossattnConfig)
AutoModelForCausalLM.register(LlavaOPTcrossattnConfig, LlavaOPTcrossattnForCausalLM)
