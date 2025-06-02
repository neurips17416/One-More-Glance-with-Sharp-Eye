import torch
import torch.nn as nn
from typing import Any, Mapping

from .bert import BertConfig, BertLMHeadModel

class Qformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.check_cross_attention_freq(config)
        # choose your own BERT config: https://huggingface.co/google/bert_uncased_L-4_H-512_A-8
        encoder_config = BertConfig.from_pretrained(config.bert_model_name)
        encoder_config.encoder_width = config.vision_width
        encoder_config.cross_attention_freq = config.cross_attention_freq        
        encoder_config.query_length = config.num_query_token
        encoder_config.add_cross_attention = True

        ignore_mismatched_sizes = False
        if hasattr(config, 'bert_num_hidden_layers'): # bert_rebase
            ignore_mismatched_sizes = True
            encoder_config.num_hidden_layers = config.bert_num_hidden_layers
            encoder_config.hidden_size = config.bert_hidden_size
            encoder_config.num_attention_heads = config.bert_num_attention_heads
            encoder_config.intermediate_size = config.bert_intermediate_size
        
        self.Qformer = BertLMHeadModel.from_pretrained(
            config.bert_model_name, config=encoder_config, ignore_mismatched_sizes=ignore_mismatched_sizes
        )
        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_token, encoder_config.hidden_size) # BERT's hiden_size
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.Qformer.cls = None
        
        # self.Qformer.config.hidden_size -> encoder_config.hidden_size
        self.llm_proj = nn.Linear(encoder_config.hidden_size, config.hidden_size_projector)

    def forward(self, image_features):
        bs, num_tokens = image_features.size()[:-1]
        query_tokens = self.query_tokens.expand(bs, -1, -1) 
        image_atts = torch.ones((bs, num_tokens), dtype=torch.long).to(image_features.device)

        query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_features,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        
        # query_output.last_hidden_state.shape = (bs, num_query_token, encoder_config.hidden_size)
        # query_tokens.size(1) = num_query_token
        image_features = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        return image_features

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if strict:
            result = super().load_state_dict(state_dict, False)
            # if result inlcudes the missing_key which contains 'position_id', then pass
            # but result includes many other missing_keys, so raise error.
            missing_keys = result.missing_keys
            if len(missing_keys) > 1:
                raise ValueError(f"Missing keys: {missing_keys}")
            if 'position_ids' not in missing_keys[0]:
                raise ValueError(f"Missing keys: {missing_keys}")
        else:
            result = super().load_state_dict(state_dict, False)
        return result
    
    def check_cross_attention_freq(self, config):
        freq_1_list = ['google/bert_uncased_L-8_H-512_A-8', 'google/bert_uncased_L-4_H-512_A-8', \
                       'google/bert_uncased_L-4_H-768_A-12', 'google/bert_uncased_L-4_H-256_A-4',\
                       'google/bert_uncased_L-2_H-768_A-12'] # medium, Small, Mini
        freq_2_list = ['google-bert/bert-base-uncased'] # Bert-base
        if config.bert_model_name in freq_1_list: # Bert-small
            assert config.cross_attention_freq == 1, "For 'google/bert_uncased_L-4_H-512_A-8', cross_attention_freq must be 1"
        elif config.bert_model_name in freq_2_list:    # Bert-base
            assert config.cross_attention_freq == 2, "For 'google-bert/bert-base-uncased', cross_attention_freq must be 2"
        else:
            raise AssertionError("Unsupported BERT model name or cross_attention_freq setting")