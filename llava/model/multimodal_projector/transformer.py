import torch
import torch.nn as nn
from .bert import BertConfig, BertLMHeadModel
from typing import Mapping, Any


class TransformerBlock(nn.Module):
    def __init__(self, config, my_vision_width=None):
        super().__init__()
        self.config = config

        # Required attributes
        required_attrs = [
                'bert_num_hidden_layers',
                'bert_hidden_size',
                'bert_num_attention_heads',
                'bert_intermediate_size'
        ]
        for attr in required_attrs:
            assert hasattr(config, attr), f'{attr} is required for TransformerBlock'

        vision_width = config.vision_width if my_vision_width is None else my_vision_width
        self.visual_proj = nn.Linear(vision_width, config.bert_hidden_size)

        # transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.bert_hidden_size, 
            nhead=config.bert_num_attention_heads,
            dim_feedforward=config.bert_intermediate_size,
            batch_first=True,
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.bert_num_hidden_layers
        )

        self.llm_proj = nn.Linear(config.bert_hidden_size, config.hidden_size_projector)

    def forward(self, image_features):
        projected_features = self.visual_proj(image_features)
        outputs = self.transformer_encoder(projected_features)
        image_features = self.llm_proj(outputs)
        return image_features

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if strict:
            result = super().load_state_dict(state_dict, False)
            missing_keys = result.missing_keys
            if len(missing_keys) > 1:
                raise ValueError(f"Missing keys: {missing_keys}")
        else:
            result = super().load_state_dict(state_dict, False)
        return result


class BERTConnector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        encoder_config = BertConfig.from_pretrained(config.bert_model_name)
        encoder_config.encoder_width = config.vision_width
        encoder_config.add_cross_attention = False  
        ignore_mismatched_sizes = False
        if hasattr(config, 'bert_num_hidden_layers'): 
            ignore_mismatched_sizes = True
            encoder_config.num_hidden_layers = config.bert_num_hidden_layers
            encoder_config.hidden_size = config.bert_hidden_size
            encoder_config.num_attention_heads = config.bert_num_attention_heads
            encoder_config.intermediate_size = config.bert_intermediate_size
        
        self.connector = BertLMHeadModel.from_pretrained(
            config.bert_model_name, config=encoder_config, ignore_mismatched_sizes=ignore_mismatched_sizes
        )
        
        self.visual_proj = nn.Linear(config.vision_width, encoder_config.hidden_size)
        
        self.connector.bert.embeddings.word_embeddings = None
        self.connector.bert.embeddings.position_embeddings = None
        for layer in self.connector.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.connector.cls = None
        
        self.llm_proj = nn.Linear(encoder_config.hidden_size, config.hidden_size_projector)

    def forward(self, image_features):
        bs, num_tokens = image_features.size()[:-1]
        projected_features = self.visual_proj(image_features)
        attention_mask = torch.ones((bs, num_tokens), dtype=torch.long).to(image_features.device)
        # token_type_ids = torch.zeros((bs, num_tokens), dtype=torch.long).to(image_features.device)

        outputs = self.connector.bert(
            query_embeds=projected_features,
            attention_mask=attention_mask,
            # token_type_ids = token_type_ids,
            return_dict=True,
        )
        
        image_features = self.llm_proj(outputs.last_hidden_state)
        
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