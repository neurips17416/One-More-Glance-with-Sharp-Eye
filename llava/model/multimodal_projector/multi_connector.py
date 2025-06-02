import torch
import torch.nn as nn
from .bert import BertConfig, BertLMHeadModel
from typing import Mapping, Any


class MultiConnectorBase(nn.Module):
    def __init__(self, config, connector_cls):
        super().__init__()
        self.config = config

        num_select_layer = len(config.mm_vision_select_layer.split(','))
        num_encoder = len(config.mm_vision_tower.split(','))

        if num_select_layer > 1:
            self.num_connector = num_select_layer
            assert isinstance(config.vision_width, int), 'vision_width should be an integer'
            vision_width = [config.vision_width for _ in range(num_select_layer)]
        elif num_encoder > 1:
            self.num_connector = num_encoder
            assert isinstance(config.vision_width, list), 'vision_width should be a list'
            vision_width = config.vision_width
        else:
            raise ValueError('Only support multiple connectors')

        # connector_cls could be TransformerBlock or MLPConnector, etc.
        self.connectors = nn.ModuleList([
            connector_cls(config, vision_width[i]) for i in range(self.num_connector)
        ])

        # learnable embeddings for each layer output, except the last layer
        self.layer_embedding = nn.Parameter(
            torch.zeros(self.num_connector - 1, config.hidden_size_projector)
        )

    def forward(self, image_features):
        """
        Args:
            image_features (list[Tensor]): A list of length `num_selected_layers`. 
                Each element in the list is a Tensor of shape [batch_size, num_patches, hidden_dim],
                representing image features for a specific layer.
        """
        projected_features = []
        for i, connector in enumerate(self.connectors):
            projected_features.append(connector(image_features[i]))

        """
        projected_features (list[Tensor]): A list of length `num_selected_layers`.
            Each element in the list is a Tensor of shape [batch_size, num_patches, projected_hidden_dim],
            representing projected image features for a specific layer.

        we want to interleaved the projected features from different layers.
        for example, if num_selected_layers = 2, then the output shape should be: [batch_size, num_patches*2, projected_hidden_dim],
        """
        # NOTE the first connector handles the output of the last layer of the vision tower 
        # refer to the feature_select function in clip_encoder.py

        # if len(self.connectors) > 2, according to GPT, the reshape operation can interleave the projected features
        # interleaved = torch.stack(projected_features, dim=2)
        # B, P, N, C = interleaved.shape
        # interleaved = interleaved.reshape(B, P * N, C)
        
        assert self.num_connector == 2, 'Only support 2 connectors for now'

        feat0 = projected_features[0] # the output of the last layer of the vision tower
        feat1 = projected_features[1] # the output of the 15th layer of the vision tower
        feat1 = feat1 + self.layer_embedding[0].unsqueeze(0).unsqueeze(1)

        B, P, C = feat0.shape
        interleaved_image_features = torch.empty(
            B, P * 2, C,
            dtype=feat0.dtype,
            device=feat0.device
        )
        interleaved_image_features[:, 0::2, :] = feat0
        interleaved_image_features[:, 1::2, :] = feat1
        
        return interleaved_image_features

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if strict:
            result = super().load_state_dict(state_dict, strict=False)
            missing_keys = result.missing_keys
            if len(missing_keys) > 1:
                raise ValueError(f"Missing keys: {missing_keys}")
        else:
            result = super().load_state_dict(state_dict, strict=False)
        return result


class MultiTransformerBlock(MultiConnectorBase):
    def __init__(self, config):
        from .transformer import TransformerBlock
        # pass TransformerBlock as connector_cls
        super().__init__(config, connector_cls=TransformerBlock)


class MultiMLPConnector(MultiConnectorBase):
    def __init__(self, config):
        from .mlp_connector import MLPConnector
        # pass MLPConnector as connector_cls
        super().__init__(config, connector_cls=MLPConnector)
