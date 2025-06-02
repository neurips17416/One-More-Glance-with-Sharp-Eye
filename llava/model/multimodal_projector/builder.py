import torch
import torch.nn as nn

from .abstractor import Abstractor
from .qformer import Qformer
from .transformer import BERTConnector, TransformerBlock
from .perceiver import PerceiverResampler
from .mlp_connector import MLPConnector
from .multi_connector import MultiTransformerBlock, MultiMLPConnector
from .detr_decoder import DETRconnector, FP_DETRconnector

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if not (',' in str(config.mm_vision_select_layer) or ',' in config.mm_vision_tower):
        if projector_type == 'identity':
            return nn.Linear(config.mm_hidden_size, config.hidden_size_projector)
        
        if projector_type == 'crossattention':
            return nn.Linear(config.mm_hidden_size, config.hidden_size)
        
        if 'mlp' in projector_type:
            return MLPConnector(config)

        if projector_type == 'identity':
            return IdentityMap()
        
        if projector_type == 'abstractor':
            return Abstractor(config)
        
        if projector_type == 'qformer':
            return Qformer(config)
        
        if projector_type == 'transformer':
            return TransformerBlock(config)
            # return BERTConnector(config)
        
        if projector_type == 'crossattention_perceiver':
            return PerceiverResampler(config)
        
        if projector_type == 'detr':
            return DETRconnector(config)
    
    else:
        if projector_type == 'transformer':
            return MultiTransformerBlock(config)
        
        if 'mlp' in projector_type:
            return MultiMLPConnector(config)
        
        if 'detr' in projector_type:
            return FP_DETRconnector(config)
        
    raise ValueError(f'Unknown projector type: {projector_type}')
