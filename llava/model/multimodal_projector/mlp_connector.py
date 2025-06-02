import torch
import torch.nn as nn
import re



def MLPConnector(config, my_vision_width=None):
    vision_width = config.mm_hidden_size if my_vision_width is None else my_vision_width
    
    # plain MLP
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', config.mm_projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(vision_width, config.hidden_size_projector)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size_projector, config.hidden_size_projector))
        return nn.Sequential(*modules)

    # Bottleneck MLP
    mlpB_gelu_match = re.match(r'^mlp(\d+)x_bott(\d+)x_gelu$', config.mm_projector_type)
    expansion: int = int(mlpB_gelu_match.group(2))
    if mlpB_gelu_match:
        mlp_depth = int(mlpB_gelu_match.group(1))
        modules = [nn.Linear(vision_width, config.hidden_size_projector * expansion)]
        for _ in range(1, mlp_depth-1):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size_projector * expansion, config.hidden_size_projector * expansion))
        modules.append(nn.Linear(config.hidden_size_projector * expansion, config.hidden_size_projector))
        return nn.Sequential(*modules)
    
    raise ValueError(f'Unknown config.mm_projector_type: {config.mm_projector_type}')