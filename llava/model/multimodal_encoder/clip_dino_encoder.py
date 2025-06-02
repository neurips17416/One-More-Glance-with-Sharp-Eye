import torch
import torch.nn as nn

from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .dino_encoder import DinoVisionTower


class ClipDinoVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        encoder_names = self.get_encoder_name(vision_tower)
        
        # kwargs = dict(delay_load=delay_load) # llava/model/builder.py Line 169 >> fix error 
        kwargs = dict(delay_load=False)
        self.clip_vision_tower = CLIPVisionTower(encoder_names[0], args=args, **kwargs)
        self.dino_vision_tower = DinoVisionTower(encoder_names[1], args=args, **kwargs)

        self.is_loaded = True

        # # DEBUG
        # self.dino_image_processor = self.dino_vision_tower.image_processor
        # self.clip_image_processor = self.clip_vision_tower.image_processor

    def get_encoder_name(self, tower_name):
        encoders = tower_name.split(',')
        assert len(encoders) == 2, f"Expected 2 encoders, got {len(encoders)}"
        assert 'clip' in encoders[0], f"Expected clip encoder, got {encoders[1]}"
        return encoders

    @torch.no_grad()
    def forward(self, images):
        clip_features = self.clip_vision_tower(images)
        dino_features = self.dino_vision_tower(images)
        image_features = [clip_features, dino_features] # num_encoder, [batch_size, num_patches, hidden_dim]
        return image_features

    @property
    def image_processor(self):
        # FIXME this is a hack to get the image processor ONLY from the clip vision tower
        return self.clip_vision_tower.image_processor

    @property
    def hidden_size(self):
        return [self.clip_vision_tower.hidden_size, self.dino_vision_tower.hidden_size]