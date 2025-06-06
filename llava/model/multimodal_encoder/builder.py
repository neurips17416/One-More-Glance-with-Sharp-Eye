import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .dino_encoder import DinoVisionTower
from .clip_dino_encoder import ClipDinoVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    
    if ',' not in vision_tower: # Single vision tower
        if "openai/clip" in vision_tower.lower():
            is_absolute_path_exists = os.path.exists(vision_tower)
            use_s2 = getattr(vision_tower_cfg, 'mm_vision_s2', False)
            if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
                if use_s2:
                    return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
                else:
                    return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
                
        if "dinov2" in vision_tower.lower():
            return DinoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    else: # Multiple vision towers
        if "clip" in vision_tower.lower() and "dinov2" in vision_tower.lower():
            return ClipDinoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
