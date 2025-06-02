import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        # FIXME I do not understand why this is needed
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        
        # fine-tune the vision tower
        if getattr(args, 'unfreeze_mm_vision_tower', False):
            self.vision_tower.requires_grad_(True)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name) # used in [train.py >> LazySupervisedDataset]
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    """
    def feature_select(self, image_forward_outs):
        
        # Select the feature from the image forward outputs
        # image_forward_outs: CLIPVisionModel forward outputs, [batch_size, num_patches, hidden_dim]
        
        if ',' not in str(self.select_layer):
            self.select_layer = int(self.select_layer)
            image_features = image_forward_outs.hidden_states[self.select_layer]
            if self.select_feature == 'patch':
                image_features = image_features[:, 1:]
            elif self.select_feature == 'cls_patch':
                image_features = image_features
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
        else:
            select_layers = list(map(int, self.select_layer.split(',')))
            assert select_layers[0] in [len(image_forward_outs.hidden_states)-2, -2], 'the first index should be -2, which means the output of the last layer'
            image_features = []
            for layer_idx in select_layers:
                layer_features = image_forward_outs.hidden_states[layer_idx]
                if self.select_feature == 'patch':
                    layer_features = layer_features[:, 1:]
                elif self.select_feature == 'cls_patch':
                    layer_features = layer_features
                else:
                    raise ValueError(f'Unexpected select feature: {self.select_feature}')
                image_features.append(layer_features) # num_selected_layers, [batch_size, num_patches, hidden_dim]
            # image_features = torch.stack(features_list, dim=1)  # [batch_size, num_selected_layers, num_patches, hidden_dim]
            
        return image_features
    """
    
    def feature_select(self, image_forward_outs):
        """
            Select the feature from the image forward outputs
            image_forward_outs: CLIPVisionModel forward outputs, [batch_size, num_patches, hidden_dim]
        """
        def process_features(features):
            if self.select_feature == 'patch':
                return features[:, 1:]
            elif self.select_feature == 'cls_patch':
                return features
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        if ',' not in str(self.select_layer):
            self.select_layer = int(self.select_layer)
            image_features = process_features(image_forward_outs.hidden_states[self.select_layer])
        else:
            select_layers = list(map(int, self.select_layer.split(',')))
            assert select_layers[0] in [len(image_forward_outs.hidden_states)-2, -2], 'the first index should be -2, which means the output of the last layer'
            image_features = [process_features(image_forward_outs.hidden_states[layer_idx]) 
                            for layer_idx in select_layers] # num_selected_layers, [batch_size, num_patches, hidden_dim]
            # image_features = torch.stack(features_list, dim=1)  # [batch_size, num_selected_layers, num_patches, hidden_dim]
            
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs)
            if not isinstance(image_features, list):
                image_features = image_features.to(images.dtype)
            else:
                image_features = [f.to(images.dtype) for f in image_features]
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):

        self.s2_scales = getattr(args, 'mm_vision_s2_scales', '224,448')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        super().__init__(vision_tower, args, delay_load)

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
