# --------------------------------------------------------
# References:
# RAE: https://github.com/bytetriper/RAE
# --------------------------------------------------------

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional
from math import sqrt
from .encoders.dinov2 import Dinov2withNorm 

class RAE(nn.Module):
    def __init__(self, 
        # ---- encoder configs ----
        encoder_config_path: str = 'facebook/dinov2-base',
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        pretrained_encoder_path: Optional[str] = None,
        # ---- noising, reshaping and normalization-----
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        
        self.encoder = Dinov2withNorm(**encoder_params)
        self.freeze_encoder = Dinov2withNorm(**encoder_params)
        print(f"encoder_config_path: {encoder_config_path}")
        
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2 
        
        if pretrained_encoder_path is not None:
            print(f"Loading pretrained encoder from {pretrained_encoder_path}")
            state_dict = torch.load(pretrained_encoder_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=False)
            
        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location='cpu')
            self.latent_mean = stats.get('mean', None)
            self.latent_var = stats.get('var', None)
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size), mode='bicubic', align_corners=False)
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        cls_token, z = self.encoder(x)
        
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
            
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)
            
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return cls_token, z
    