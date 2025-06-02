"""
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Based on huggingface code base
 * https://github.com/lucidrains/flamingo-pytorch/blob/main/flamingo_pytorch/flamingo_pytorch.py
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many

def exists(val):
    return val is not None

def FeedForward(input_dim, output_dim, mult = 4):
    inner_dim = int(input_dim * mult)
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, output_dim, bias = False)
    )

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()

        num_latents = config.num_learnable_latents      # number of learnable queries
        dim = config.perceiver_hidden_size              # dimension of output 
        heads = config.perceiver_num_heads              # number of attention heads (8, dim=512, inner_dim=dim_headxheads=512)
        dim_head = dim // heads                         # dimension of each head (64)
        num_media_embeds = config.num_media_embeds      # number of images (1)
        depth = config.perceiver_depth                  # perceiver resampler depth (2)
        ff_mult = config.perceiver_ff_mult              # feedforward multiplier (4, dim=512, inner_dim=2048)
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
            
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))

        self.vis_proj = nn.Linear(config.mm_hidden_size, dim)

        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(input_dim = dim, output_dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

        self.llm_proj = nn.Linear(dim, config.hidden_size)

    def forward(self, image_features):
        x = self.vis_proj(image_features)
        assert x.ndim == 3, 'image features must be 3D'
        x = rearrange(x, 'b n d -> b 1 n d')

        times = x.shape[1]
        x = x + self.media_pos_emb[:times]

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.norm(latents.squeeze(1))

        latents = self.llm_proj(latents)

        return latents