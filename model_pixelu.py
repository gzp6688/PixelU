# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# JiT: https://github.com/LTH14/JiT
# U-REPA: https://github.com/YuchuanTian/U-REPA
# --------------------------------------------------------

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm
from util.projectors import ProjectionLayer

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Downsample(nn.Module):
    """
    Spatial downsampling for Image features using Conv2d + PixelUnshuffle.
    """
    def __init__(self, n_feat, out_feat):
        super(Downsample, self).__init__()
        if out_feat is None:
            out_feat = n_feat * 2
        self.n_feat = n_feat
        self.out_feat = out_feat
        # PixelUnshuffle(2) increases channel by 4x, so we output out_feat // 4 from conv
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, out_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x, H, W):
        # x: (B, N, C) -> Need (B, C, H, W)
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.body(x)
        # Return (B, N_new, C_new)
        return x.view(B, self.out_feat, -1).permute(0, 2, 1)

class Upsample(nn.Module):
    """
    Spatial upsampling for Image features using Conv2d + PixelShuffle.
    """
    def __init__(self, n_feat, out_channels):
        super(Upsample, self).__init__()
        self.n_feat = n_feat
        self.out_feat = out_channels
        # PixelShuffle(2) decreases channel by 4x, so we need out_channels * 4 from conv
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, out_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.body(x)
        return x.view(B, self.out_feat, -1).permute(0, 2, 1)

class TokenResample(nn.Module):
    """
    Resampling for Non-Spatial Tokens (e.g., In-context Class Tokens).
    Only adjusts channel dimensions, keeps sequence length unchanged.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # If dimension changes, project it; otherwise Identity
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.proj(x)


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings

from torch.nn.functional import scaled_dot_product_attention
# def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
#     L, S = query.size(-2), key.size(-2)
#     scale_factor = 1 / math.sqrt(query.size(-1))
#     attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

#     with torch.cuda.amp.autocast(enabled=False):
#         attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
#     attn_weight += attn_bias
#     attn_weight = torch.softmax(attn_weight, dim=-1)
#     attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
#     return attn_weight @ value

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, skip=False):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Skip connection handling for U-Net Decoder blocks
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None
        
    @torch.compile
    def forward(self, x, c, feat_rope=None, skip=None):
        # Handle Skip Connection
        if self.skip_linear is not None and skip is not None:
            # Skip contains [Token, Image], x contains [Token, Image]
            # Concatenate along channel dim
            x = torch.cat([x, skip], dim=-1)
            x = self.skip_linear(x)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class UiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=[10, 16, 10],  # [Encoder, Middle, Decoder]
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        skip=True,
        skip_all_blocks=True,
        encoder_depth=4,
        z_dims=[],
        projector_dim=2048,
        projection_layer_type="conv",
        proj_kwargs_kernel_size=3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.num_classes = num_classes
        self.depth = depth

        self.encoder_depth = encoder_depth
        self.z_dims = z_dims
        self.projector_dim = projector_dim
        self.proj_kwargs_kernel_size = proj_kwargs_kernel_size
        self.projection_layer_type = projection_layer_type

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        
        # Image Patch Embedding (Bottleneck)
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )
        
        # Middle Stage RoPE (Half Resolution)
        # Note: We assume num_heads scales if hidden_dim scales to keep head_dim constant, 
        # or we re-calculate based on fixed num_heads. Here assuming fixed num_heads.
        mid_head_dim = hidden_size // num_heads
        self.feat_rope_mid = VisionRotaryEmbeddingFast(
            dim=mid_head_dim // 2,
            pt_seq_len=hw_seq_len // 2, 
            num_cls_token=self.in_context_len
        )

        # ---------------- Blocks ----------------
        self.blocks0 = nn.ModuleList([]) # Encoder
        self.blocks1 = nn.ModuleList([]) # Middle
        self.blocks2 = nn.ModuleList([]) # Decoder

        # Encoder Blocks
        for i in range(depth[0]):
            self.blocks0.append(
                JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_drop=0.0, proj_drop=0.0, skip=False)
            )

        # Middle Blocks 
        for i in range(depth[1]):
            self.blocks1.append(
                JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop, skip=False)
            )

        # Decoder Blocks (with skip)
        self.blocks2.append(
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_drop=0.0, proj_drop=0.0, skip=skip)
        )
        for i in range(depth[2]-1):
            self.blocks2.append(
                JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_drop=0.0, proj_drop=0.0, skip=skip_all_blocks if skip else False)
            )

        if len(z_dims) > 0:
            self.projectors = nn.ModuleList([
                ProjectionLayer(projection_layer_type, hidden_size=hidden_size, z_dim=z_dim, projector_dim=projector_dim, proj_kwargs_kernel_size=proj_kwargs_kernel_size) for z_dim in z_dims
            ])
        else:
            self.projectors = None

        # ---------------- Samplers ----------------
        # Spatial Down/Upsamplers for Images
        self.downsampler = Downsample(hidden_size, hidden_size)
        self.upsampler = Upsample(hidden_size, hidden_size)

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation
        for stage in [self.blocks0, self.blocks1, self.blocks2]:
            for block in stage:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        """
        # class and time embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb
        
        # Image Feature Embedding
        x = self.x_embedder(x)
        x += self.pos_embed

        skips = []
        block_idx = 0
        zs_tilde = None
        zs_tilde_original = None
        # --- Stage 0: Encoder ---
        for block in self.blocks0:
            # RoPE handles tokens by skipping rotation for the first in_context_len tokens
            x = block(x, c, feat_rope=self.feat_rope)
            skips.append(x) # Save [Token, Image] for skip connection
            if self.projectors is not None and (block_idx + 1) == self.encoder_depth:
                # REPA only aligns the patch tokens, skip the in-context tokens in the front
                if x.shape[1] == self.x_embedder.num_patches:
                    x_ = x
                else:
                    x_ = x[:, self.in_context_len:]
                zs_tilde = [projector(x_) for projector in self.projectors]
                zs_tilde_original = [x_.clone() for _ in self.projectors]
            block_idx += 1

        _, N, _ = x.shape
        H = W = int(N**0.5)

        # Image -> Spatial Downsample
        x = self.downsampler(x, H, W)
        
        # Inject In-context Class Tokens
        if self.in_context_len > 0:
            in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
            in_context_tokens += self.in_context_posemb
            x = torch.cat([in_context_tokens, x], dim=1) # [B, T_cls + T_img, D]
        
        # --- Stage 1: Middle ---
        for block in self.blocks1:
            x = block(x, c, feat_rope=self.feat_rope_mid)
            if self.projectors is not None and (block_idx + 1) == self.encoder_depth:
                # REPA only aligns the patch tokens, skip the in-context tokens in the front
                if x.shape[1] == self.x_embedder.num_patches // 4: 
                    x_ = x
                else:
                    x_ = x[:, self.in_context_len:]
                zs_tilde = [projector(x_) for projector in self.projectors]
                zs_tilde_original = [x_.clone() for _ in self.projectors]
            block_idx += 1

        # --- Split, Upsample, Merge ---
        if self.in_context_len > 0:
            tokens = x[:, :self.in_context_len]
            images = x[:, self.in_context_len:]
        else:
            tokens = None
            images = x
        
        _, N, _ = images.shape
        H_mid = W_mid = int(N**0.5)

        # Image -> Spatial Upsample
        images_up = self.upsampler(images, H_mid, W_mid)
        
        # Token -> Channel Resample
        if tokens is not None:
            x = torch.cat([tokens, images_up], dim=1)
        else:
            x = images_up

        # --- Stage 2: Decoder ---
        for block in self.blocks2:
            skip = skips.pop()
            incontext_token = x[:, :self.in_context_len]
            skip = torch.cat([incontext_token, skip], dim=1)
            x = block(x, c, feat_rope=self.feat_rope_incontext, skip=skip)
            if self.projectors is not None and (block_idx + 1) == self.encoder_depth:
                # REPA only aligns the patch tokens, skip the in-context tokens in the front
                if x.shape[1] == self.x_embedder.num_patches:
                    x_ = x
                else:
                    x_ = x[:, self.in_context_len:]
                zs_tilde = [projector(x_) for projector in self.projectors]
                zs_tilde_original = [x_.clone() for _ in self.projectors]
            block_idx += 1

        # --- Output ---
        # Strip tokens to return image only
        if self.in_context_len > 0:
            x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        if zs_tilde is None:
            return output, zs_tilde, zs_tilde_original, None
        else:
            return output, self.scale_zs(zs_tilde)[0], self.scale_zs(zs_tilde_original)[0], None
    
    @torch.compile
    def scale_zs(self, zs):
        project_size = 16
        ls = list()
        for z in zs:
            B, N, C = z.shape
            H = W = round(int(N)**0.5)# assume H==W
            if N != project_size * project_size:
                z_reshaped = z.permute(0,2,1).reshape(B, C, H, W)
                z_reshaped = F.interpolate(z_reshaped, size=(project_size, project_size), mode='bicubic')
                ls.append(z_reshaped.reshape(B, C, -1).permute(0,2,1))
            else:
                ls.append(z) # do nothing
        return ls
    

def UiT_B_16(**kwargs):
    return UiT(depth=[4, 4, 4], hidden_size=768, num_heads=12, 
               bottleneck_dim=128, in_context_len=32, patch_size=16, skip=True, skip_all_blocks=True,
               encoder_depth=6, z_dims=[768], projection_layer_type="conv", **kwargs)

def UiT_L_16(**kwargs):
    return UiT(depth=[8, 8, 8], hidden_size=1024, num_heads=16, 
               bottleneck_dim=128, in_context_len=32, patch_size=16, skip=True, skip_all_blocks=True,
               encoder_depth=12, z_dims=[768], projection_layer_type="conv", **kwargs)

def UiT_H_16(**kwargs):
    return UiT(depth=[8, 20, 8], hidden_size=1280, num_heads=16, 
               bottleneck_dim=256, in_context_len=32, patch_size=16, skip=True, skip_all_blocks=True,
               encoder_depth=18, z_dims=[768], projection_layer_type="conv", **kwargs)

def UiT_H_32(**kwargs):
    return UiT(depth=[8, 20, 8], hidden_size=1280, num_heads=16, 
               bottleneck_dim=256, in_context_len=32, patch_size=32, skip=True, skip_all_blocks=True,
               encoder_depth=18, z_dims=[768], projection_layer_type="conv", **kwargs)


PixelU_models = {

    'PixelU-B-16': UiT_B_16,
    'PixelU-L-16': UiT_L_16,
    'PixelU-H-16': UiT_H_16,
    'PixelU-H-32': UiT_H_32,

}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = UiT_B_16().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256).to(device)
    t = torch.randn(batch_size).to(device)
    y = torch.randint(0, 1000, (batch_size,)).to(device)
    out, zs, zs_tile, _ = model(x, t, y)
    
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")