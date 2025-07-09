# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
# Modified by Victor Wåhlstrand (2025) for the STRAP project.

from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from timm.models.vision_transformer import Block
from timm.layers.patch_embed import PatchEmbed
from models.backbones.utils.mask import MaskEmbed, remove_empty_patches_and_pad, random_masking
from models.backbones.utils.pos_embed import get_2d_sincos_pos_embed
from typing import Tuple, Callable, List, Type


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer: Type[nn.Module] = nn.LayerNorm, norm_pix_loss=False, threshold=0.5,
                 mask_ratio=0.5):
        super().__init__()

        self.mask_ratio = mask_ratio

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.in_chans = in_chans
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.mask_embed = MaskEmbed(
            img_size=img_size,
            patch_size=patch_size,
            threshold=threshold
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 *C)
        """
        p = int(self.patch_embed.patch_size[0])
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = int(self.patch_embed.patch_size[0])
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, 
                        imgs: Tensor, 
                        masks: Tensor, 
                        mask_ratio: float) -> Tuple[Tensor, List[Tensor] | Tensor, List[Tensor] | Tensor]:

        x       = self.patch_embed(imgs)
        mask    = self.mask_embed(masks)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # Remove all empty tokens and pad the sequences
        x, mask, lengths, idxs = remove_empty_patches_and_pad(x, mask)
  
        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        if mask_ratio < 1.0:
            x, ids_kept, ids_restore = random_masking(x, lengths, idxs, mask_ratio)
        else:
            ids_kept = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1)
            ids_restore = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1)
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, ids_kept, ids_restore

    def forward_decoder(self, x, ids_kept, ids_restore) -> Tensor:
        
        B, L, D = x.shape

        # embed tokens
        x = self.decoder_embed(x)

        cls_token = x[:, :1, :]
        x = x[:, 1:, :]

        # Append mask tokens to sequence
        x_with_mask_tokens = []
        for i in range(B):
            mask_tokens = self.mask_token.repeat(1, len(ids_restore[i]), 1)
            sample      = x[i, :len(ids_kept[i]), :] # Remove padding

            # Add position embeddings
            sample      = sample + self.decoder_pos_embed[:, ids_kept[i], :] # Add pos embed to the correct indices
            mask_tokens = mask_tokens + self.decoder_pos_embed[:, ids_restore[i], :] # Add pos embed to the correct indices

            sample = torch.cat([sample, mask_tokens], dim=1) # Add mask tokens to the end of the sequence
            
            sample = sample.squeeze() 
            x_with_mask_tokens.append(sample)

        x = pad_sequence(x_with_mask_tokens, batch_first=True)

        # Add cls token back
        x = torch.cat([cls_token, x], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        # print(x.shape)

        # remove cls token
        x = x[:, 1:, :]
        B, L, D = x.shape

        return x

    def forward_loss(self, imgs, pred, ids_kept, ids_restore) -> Tensor | float:
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*p*C]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        targets = self.patchify(imgs) # [N, L, p*p*C]

        loss = 0
        for i in range(len(imgs)):
            restored = pred[i][-len(ids_restore[i]):]
            target = targets[i][ids_restore[i]]

            loss += nn.functional.mse_loss(restored, target)  # Normalize by the number of patches

        loss = loss / len(imgs)

        return loss

    def forward(self, imgs: Tensor, masks: Tensor, mask_ratio: float = 1.0) -> Tuple[Tensor | float, Tensor, Tensor, List[Tensor] | Tensor, List[Tensor] | Tensor]:

        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        latent, ids_kept, ids_restore = self.forward_encoder(imgs, masks, mask_ratio)
        
        pred = self.forward_decoder(latent, ids_kept, ids_restore)  # [N, L, p*p*3]

        loss = self.forward_loss(imgs, pred, ids_kept, ids_restore)

        return loss, pred, latent, ids_kept, ids_restore


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks