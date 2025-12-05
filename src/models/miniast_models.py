# -*- coding: utf-8 -*-
# @Time    : 2024
# @Author  : Adapted from Yuan Gong's AST and MiniViT
# @File    : miniast_models.py

"""
MiniAST: Audio Spectrogram Transformer with Weight Multiplexing (No Distillation)

This module adapts the MiniViT compression framework to AST, implementing:
1. Weight Sharing: Share weights across consecutive transformer blocks
2. Weight Transformation: Linear transformations for MSA and depth-wise conv for MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import os
import wget
os.environ['TORCH_HOME'] = '../../pretrained_models'
import timm
from timm.models.layers import to_2tuple, trunc_normal_


class PatchEmbed(nn.Module):
    """Patch embedding layer - same as original AST."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class AttentionTransform(nn.Module):
    """
    Linear transformation for attention weights (before and after softmax).
    Implements Eq. (6-7) from MiniViT paper.
    """
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        # Transformation matrices F^(1) and F^(2)
        self.transform_before = nn.Parameter(torch.eye(num_heads))
        self.transform_after = nn.Parameter(torch.eye(num_heads))
    
    def forward(self, attn_weights):
        """
        Args:
            attn_weights: (B, num_heads, N, N) attention weights before softmax
        Returns:
            transformed attention weights after softmax
        """
        B, H, N, _ = attn_weights.shape
        
        # Transform before softmax
        attn_reshaped = attn_weights.permute(0, 2, 3, 1)  # (B, N, N, H)
        attn_transformed = torch.matmul(attn_reshaped, self.transform_before.t())
        attn_transformed = attn_transformed.permute(0, 3, 1, 2)  # (B, H, N, N)
        
        # Softmax
        attn_probs = F.softmax(attn_transformed, dim=-1)
        
        # Transform after softmax
        attn_reshaped = attn_probs.permute(0, 2, 3, 1)
        attn_transformed = torch.matmul(attn_reshaped, self.transform_after.t())
        attn_transformed = attn_transformed.permute(0, 3, 1, 2)
        
        return attn_transformed


class MLPTransform(nn.Module):
    """
    Depth-wise convolution transformation for MLP.
    Implements Eq. (8) from MiniViT paper.
    """
    def __init__(self, embed_dim, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv1d(
            embed_dim, embed_dim, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            groups=embed_dim
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, D) input features
        Returns:
            transformed features (B, N, D)
        """
        x_t = x.transpose(1, 2)  # (B, D, N)
        x_t = self.dwconv(x_t)
        x_t = x_t.transpose(1, 2)  # (B, N, D)
        x_t = self.norm(x_t)
        return x_t


class WeightSharedBlock(nn.Module):
    """
    Wrapper that applies weight transformation to a shared transformer block.
    """
    def __init__(self, shared_block, num_heads, embed_dim, 
                 use_attn_transform=True, use_mlp_transform=True, 
                 mlp_kernel_size=7):
        super().__init__()
        self.shared_block = shared_block
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.use_attn_transform = use_attn_transform
        self.use_mlp_transform = use_mlp_transform
        
        if use_attn_transform:
            self.attn_transform = AttentionTransform(num_heads)
        
        if use_mlp_transform:
            self.mlp_transform = MLPTransform(embed_dim, kernel_size=mlp_kernel_size)
        
        # Layer-specific normalization (NOT shared)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        attn = self.shared_block.attn
        mlp = self.shared_block.mlp
        
        # === Attention with transformation ===
        residual = x
        x = self.norm1(x)
        
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = attn.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention weights
        attn_weights = (q @ k.transpose(-2, -1)) * attn.scale
        
        # Apply transformation
        if self.use_attn_transform:
            attn_probs = self.attn_transform(attn_weights)
        else:
            attn_probs = F.softmax(attn_weights, dim=-1)
        
        attn_probs = attn.attn_drop(attn_probs)
        
        x = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        x = attn.proj(x)
        x = attn.proj_drop(x)
        x = residual + x
        
        # === MLP with transformation ===
        residual = x
        x = self.norm2(x)
        
        if self.use_mlp_transform:
            x = self.mlp_transform(x)
        
        x = mlp(x)
        x = residual + x
        
        return x


class MiniASTModel(nn.Module):
    """
    MiniAST: Audio Spectrogram Transformer with Weight Multiplexing
    
    Args:
        label_dim: Number of output classes
        fstride: Frequency stride for patch splitting
        tstride: Time stride for patch splitting
        input_fdim: Number of frequency bins (default 128)
        input_tdim: Number of time frames
        imagenet_pretrain: Use ImageNet pretrained weights
        audioset_pretrain: Use AudioSet pretrained weights
        model_size: Model size ('tiny224', 'small224', 'base224', 'base384')
        num_shared_layers: Number of layers to share weights
        use_attn_transform: Enable attention transformation
        use_mlp_transform: Enable MLP transformation
        mlp_kernel_size: Kernel size for MLP depth-wise convolution
        verbose: Print model summary
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, 
                 input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, 
                 model_size='base384', num_shared_layers=2, 
                 use_attn_transform=True, use_mlp_transform=True,
                 mlp_kernel_size=7, verbose=True):
        
        super(MiniASTModel, self).__init__()
        
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5'
        
        self.num_shared_layers = num_shared_layers
        self.use_attn_transform = use_attn_transform
        self.use_mlp_transform = use_mlp_transform
        
        if verbose:
            print('---------------MiniAST Model Summary---------------')
            print(f'ImageNet pretraining: {imagenet_pretrain}, AudioSet pretraining: {audioset_pretrain}')
            print(f'Weight sharing: every {num_shared_layers} layers')
            print(f'Attention transform: {use_attn_transform}, MLP transform: {use_mlp_transform}')
        
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
                self.num_heads = 3
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
                self.num_heads = 6
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
                self.num_heads = 12
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
                self.num_heads = 12
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim), 
                nn.Linear(self.original_embedding_dim, label_dim)
            )
            
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            
            if verbose:
                print(f'Frequency stride={fstride}, time stride={tstride}')
                print(f'Number of patches={num_patches}')
            
            # Modify patch embedding for single channel
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, 
                                       kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain:
                new_proj.weight = torch.nn.Parameter(
                    torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1)
                )
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj
            
            # Adapt positional embedding
            if imagenet_pretrain:
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(
                    1, self.original_num_patches, self.original_embedding_dim
                ).transpose(1, 2).reshape(
                    1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw
                )
                
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, 
                        int(self.oringal_hw / 2) - int(t_dim / 2): 
                        int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = F.interpolate(new_pos_embed, 
                                                  size=(self.oringal_hw, t_dim), 
                                                  mode='bilinear')
                
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, 
                        int(self.oringal_hw / 2) - int(f_dim / 2): 
                        int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = F.interpolate(new_pos_embed, 
                                                  size=(f_dim, t_dim), 
                                                  mode='bilinear')
                
                new_pos_embed = new_pos_embed.reshape(
                    1, self.original_embedding_dim, num_patches
                ).transpose(1, 2)
                
                self.v.pos_embed = nn.Parameter(torch.cat(
                    [self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1
                ))
            else:
                new_pos_embed = nn.Parameter(
                    torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim)
                )
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)
            
            self._apply_weight_multiplexing(mlp_kernel_size, verbose)
        
        elif audioset_pretrain:
            if imagenet_pretrain == False:
                raise ValueError('AudioSet pretrained model requires imagenet_pretrain=True')
            if model_size != 'base384':
                raise ValueError('Only base384 AudioSet pretrained model available')
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.num_heads = 12
            
            if not os.path.exists('../../pretrained_models/audioset_10_10_0.4593.pth'):
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='../../pretrained_models/audioset_10_10_0.4593.pth')
            
            sd = torch.load('../../pretrained_models/audioset_10_10_0.4593.pth', map_location=device)
            
            from ast_models import ASTModel
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, 
                                   input_fdim=128, input_tdim=1024, 
                                   imagenet_pretrain=False, audioset_pretrain=False, 
                                   model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim), 
                nn.Linear(self.original_embedding_dim, label_dim)
            )
            
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            
            if verbose:
                print(f'Frequency stride={fstride}, time stride={tstride}')
                print(f'Number of patches={num_patches}')
            
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(
                1, 1212, 768
            ).transpose(1, 2).reshape(1, 768, 12, 101)
            
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 
                    50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            else:
                new_pos_embed = F.interpolate(new_pos_embed, 
                                              size=(12, t_dim), mode='bilinear')
            
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 
                    6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            elif f_dim > 12:
                new_pos_embed = F.interpolate(new_pos_embed, 
                                              size=(f_dim, t_dim), mode='bilinear')
            
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat(
                [self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1
            ))
            
            self._apply_weight_multiplexing(mlp_kernel_size, verbose)
    
    def _apply_weight_multiplexing(self, mlp_kernel_size, verbose):
        """Apply weight multiplexing to transformer blocks."""
        num_blocks = len(self.v.blocks)
        num_shared_groups = num_blocks // self.num_shared_layers
        
        if verbose:
            print(f'Total blocks: {num_blocks}, Shared groups: {num_shared_groups}')
        
        self.shared_blocks = nn.ModuleList()
        self.weight_shared_layers = nn.ModuleList()
        
        for group_idx in range(num_shared_groups):
            shared_block_idx = group_idx * self.num_shared_layers
            shared_block = self.v.blocks[shared_block_idx]
            self.shared_blocks.append(shared_block)
            
            group_layers = nn.ModuleList()
            for layer_idx in range(self.num_shared_layers):
                transformed_layer = WeightSharedBlock(
                    shared_block=shared_block,
                    num_heads=self.num_heads,
                    embed_dim=self.original_embedding_dim,
                    use_attn_transform=self.use_attn_transform,
                    use_mlp_transform=self.use_mlp_transform,
                    mlp_kernel_size=mlp_kernel_size
                )
                group_layers.append(transformed_layer)
            
            self.weight_shared_layers.append(group_layers)
        
        # Handle remaining blocks
        remaining_start = num_shared_groups * self.num_shared_layers
        self.remaining_blocks = nn.ModuleList()
        for i in range(remaining_start, num_blocks):
            self.remaining_blocks.append(self.v.blocks[i])
        
        if verbose:
            self._print_param_count()
    
    def _print_param_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params / 1e6:.2f}M')
        print(f'Trainable parameters: {trainable_params / 1e6:.2f}M')
    
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, 
                              kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    @autocast()
    def forward(self, x):
        """
        Args:
            x: Input spectrogram (batch_size, time_frame_num, frequency_bins)
        Returns:
            prediction: Class logits
        """
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        
        B = x.shape[0]
        x = self.v.patch_embed(x)
        
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        # Process through weight-shared layers
        for group_layers in self.weight_shared_layers:
            for layer in group_layers:
                x = layer(x)
        
        # Process remaining blocks
        for block in self.remaining_blocks:
            x = block(x)
        
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        x = self.mlp_head(x)
        
        return x


if __name__ == '__main__':
    print("Testing MiniAST Model...")
    
    input_tdim = 100
    miniast = MiniASTModel(
        input_tdim=input_tdim,
        num_shared_layers=2,
        use_attn_transform=True,
        use_mlp_transform=True,
        imagenet_pretrain=False
    )
    
    test_input = torch.rand([2, input_tdim, 128])
    test_output = miniast(test_input)
    print(f'Output shape: {test_output.shape}')
    print("MiniAST Model test passed!")