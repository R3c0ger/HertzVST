import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TemporalSelfAttention(nn.Module):
    """
    Temporal Self-Attention Layer, performing self-attention over the temporal dimension.
    Inspired by the approach in AnimateDiff.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim or (channels // num_heads)
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.to_qkv = nn.Linear(channels, channels * 3, bias=False)
        self.to_out = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, num_frames):
        """
        Args:
            x: (batch*frames, channels, h, w) or (batch, channels, frames, h, w)
            num_frames: number of frames
        
        Returns:
            (batch*frames, channels, h, w) or (batch, channels, frames, h, w)
        """
        # Handle different input formats
        if x.ndim == 4:
            # (batch*frames, channels, h, w)
            b_f, c, h, w = x.shape
            batch = b_f // num_frames
            
            # Rearrange to (batch, frames, height*width, channels)
            x = rearrange(x, '(b f) c h w -> b f (h w) c', f=num_frames, b=batch)
            input_format = '4d'
        elif x.ndim == 5:
            # (batch, channels, frames, h, w)
            b, c, f, h, w = x.shape
            x = rearrange(x, 'b c f h w -> b f (h w) c', f=f)
            input_format = '5d'
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # Apply self-attention
        qkv = self.to_qkv(x)  # (batch, frames, h*w, channels*3)
        qkv = rearrange(qkv, 'b f n (three h d) -> three b h f n d', three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores (over the temporal dimension)
        attn = torch.einsum('b h f n d, b h g n d -> b h f g', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.einsum('b h f g, b h g n d -> b h f n d', attn, v)
        out = rearrange(out, 'b h f n d -> b f n (h d)')
        out = self.to_out(out)
        
        # Residual connection
        out = out + x
        
        # Restore original format
        if input_format == '4d':
            out = rearrange(out, 'b f (h w) c -> (b f) c h w', h=h, w=w)
        else:  # 5d
            out = rearrange(out, 'b f (h w) c -> b c f h w', h=h, w=w)
        
        return out


class TemporalAttentionEnhancement:
    """
    Temporal Attention Enhancement Wrapper to apply temporal attention 
    during the denoising process.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: int = None,
        dropout: float = 0.0,
        apply_layers: list = None,
    ):
        """
        Args:
            channels: number of feature channels
            num_heads: number of attention heads
            head_dim: dimension of each head
            dropout: dropout rate
            apply_layers: list of layers to apply temporal attention, 
                e.g., ['down_blocks.0', 'mid_block', 'up_blocks.2']
        """
        self.temporal_attn = TemporalSelfAttention(
            channels=channels,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.apply_layers = apply_layers or []
        
    def apply_to_latents(self, latents, num_frames):
        """
        Apply temporal attention to latents
        
        Args:
            latents: (batch, channels, frames, h, w)
            num_frames: number of frames
        
        Returns:
            (batch, channels, frames, h, w)
        """
        # Apply temporal attention
        enhanced_latents = self.temporal_attn(latents, num_frames)
        return enhanced_latents


def create_temporal_attention_enhancement(
    channels: int = 320,
    num_heads: int = 8,
    head_dim: int = None,
    dropout: float = 0.0,
):
    """
    Factory function to create a Temporal Attention Enhancement module.
    
    Args:
        channels: number of feature channels (default 320, corresponding to the middle layer of UNet)
        num_heads: number of attention heads
        head_dim: dimension of each head
        dropout: dropout rate
    
    Returns:
        TemporalAttentionEnhancement instance
    """
    return TemporalAttentionEnhancement(
        channels=channels,
        num_heads=num_heads,
        head_dim=head_dim,
        dropout=dropout,
    )

