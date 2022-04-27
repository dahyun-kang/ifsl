import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AttentiveSqueezeLayer(nn.Module):
    """
    Attentive squeeze layer consisting of a global self-attention layer followed by a feed-forward MLP
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, heads=8, groups=4, pool_kv=False):
        super(AttentiveSqueezeLayer, self).__init__()
        self.attn = Attention(in_channels, out_channels, kernel_size, stride, padding, bias, heads, groups, pool_kv)
        self.ff = FeedForward(out_channels, groups)

    def forward(self, input):
        x, support_mask = input
        batch, c, qh, qw, sh, sw = x.shape
        x = rearrange(x, 'b c d t h w -> b c (d t) h w')
        out = self.attn((x, support_mask))
        out = self.ff(out)
        out = rearrange(out, 'b c (d t) h w -> b c d t h w', d=qh, t=qw)
        return out, support_mask


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, heads=8, groups=4, pool_kv=False):
        super(Attention, self).__init__()
        self.heads = heads
        '''
        Size of conv output = floor((input  + 2 * pad - kernel) / stride) + 1
        The second condition of `retain_dim` checks the spatial size consistency by setting input=output=0;
        Use this term with caution to check the size consistency for generic cases!
        '''
        retain_dim = in_channels == out_channels and math.floor((2 * padding - kernel_size) / stride) == -1
        hidden_channels = out_channels // 2
        assert hidden_channels % self.heads == 0, "out_channels should be divided by heads. (example: out_channels: 40, heads: 4)"

        ksz_q = (1, kernel_size, kernel_size)
        str_q = (1, stride, stride)
        pad_q = (0, padding, padding)

        self.short_cut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=ksz_q, stride=str_q, padding=pad_q, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        ) if not retain_dim else nn.Identity()

        # Convolutional embeddings for (q, k, v)
        self.qhead = nn.Conv3d(in_channels, hidden_channels, kernel_size=ksz_q, stride=str_q, padding=pad_q, bias=bias)

        ksz = (1, kernel_size, kernel_size) if pool_kv else (1, 1, 1)
        str = (1, stride, stride) if pool_kv else (1, 1, 1)
        pad = (0, padding, padding) if pool_kv else (0, 0, 0)

        self.khead = nn.Conv3d(in_channels, hidden_channels, kernel_size=ksz, stride=str, padding=pad, bias=bias)
        self.vhead = nn.Conv3d(in_channels, hidden_channels, kernel_size=ksz, stride=str, padding=pad, bias=bias)

        self.agg = nn.Sequential(
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
        self.out_norm = nn.GroupNorm(groups, out_channels)

    def forward(self, input):
        x, support_mask = input

        x_ = self.short_cut(x)
        q_out = self.qhead(x)
        k_out = self.khead(x)
        v_out = self.vhead(x)

        q_h, q_w = q_out.shape[-2:]
        k_h, k_w = k_out.shape[-2:]

        q_out = rearrange(q_out, 'b (g c) t h w -> b g c t (h w)', g=self.heads)
        k_out = rearrange(k_out, 'b (g c) t h w -> b g c t (h w)', g=self.heads)
        v_out = rearrange(v_out, 'b (g c) t h w -> b g c t (h w)', g=self.heads)

        out = torch.einsum('b g c t l, b g c t m -> b g t l m', q_out, k_out)
        if support_mask is not None:
            out = self.attn_mask(out, support_mask, spatial_size=(k_h, k_w))
        out = F.softmax(out, dim=-1)
        out = torch.einsum('b g t l m, b g c t m -> b g c t l', out, v_out)
        out = rearrange(out, 'b g c t (h w) -> b (g c) t h w', h=q_h, w=q_w)
        out = self.agg(out)

        return self.out_norm(out + x_)

    def attn_mask(self, x, mask, spatial_size):
        assert mask is not None
        mask = F.interpolate(mask.float().unsqueeze(1), spatial_size, mode='bilinear', align_corners=True)
        mask = rearrange(mask, 'b 1 h w -> b 1 1 1 (h w)')
        out = x.masked_fill_(mask == 0, -1e9)
        return out


class FeedForward(nn.Module):
    def __init__(self, out_channels, groups=4, size=2):
        super(FeedForward, self).__init__()
        hidden_channels = out_channels // size
        self.ff = nn.Sequential(
            nn.Conv3d(out_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.out_norm = nn.GroupNorm(groups, out_channels)

    def forward(self, x):
        x_ = x
        out = self.ff(x)
        return self.out_norm(out + x_)
