import torch
import torch.nn as nn
import torch.nn.functional as F


"""
We are using conv1d for calculating q, k, v vectors because
h = self.group_norm(h_in).view(B, C, H*W) -- this line reshaped the tensor 
from shape (B, C, H, W) → (B, C, N) where N = H x W.
Conv1D(ch, ch, 1) acts like a linear layer on each token.

nn.Conv2d(in_channels, out_channels, kernel_size)
Internally, the weight tensor has shape:
(out_channels, in_channels, kernel_height, kernel_width)

nn.Conv1d(in_channels=ch, out_channels=ch, kernel_size=1)
The weight tensor has shape:
(out_channels, in_channels, kernel_size)


"""


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.k = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.v = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.project_out_layer = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, x):
        """
        x: Input tensor of shape (B, C, H, W)
        returns: Tensor of the same shape with attention applied
        """

        B, C, H, W = x.shape

        # Preserving input x for residual connection
        h_in = x
        h = self.group_norm(h_in).view(B, C, H * W)  # shape: (B, C, H*W)

        # Compute queries, keys, and values
        q = self.q(h)  # (B, C, H*W)
        k = self.k(h)  # (B, C, H*W)
        v = self.v(h)  # (B, C, H*W)

        # Transpose q to (B, H*W, C), keep k as (B, C, H*W)
        attention_scores = torch.bmm(q.transpose(1, 2), k) * (C**-0.5)  # (B, H*W, H*W)=(B, N, N)

        # Applying softmax to attentions scores
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Attention output after multiplying with value vectos--> same as contextual embeddings in attention is all you need
        attention_output = torch.bmm(attention_weights, v.transpose(1, 2)).transpose(1, 2)  # (B, H*W, C) again into (B,C,H*W)

        # Applying out projection
        out = self.project_out_layer(attention_output).view(B, C, H, W)

        # Add residual
        return h_in + out
