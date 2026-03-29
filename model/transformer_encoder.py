"""
Transformer-based sequence encoder for CTC recognition.

Replaces the BiLSTM in the CRNN pipeline.  The multi-head self-attention
mechanism handles long-range dependencies in multi-word prescription lines
better than recurrent models.

Architecture:
    PositionalEncoding → N × TransformerEncoderLayer → output
"""
import math
import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    TRANSFORMER_HEADS, TRANSFORMER_LAYERS,
    TRANSFORMER_DIM, TRANSFORMER_FF_DIM, TRANSFORMER_DROPOUT
)


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Input:  (seq_len, batch, d_model)
    Output: (seq_len, batch, d_model)
    """

    def __init__(self, d_model: int, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerSequenceEncoder(nn.Module):
    """
    Transformer encoder for sequence modelling in CRNN.

    Takes CNN feature columns (seq_len, batch, d_input) and outputs
    contextualised features (seq_len, batch, d_model).

    If d_input != d_model, a linear projection is applied first.
    """

    def __init__(self, d_input: int,
                 d_model: int = TRANSFORMER_DIM,
                 n_heads: int = TRANSFORMER_HEADS,
                 n_layers: int = TRANSFORMER_LAYERS,
                 d_ff: int = TRANSFORMER_FF_DIM,
                 dropout: float = TRANSFORMER_DROPOUT):
        super().__init__()

        self.d_model = d_model

        # Project CNN output to transformer dimension if needed
        self.input_proj = (
            nn.Linear(d_input, d_model) if d_input != d_model else nn.Identity()
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # (seq, batch, features) format
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch, d_input)

        Returns:
            (seq_len, batch, d_model)
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        return x


if __name__ == "__main__":
    # Quick shape test
    enc = TransformerSequenceEncoder(d_input=512)
    x = torch.randn(64, 2, 512)  # seq_len=64, batch=2, features=512
    out = enc(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    n_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"Transformer params: {n_params:,}")
