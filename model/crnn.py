"""
CRNN (Convolutional Recurrent Neural Network) for handwriting OCR.
CNN feature extractor → BiLSTM → Linear output → CTC loss.
"""
import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES,
    CNN_OUTPUT_CHANNELS, RNN_HIDDEN_SIZE, RNN_NUM_LAYERS, RNN_DROPOUT
)


class CRNN(nn.Module):
    """
    CRNN architecture for CTC-based handwriting recognition.

    Input:  (batch, 1, H, W)  — grayscale image
    Output: (seq_len, batch, num_classes) — log probabilities for CTC
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 rnn_hidden: int = RNN_HIDDEN_SIZE,
                 rnn_layers: int = RNN_NUM_LAYERS,
                 rnn_dropout: float = RNN_DROPOUT):
        super().__init__()

        self.num_classes = num_classes

        # ── CNN Feature Extractor ──
        # Input:  (B, 1, 64, 256)
        # Output: (B, 512, 1, W') where W' = 256 / 4 = 64
        self.cnn = nn.Sequential(
            # Block 1: 1 → 64
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 64, 32, 128)

            # Block 2: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 128, 16, 64)

            # Block 3: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # (B, 256, 8, 64)

            # Block 4: 256 → 512
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # (B, 512, 4, 64)

            # Block 5: 512 → 512, collapse height
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),  # (B, 512, 1, W')
        )

        # ── BiLSTM Sequence Model ──
        self.rnn = nn.LSTM(
            input_size=CNN_OUTPUT_CHANNELS,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=rnn_dropout if rnn_layers > 1 else 0,
            batch_first=False,
        )

        # ── Output Projection ──
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)  # *2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Log-probabilities of shape (seq_len, B, num_classes)
        """
        # CNN: (B, 1, H, W) → (B, 512, 1, W')
        conv = self.cnn(x)

        # Reshape: (B, 512, 1, W') → (B, 512, W') → (W', B, 512)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)        # (B, 512, W')
        conv = conv.permute(2, 0, 1)  # (W', B, 512)

        # BiLSTM: (W', B, 512) → (W', B, 512)
        rnn_out, _ = self.rnn(conv)

        # Output: (W', B, 512) → (W', B, num_classes)
        output = self.fc(rnn_out)

        # Log softmax for CTC
        output = torch.nn.functional.log_softmax(output, dim=2)

        return output


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = CRNN()
    x = torch.randn(2, 1, IMG_HEIGHT, IMG_WIDTH)
    out = model(x)
    print(f"Input:      {x.shape}")
    print(f"Output:     {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")
