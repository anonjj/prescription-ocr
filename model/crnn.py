"""
CRNN (Convolutional Recurrent Neural Network) for handwriting OCR.

Supports multiple configurations:
  - CNN backbone: VGG-style (original) or EfficientNet-B0 (pretrained)
  - Sequence model: BiLSTM (original) or Transformer encoder
  - Optional STN (Spatial Transformer Network) for geometric rectification

Pipeline: [STN] → CNN → Sequence Model → Linear → CTC loss
"""
import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES,
    CNN_OUTPUT_CHANNELS, RNN_HIDDEN_SIZE, RNN_NUM_LAYERS, RNN_DROPOUT,
    CNN_BACKBONE, SEQ_MODEL, USE_STN,
    TRANSFORMER_DIM,
)


# ── VGG-style CNN Backbone (original) ──────────────────────────────────────────

class VGGBackbone(nn.Module):
    """
    Original 5-block VGG-style CNN feature extractor.
    Input:  (B, 1, 64, 256)
    Output: (B, 512, 1, W') where W' ≈ 64
    """
    def __init__(self):
        super().__init__()
        self.output_channels = 512

        self.features = nn.Sequential(
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

    def forward(self, x):
        return self.features(x)


# ── EfficientNet-B0 Backbone ──────────────────────────────────────────────────

class EfficientNetBackbone(nn.Module):
    """
    Pretrained EfficientNet-B0 adapted for single-channel OCR input.

    - Replaces the first conv to accept 1-channel grayscale input
    - Removes the classifier head
    - Adds adaptive pooling to collapse height to 1 for RNN input

    Input:  (B, 1, 64, 256)
    Output: (B, 1280, 1, W')
    """
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            # Fallback for older torchvision
            try:
                from efficientnet_pytorch import EfficientNet
                effnet = EfficientNet.from_pretrained("efficientnet-b0")
                # efficientnet_pytorch has a different API
                self._is_pytorch_impl = True
            except ImportError:
                raise ImportError(
                    "EfficientNet backbone requires torchvision>=0.13 or "
                    "efficientnet_pytorch. Install: pip install efficientnet_pytorch"
                )

        self.output_channels = 1280  # EfficientNet-B0 final feature channels

        if hasattr(effnet, 'features'):
            # torchvision implementation
            features = list(effnet.features.children())

            # Replace first conv: 3-channel → 1-channel
            first_conv = features[0]  # ConvBnActivation block
            first_layer = first_conv[0]  # actual Conv2d
            new_conv = nn.Conv2d(
                1, first_layer.out_channels,
                kernel_size=first_layer.kernel_size,
                stride=first_layer.stride,
                padding=first_layer.padding,
                bias=False,
            )
            # Initialize new conv with mean of pretrained RGB weights
            with torch.no_grad():
                new_conv.weight.copy_(first_layer.weight.mean(dim=1, keepdim=True))

            features[0][0] = new_conv
            self.features = nn.Sequential(*features)
        else:
            # efficientnet_pytorch implementation
            old_conv = effnet._conv_stem
            new_conv = nn.Conv2d(
                1, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            effnet._conv_stem = new_conv
            self.features = effnet

        # Adaptive pool to collapse height
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        if hasattr(self, '_is_pytorch_impl') and self._is_pytorch_impl:
            # efficientnet_pytorch forward
            x = self.features.extract_features(x)
        else:
            x = self.features(x)
        x = self.adaptive_pool(x)  # (B, 1280, 1, W')
        return x


# ── Main CRNN Model ───────────────────────────────────────────────────────────

class CRNN(nn.Module):
    """
    CRNN architecture for CTC-based handwriting recognition.

    Input:  (batch, 1, H, W)  — grayscale image
    Output: (seq_len, batch, num_classes) — log probabilities for CTC

    Args:
        num_classes: Number of output classes (including CTC blank)
        rnn_hidden: Hidden size for BiLSTM
        rnn_layers: Number of BiLSTM layers
        rnn_dropout: Dropout between BiLSTM layers
        backbone: CNN backbone — "vgg" or "efficientnet"
        seq_model: Sequence model — "bilstm" or "transformer"
        use_stn: Whether to prepend a Spatial Transformer Network
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 rnn_hidden: int = RNN_HIDDEN_SIZE,
                 rnn_layers: int = RNN_NUM_LAYERS,
                 rnn_dropout: float = RNN_DROPOUT,
                 backbone: str = CNN_BACKBONE,
                 seq_model: str = SEQ_MODEL,
                 use_stn: bool = USE_STN):
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone
        self.seq_model_name = seq_model

        # ── Optional STN ──
        if use_stn:
            from model.stn import STN
            self.stn = STN(in_channels=1)
        else:
            self.stn = None

        # ── CNN Backbone ──
        if backbone == "efficientnet":
            self.cnn = EfficientNetBackbone()
        else:  # "vgg"
            self.cnn = VGGBackbone()

        cnn_out_channels = self.cnn.output_channels

        # ── Sequence Model ──
        if seq_model == "transformer":
            from model.transformer_encoder import TransformerSequenceEncoder
            self.seq = TransformerSequenceEncoder(d_input=cnn_out_channels)
            fc_input_size = TRANSFORMER_DIM
        else:  # "bilstm"
            self.seq = nn.LSTM(
                input_size=cnn_out_channels,
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                bidirectional=True,
                dropout=rnn_dropout if rnn_layers > 1 else 0,
                batch_first=False,
            )
            fc_input_size = rnn_hidden * 2  # *2 for bidirectional

        # ── Output Projection ──
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Log-probabilities of shape (seq_len, B, num_classes)
        """
        # Optional STN: (B, 1, H, W) → (B, 1, H, W)
        if self.stn is not None:
            x = self.stn(x)

        # CNN: (B, 1, H, W) → (B, C, 1, W')
        conv = self.cnn(x)

        # Reshape: (B, C, 1, W') → (B, C, W') → (W', B, C)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)        # (B, C, W')
        conv = conv.permute(2, 0, 1)  # (W', B, C)

        # Sequence model: (W', B, C) → (W', B, hidden)
        if self.seq_model_name == "transformer":
            seq_out = self.seq(conv)
        else:
            seq_out, _ = self.seq(conv)

        # Output: (W', B, hidden) → (W', B, num_classes)
        output = self.fc(seq_out)

        # Log softmax for CTC
        output = torch.nn.functional.log_softmax(output, dim=2)

        return output


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test — all combinations
    combos = [
        ("vgg", "bilstm", False),
        ("vgg", "bilstm", True),
        ("vgg", "transformer", False),
    ]

    # Only test EfficientNet if available
    try:
        from torchvision.models import efficientnet_b0
        combos.append(("efficientnet", "bilstm", False))
        combos.append(("efficientnet", "transformer", False))
    except ImportError:
        print("  (skipping EfficientNet — not installed)")

    for backbone, seq_model, use_stn in combos:
        model = CRNN(backbone=backbone, seq_model=seq_model, use_stn=use_stn)
        x = torch.randn(2, 1, IMG_HEIGHT, IMG_WIDTH)
        out = model(x)
        stn_tag = "+STN" if use_stn else ""
        print(f"  {backbone:>13} + {seq_model:<11} {stn_tag:>4}  →  "
              f"output {out.shape}  |  {count_parameters(model):>10,} params")
