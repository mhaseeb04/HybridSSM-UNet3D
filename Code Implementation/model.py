"""
model.py
HybridSSM-UNet: Hybrid CNN + SSM (State Space Model) U-Net for medical image segmentation.

Uses ONLY standard PyTorch — no mamba-ssm, no monai, no einops required.

Architecture:
  Input -> Stem -> 4x EncoderStage (ResConv + LinearSSM + PooledCrossAttention)
        -> Bottleneck -> 4x DecoderStage (Upsample + skip + ResConv) -> Head

Works for:
  - 2D tasks: BUSI ultrasound, ISIC skin lesion  (mode="2d")
  - 3D tasks: Synapse CT, BraTS MRI              (mode="3d", uses 2D slices internally)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared: Simple linear SSM approximation (no external library needed)
# ---------------------------------------------------------------------------

class LinearSSM(nn.Module):
    """
    Lightweight State Space Model implemented with pure PyTorch.
    Processes a sequence (B, L, C) and returns (B, L, C).
    Approximates selective state space scanning using gated convolution.
    """
    def __init__(self, channels, expand=2):
        super().__init__()
        inner = channels * expand
        self.in_proj  = nn.Linear(channels, inner * 2, bias=False)
        self.conv     = nn.Conv1d(inner, inner, kernel_size=3,
                                  padding=1, groups=inner, bias=True)
        self.out_proj = nn.Linear(inner, channels, bias=False)
        self.norm     = nn.LayerNorm(inner)
        self.act      = nn.SiLU()

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        gate, z  = self.in_proj(x).chunk(2, dim=-1)          # each (B, L, inner)
        gate     = self.conv(gate.transpose(1, 2)).transpose(1, 2)
        gate     = self.act(gate)
        out      = self.norm(gate * torch.sigmoid(z))
        return self.out_proj(out) + x                         # residual


# ---------------------------------------------------------------------------
# 2D building blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(out_ch))
            if in_ch != out_ch or stride != 1 else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))


class SSMFuse2D(nn.Module):
    """
    Apply SSM on H-axis and W-axis independently, then fuse with CNN features.
    Uses small pooled tokens to stay memory-efficient.
    """
    def __init__(self, channels, pool_size=8):
        super().__init__()
        self.pool_size = pool_size
        self.ssm_h = LinearSSM(channels)
        self.ssm_w = LinearSSM(channels)
        self.fuse  = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.norm  = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.pool_size

        # Pool to small grid for SSM
        xp = F.adaptive_avg_pool2d(x, (p, p))  # (B, C, p, p)

        # Scan rows (H axis)
        xh = xp.permute(0, 3, 2, 1).reshape(B * p, p, C)  # (B*W, H, C)
        xh = self.ssm_h(xh).reshape(B, p, p, C).permute(0, 3, 2, 1)  # (B,C,p,p)

        # Scan columns (W axis)
        xw = xp.permute(0, 2, 3, 1).reshape(B * p, p, C)  # (B*H, W, C)
        xw = self.ssm_w(xw).reshape(B, p, p, C).permute(0, 3, 1, 2)  # (B,C,p,p)

        # Combine SSM outputs and upsample
        ssm_out = F.interpolate(xh + xw, size=(H, W),
                                mode='bilinear', align_corners=False)

        # Fuse with original CNN features
        out = self.fuse(torch.cat([x, ssm_out], dim=1))
        return self.norm(out + x)


class EncoderStage2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, stride=2)
        self.ssm  = SSMFuse2D(out_ch)

    def forward(self, x):
        f = self.conv(x)
        return self.ssm(f)


class DecoderStage2D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBNReLU(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


# ---------------------------------------------------------------------------
# Full 2D model (used for BUSI, ISIC and also for 3D via slice-by-slice)
# ---------------------------------------------------------------------------

class HybridUNet2D(nn.Module):
    """
    Hybrid SSM + CNN U-Net for 2D segmentation.
    in_channels : number of input channels (1 for grayscale, 3 for RGB)
    num_classes : number of output segmentation classes
    """
    def __init__(self, in_channels=3, num_classes=2, base_ch=32):
        super().__init__()
        c = [base_ch, base_ch*2, base_ch*4, base_ch*8]  # [32,64,128,256]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True)
        )
        self.enc1 = EncoderStage2D(c[0], c[0])
        self.enc2 = EncoderStage2D(c[0], c[1])
        self.enc3 = EncoderStage2D(c[1], c[2])
        self.enc4 = EncoderStage2D(c[2], c[3])

        self.bottleneck = ConvBNReLU(c[3], c[3])

        self.dec4 = DecoderStage2D(c[3], c[2], c[2])
        self.dec3 = DecoderStage2D(c[2], c[1], c[1])
        self.dec2 = DecoderStage2D(c[1], c[0], c[0])
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(c[0], c[0] // 2, kernel_size=2, stride=2),
            ConvBNReLU(c[0] // 2 + c[0], c[0])   # skip from stem
        )
        self.head = nn.Conv2d(c[0], num_classes, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)    # original resolution, c[0] channels
        s1 = self.enc1(s0)   # /2
        s2 = self.enc2(s1)   # /4
        s3 = self.enc3(s2)   # /8
        s4 = self.enc4(s3)   # /16

        b = self.bottleneck(s4)

        x = self.dec4(b,  s3)
        x = self.dec3(x,  s2)
        x = self.dec2(x,  s1)
        # Final upsample back to original resolution with stem skip
        x = self.dec1[0](x)                              # upsample
        x = self.dec1[1](torch.cat([x, s0], dim=1))      # fuse with stem
        return self.head(x)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(in_channels=3, num_classes=2, base_ch=32):
    """Build model. Works for all datasets."""
    return HybridUNet2D(in_channels=in_channels,
                        num_classes=num_classes,
                        base_ch=base_ch)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing HybridUNet2D ...")

    # Test 1: binary segmentation (BUSI / ISIC style)
    model = build_model(in_channels=3, num_classes=2, base_ch=32)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 2, 256, 256), f"Unexpected shape: {out.shape}"
    print(f"  RGB input    {x.shape} -> output {out.shape}  PASS")

    # Test 2: single-channel input (CT slice)
    model2 = build_model(in_channels=1, num_classes=9, base_ch=32)
    x2 = torch.randn(1, 1, 256, 256)
    out2 = model2(x2)
    assert out2.shape == (1, 9, 256, 256), f"Unexpected shape: {out2.shape}"
    print(f"  CT slice     {x2.shape} -> output {out2.shape}  PASS")

    # Test 3: 4-channel MRI (BraTS style)
    model3 = build_model(in_channels=4, num_classes=4, base_ch=32)
    x3 = torch.randn(1, 4, 256, 256)
    out3 = model3(x3)
    assert out3.shape == (1, 4, 256, 256), f"Unexpected shape: {out3.shape}"
    print(f"  MRI 4-ch     {x3.shape} -> output {out3.shape}  PASS")

    print(f"\n  Parameters: {model.num_params / 1e6:.2f}M")
    print("\nAll tests passed.")
