import torch
import torch.nn as nn

# ----------------------------
# Basic Conv Block
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, padding=1),
            nn.BatchNorm2d(out_c),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)

# ----------------------------
# RGB Backbone
# ----------------------------
class BackboneRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 32, 3, 2),
            ConvBlock(32, 64, 3, 2),
            ConvBlock(64, 128, 3, 2)
        )

    def forward(self, x):
        return self.layers(x)

# ----------------------------
# Texture Backbone (Edge + LBP)
# ----------------------------
class BackboneTexture(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(2, 32, 3, 2),
            ConvBlock(32, 64, 3, 2),
            ConvBlock(64, 128, 3, 2)
        )

    def forward(self, edge, lbp):
        x = torch.cat([edge, lbp], dim=1)
        return self.layers(x)

# ----------------------------
# Fusion
# ----------------------------
class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock(256, 128, 1, 1)

    def forward(self, rgb_feat, tex_feat):
        x = torch.cat([rgb_feat, tex_feat], dim=1)
        return self.conv(x)

# ----------------------------
# Final Model
# ----------------------------
class DualStreamModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.rgb_backbone = BackboneRGB()
        self.tex_backbone = BackboneTexture()
        self.fusion = Fusion()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, rgb, edge, lbp):
        rgb_feat = self.rgb_backbone(rgb)
        tex_feat = self.tex_backbone(edge, lbp)

        fused = self.fusion(rgb_feat, tex_feat)

        x = self.pool(fused)
        return self.fc(x)