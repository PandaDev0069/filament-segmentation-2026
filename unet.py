import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ModifiedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super().__init__()

        # Encoder (Contracting Path)
        # init_features=32 helps save GPU memory on 2k resolution images
        self.inc = DoubleConv(in_channels, init_features)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(init_features, init_features * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(init_features * 2, init_features * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(init_features * 4, init_features * 8)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(init_features * 8, init_features * 16)
        )

        # Decoder (Expanding Path)
        self.up1 = nn.ConvTranspose2d(
            init_features * 16, init_features * 8, kernel_size=2, stride=2
        )
        self.conv_up1 = DoubleConv(init_features * 16, init_features * 8)

        self.up2 = nn.ConvTranspose2d(
            init_features * 8, init_features * 4, kernel_size=2, stride=2
        )
        self.conv_up2 = DoubleConv(init_features * 4 * 2, init_features * 4)

        self.up3 = nn.ConvTranspose2d(
            init_features * 4, init_features * 2, kernel_size=2, stride=2
        )
        self.conv_up3 = DoubleConv(init_features * 2 * 2, init_features * 2)

        self.up4 = nn.ConvTranspose2d(
            init_features * 2, init_features, kernel_size=2, stride=2
        )
        self.conv_up4 = DoubleConv(init_features * 2, init_features)

        self.outc = nn.Conv2d(init_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with dynamic size matching for uneven dimensions like 2028
        x = self.up1(x5)
        x = self._match_and_concat(x, x4)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = self._match_and_concat(x, x3)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = self._match_and_concat(x, x2)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = self._match_and_concat(x, x1)
        x = self.conv_up4(x)

        logits = self.outc(x)
        return logits

    def _match_and_concat(self, upsampled, bypass):
        """Pads the upsampled tensor if it doesn't align with the encoder skip layer due to rounding."""
        diffY = bypass.size()[2] - upsampled.size()[2]
        diffX = bypass.size()[3] - upsampled.size()[3]

        # Pad upsampled to match bypass tensor size exactly
        upsampled = F.pad(
            upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )

        return torch.cat([bypass, upsampled], dim=1)


# --- Verification ---
if __name__ == "__main__":
    # Simulate a single batch of a 2028x2048 grayscale image
    model = ModifiedUNet(in_channels=1, out_channels=1, init_features=32)
    dummy_input = torch.randn(1, 1, 2028, 2048)
    output = model(dummy_input)

    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
