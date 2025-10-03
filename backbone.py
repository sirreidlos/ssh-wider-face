import logging
import torch
import torch.nn as nn
import torchvision
from itertools import product as product

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Backbone")


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()


class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("Backbone.MobileNetV2")

        # Load MobileNetV2 with pretrained weights
        mnet = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # Extract feature layers at different strides
        # features[:2] = stride 2
        # features[:4] = stride 4
        # features[:7] = stride 8
        # features[:14] = stride 16
        # features[:18] = stride 32

        self.features_stride2 = mnet.features[:2]  # stride 2
        self.features_stride4 = mnet.features[2:4]  # stride 4
        self.features_stride8 = mnet.features[4:7]  # stride 8
        self.features_stride16 = mnet.features[7:14]  # stride 16
        self.features_stride32 = mnet.features[14:]  # stride 32

        # Get output channels for each feature level
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            s2 = self.features_stride2(dummy_input)
            s4 = self.features_stride4(s2)
            s8 = self.features_stride8(s4)
            s16 = self.features_stride16(s8)
            s32 = self.features_stride32(s16)

            # Expected shapes at 224x224:
            # s4: [B, C, 56, 56] (stride 4)
            # s8: [B, C, 28, 28] (stride 8)
            # s16: [B, C, 14, 14] (stride 16)

        self.logger.debug("Initialized MobileNetV2 backbone with multi-stride features")
        self.logger.debug(f"stride 4 output: {s4.shape} (expected [B, C, 56, 56])")
        self.logger.debug(f"stride 8 output: {s8.shape} (expected [B, C, 28, 28])")
        self.logger.debug(f"stride 16 output: {s16.shape} (expected [B, C, 14, 14])")
        self.logger.debug(f"stride 32 output: {s32.shape} (expected [B, C, 7, 7])")

        # Channel adjustment layers to get consistent channel counts
        # We'll output 128, 96, and 256 channels respectively
        self.reduce_stride4 = nn.Sequential(
            nn.Conv2d(s4.shape[1], 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
        )

        self.reduce_stride8 = nn.Sequential(
            nn.Conv2d(s8.shape[1], 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU6(inplace=True),
        )

        self.reduce_stride16 = nn.Sequential(
            nn.Conv2d(s16.shape[1], 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        self.logger.debug(f"Input shape: {x.shape}")

        # Extract features at different strides
        s2 = self.features_stride2(x)
        self.logger.debug(f"After stride 2: {s2.shape}")

        s4 = self.features_stride4(s2)
        self.logger.debug(f"After stride 4: {s4.shape}")

        s8 = self.features_stride8(s4)
        self.logger.debug(f"After stride 8: {s8.shape}")

        s16 = self.features_stride16(s8)
        self.logger.debug(f"After stride 16: {s16.shape}")

        # Apply channel reduction
        f4 = self.reduce_stride4(s4)
        self.logger.debug(f"After reduce stride 4: {f4.shape}")

        f8 = self.reduce_stride8(s8)
        self.logger.debug(f"After reduce stride 8: {f8.shape}")

        f16 = self.reduce_stride16(s16)
        self.logger.debug(f"After reduce stride 16: {f16.shape}")

        # Return features at different scales:
        # - f4: 128 channels, stride 4 (160x160 at 640x640 input)
        # - f8: 96 channels, stride 8 (80x80 at 640x640 input)
        # - f16: 256 channels, stride 16 (40x40 at 640x640 input)
        return f4, f8, f16
