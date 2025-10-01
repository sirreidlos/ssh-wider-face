from torch import nn
from itertools import product as product
import torchvision


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()


class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        mnet = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        self.features = mnet.features

    def forward(self, x):
        c3 = None
        c4 = None
        c5 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 6:  # stride 8, 32 channels
                c3 = x
            elif i == 13:  # stride 16, 96 channels
                c4 = x
            elif i == 18:  # stride 32, 1280 channels
                c5 = x
        return c3, c4, c5
