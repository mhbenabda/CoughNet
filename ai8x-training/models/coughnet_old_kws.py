###################################################################################################
# CoughNet network
# Habib Ben Abda
# Machine Learning on Microcontrollers
# 2025 - ETH Zurich
###################################################################################################
"""
CoughNet network description
copied from ai85net-kws.py
"""
from torch import nn

import ai8x


class CoughNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=1, dimensions=(128, 128),
                 fc_inputs=30, bias=False, **kwargs):
        super().__init__()

        print("channels: ", num_channels)
        print("dimensions: ", dimensions)
        # AI84 Limits
        assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 15, 3, # kernel size 3
                                          padding=1, bias=bias, **kwargs)
        # padding 1 -> no change in dimensions -> 15x28x28

        pad = 2 if dim == 28 else 1
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(15, 30, 3, pool_size=2, pool_stride=2,
                                                 padding=pad, bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 30x14x14
        if pad == 2:
            dim += 2  # padding 2 -> 30x16x16

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(30, 60, 3, pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 60x8x8

        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(60, 30, 3, pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 30x4x4

        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(30, 30, 3, pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 30x2x2

        self.conv6 = ai8x.FusedConv2dReLU(30, fc_inputs, 3, padding=1, bias=bias, **kwargs)

        self.fc = ai8x.SoftwareLinear(fc_inputs*dim*dim, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def coughnet(pretrained=False, **kwargs):
    """
    Constructs a CoughNet model.
    """
    assert not pretrained
    return CoughNet(**kwargs)


models = [
    {
        'name': 'coughnet',
        'min_input': 1,
        'dim': 2, # 2D input
    },
]
