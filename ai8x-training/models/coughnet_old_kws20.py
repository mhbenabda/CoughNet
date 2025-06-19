###################################################################################################
# CoughNet network
# Habib Ben Abda
# Machine Learning on Microcontrollers
# 2025 - ETH Zurich
###################################################################################################
"""
CoughNet network description

copied from ai85net-kws20.py
"""
from torch import nn

import ai8x


class CoughNet(nn.Module):
    """
    Compound KWS20 Audio net, starting with Conv1Ds with kernel_size=1
    and then switching to Conv2Ds
    """

    # num_classes = n keywords + 1 unknown
    def __init__(
            self,
            num_classes=1,
            num_channels=128,
            dimensions=(128, 1),  # pylint: disable=unused-argument
            fc_inputs=7,
            bias=False,
            **kwargs
    ):
        super().__init__()

        self.voice_conv1 = ai8x.FusedConv1dReLU(num_channels, 100, 1, stride=1, padding=0,
                                                bias=bias, **kwargs)

        self.voice_conv2 = ai8x.FusedConv1dReLU(100, 100, 1, stride=1, padding=0,
                                                bias=bias, **kwargs)

        self.voice_conv3 = ai8x.FusedConv1dReLU(100, 50, 1, stride=1, padding=0,
                                                bias=bias, **kwargs)

        self.voice_conv4 = ai8x.FusedConv1dReLU(50, 16, 1, stride=1, padding=0,
                                                bias=bias, **kwargs)

        self.kws_conv1 = ai8x.FusedConv2dReLU(16, 32, 3, stride=1, padding=1,
                                              bias=bias, **kwargs)

        self.kws_conv2 = ai8x.FusedConv2dReLU(32, 64, 3, stride=1, padding=1,
                                              bias=bias, **kwargs)

        self.kws_conv3 = ai8x.FusedConv2dReLU(64, 64, 3, stride=1, padding=1,
                                              bias=bias, **kwargs)

        self.kws_conv4 = ai8x.FusedConv2dReLU(64, 30, 3, stride=1, padding=1,
                                              bias=bias, **kwargs)

        self.kws_conv5 = ai8x.FusedConv2dReLU(30, fc_inputs, 3, stride=1, padding=1,
                                              bias=bias, **kwargs)

        self.fc = ai8x.Linear(fc_inputs * 128, num_classes, bias=bias, wide=True, **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Run CNN
        x = self.voice_conv1(x)
        x = self.voice_conv2(x)
        x = self.voice_conv3(x)
        x = self.voice_conv4(x)
        x = x.view(x.shape[0], x.shape[1], 16, -1)
        x = self.kws_conv1(x)
        x = self.kws_conv2(x)
        x = self.kws_conv3(x)
        x = self.kws_conv4(x)
        x = self.kws_conv5(x)
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
        'dim': 1, # i.e expects 1D input
    },
]
