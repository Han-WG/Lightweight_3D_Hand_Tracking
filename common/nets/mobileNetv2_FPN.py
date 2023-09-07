from torch import nn
import torch


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class Upsampling(nn.Sequential):
    def __int__(self, in_channel, out_channel, kernel_size=3):
        padding = (kernel_size - 1) // 2
        super(Upsampling, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, 1, padding, groups=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(out_channel, out_channel, kernel_size, 1, padding, groups=1),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, k, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, kernel_size=k, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


def read_setting(input_channel, inverted_residual_setting):
    blocks = []
    block = InvertedResidual
    t, k, c, n, s = inverted_residual_setting
    output_channel = _make_divisible(c * 1.0, 8)
    for i in range(n):
        stride = s if i == 0 else 1
        blocks.append(block(input_channel, output_channel, k, stride, expand_ratio=t))
        input_channel = output_channel
    return blocks, input_channel

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=63, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        blocks = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, k, c, n, s
            [1, 3, 16, 1, 1],
            [6, 3, 32, 2, 2],
            [6, 3, 48, 2, 2],
            [6, 3, 64, 3, 2],
            [6, 3, 96, 3, 1],
            [6, 3, 128, 2, 2],
        ]

        # inverted_residual_setting = [
        #     # t, c, n, s
        #     [1, 16, 1, 1],
        #     [6, 24, 2, 2],
        #     [6, 32, 3, 2],
        #     [6, 64, 4, 2],
        #     [6, 96, 3, 1],
        #     [6, 160, 3, 2],
        #     [6, 320, 1, 1],
        # ]
        input_feature = []
        out_feature = []
        # conv1 layer
        input_feature.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        t, k, c, n, s = inverted_residual_setting[0]
        output_channel = _make_divisible(c * alpha, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            input_feature.append(blocks(input_channel, output_channel, k, stride, expand_ratio=t))
            input_channel = output_channel
        # building last several layers
        self.input_feature = nn.Sequential(*input_feature)
        block2, input_channel = read_setting(input_channel, inverted_residual_setting[1])
        self.block2 = nn.Sequential(*block2)
        block3, input_channel = read_setting(input_channel, inverted_residual_setting[2])
        self.block3 = nn.Sequential(*block3)
        block41, input_channel = read_setting(input_channel, inverted_residual_setting[3])
        self.block41 = nn.Sequential(*block41)
        block4, input_channel = read_setting(input_channel, inverted_residual_setting[4])
        self.block4 = nn.Sequential(*block4)
        block5, input_channel = read_setting(input_channel, inverted_residual_setting[5])
        self.block5 = nn.Sequential(*block5)
        self.up_sample1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, groups=1),
                                        nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(64, 64, 3, 1, 1, groups=1))
        self.up_sample2 = nn.Sequential(nn.Conv2d(160, 64, 3, 1, 1, groups=1),
                                        nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(64, 48, 3, 1, 1, groups=1))
        self.up_sample3 = nn.Sequential(nn.Conv2d(96, 48, 3, 1, 1, groups=1),
                                        nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(48, 32, 3, 1, 1, groups=1))

        self.down_block1 = nn.Sequential(blocks(64, 96, 3, 2, 4))
        self.down_block2 = nn.Sequential(blocks(192, 128, 3, 2, 4))
        self.down_block3 = nn.Sequential(blocks(288, 128, 3, 2, 4))

        out_feature.append(ConvBNReLU(256, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*out_feature)
        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def read_setting(self, input_c, inverted_residual_setting):
        blocks = []
        block = InvertedResidual
        t, k, c, n, s = inverted_residual_setting
        output_channel = _make_divisible(c * 1.0, 8)
        for i in range(n):
            stride = s if i == 0 else 1
            blocks.append(block(input_c, output_channel, k, stride, expand_ratio=t))
            input_c = output_channel

    def forward(self, x):
        feature_x = self.input_feature(x)
        block2 = self.block2(feature_x)
        block3 = self.block3(block2)
        block41 = self.block41(block3)
        block4 = self.block4(block41)
        block5 = self.block5(block4)

        up_block1 = self.up_sample1(block5)
        up_concat = torch.concat((up_block1, block4), dim=1)
        up_block2 = self.up_sample2(up_concat)
        up_concat2 = torch.concat((up_block2, block3), dim=1)
        up_block3 = self.up_sample3(up_concat2)
        up_concat3 = torch.concat((up_block3, block2), dim=1)

        down_block1 = self.down_block1(up_concat3)
        down_concat = torch.concat((down_block1, up_concat2), dim=1)
        down_block2 = self.down_block2(down_concat)
        down_concat2 = torch.concat((down_block2, up_concat), dim=1)
        down_block3 = self.down_block3(down_concat2)
        down_concat3 = torch.concat((down_block3, block5), dim=1)
        feature = self.features(down_concat3)
        x = self.avgpool(feature)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    input_img = torch.randn(1, 3, 224, 224)
    MobileNetV2().forward(input_img)
