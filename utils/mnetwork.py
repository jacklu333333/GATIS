import numpy as np
import torch
import torch.nn as nn


class channelShffuleNet(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        (N, C, H, W) = x.shape
        x = x.reshape(N, self.groups, C // self.groups, H, W)
        # shffule the groups
        order = torch.randperm(self.groups - 1)
        order = torch.cat((order, torch.tensor([self.groups - 1])), dim=0)
        x = x[:, order, :, :, :]
        return x.reshape(N, C, H, W)


class reshapeNet(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


class swapaxesNet(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.swapaxes(self.dim1, self.dim2)


class repeatNet(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.repeat(self.shape)


# Depthwise Separable Convolution
class depthwiseSeparableConvolution2d(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True
    ):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock2d, self).__init__()
        self.conv1 = nn.Sequential(
            depthwiseSeparableConvolution2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            depthwiseSeparableConvolution2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.activation = nn.PReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)
        return out


class mResnet2d(torch.nn.Module):
    def __init__(self, in_channel=4, H=100, W=51):
        super().__init__()
        self.in_channel = in_channel
        self.H = H
        self.W = W
        self.conv1 = nn.Sequential(
            depthwiseSeparableConvolution2d(
                self.in_channel, 64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        # output shape = 64 x 100 x 51

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # output shape = 64 x 100 x 51

        self.stage1 = self._make_stage(
            64, 128, 2, stride=2
        )  # output shape = 128 x 50 x 25
        self.stage2 = self._make_stage(
            128, 256, 3, stride=2
        )  # output shape = 256 x 25 x 13
        self.stage3 = self._make_stage(
            256, 512, 5, stride=2
        )  # output shape = 512 x 13 x 7
        self.stage4 = self._make_stage(
            512, 1024, 2, stride=2
        )  # output shape = 1024 x 7 x 4

        # self.stage5 = self._make_stage(
        #     512, 1024, 3, stride=2
        # )  # output shape = 1024 x 4 x 2

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Sequential(
            nn.Flatten(),
        )

    def _make_stage(self, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Dropout2d(0.1),
                depthwiseSeparableConvolution2d(
                    # nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(ResidualBlock2d(in_channels, out_channels, stride, downsample))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock2d(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.view(-1, self.in_channel, self.H, self.W)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # x = self.stage5(x)
        x = self.avg_pool(x)
        x = self.flatten(x)

        return x


class depthwiseSeparableConvolution1d(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True
    ):
        super().__init__()
        self.depthwise = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1d, self).__init__()
        self.conv1 = nn.Sequential(
            depthwiseSeparableConvolution1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.InstanceNorm1d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            depthwiseSeparableConvolution1d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.downsample = downsample
        self.activation = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)
        return out


class mResnet1d(torch.nn.Module):
    def __init__(self, in_channel=4, H=100):
        super().__init__()
        self.in_channel = in_channel
        self.H = H
        self.base_channel = 128
        self.conv1 = nn.Sequential(
            depthwiseSeparableConvolution1d(
                self.in_channel, self.base_channel, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm1d(self.base_channel),
            nn.PReLU(),
        )
        # output shape = 64 x 100

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        # output shape = 64 x 100

        self.stage1 = self._make_stage(
            self.base_channel, self.base_channel * 2, 2, stride=2
        )  # output shape = 128 x 50
        self.stage2 = self._make_stage(
            self.base_channel * 2, self.base_channel * 4, 3, stride=2
        )  # output shape = 256 x 25
        self.stage3 = self._make_stage(
            self.base_channel * 4, self.base_channel * 8, 5, stride=2
        )  # output shape = 512 x 13
        self.stage4 = self._make_stage(
            self.base_channel * 8, self.base_channel * 8, 2, stride=2
        )  # output shape = 1024 x 7
        # self.stage5 = self._make_stage(
        #     self.base_channel * 8, self.base_channel * 8, 3, stride=2
        # )  # output shape = 1024 x 4

        self.avg_pool = nn.AdaptiveAvgPool1d((1,))
        self.flatten = nn.Sequential(
            nn.Flatten(),
        )

    def _make_stage(self, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )
        layers = []
        layers.append(ResidualBlock1d(in_channels, out_channels, stride, downsample))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1d(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.view(-1, self.in_channel, self.H, self.W)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # x = self.stage5(x)
        x = self.avg_pool(x)
        x = self.flatten(x)

        return x


class TrigonometricActivation(nn.Module):
    def __init__(self):
        super(TrigonometricActivation, self).__init__()
        # self.filter = mThresholdActivation(threshold=0.01, value=0.0)

    def forward(self, x):
        # normalize according to the nrom

        out = torch.functional.F.tanh(x)
        norm = torch.linalg.norm(out, dim=1, keepdim=True)
        # replace zeros with 1
        norm = norm.clone()
        norm[norm == 0] = 1
        out = out / norm
        return out


class mThresholdActivation(nn.Module):
    def __init__(self, threshold=0.5, value=0.0):
        super(mThresholdActivation, self).__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, x):
        temp = torch.abs(x)
        out = torch.clone(x)
        out[temp < self.threshold] = self.value
        # find the first value over the threshold and set the before value to 0

        return out


class vectorAngleActivation(nn.Module):
    def __init__(self):
        super(vectorAngleActivation, self).__init__()
        pass

    def forward(self, x):
        strength = torch.functional.F.relu(x[:, 0])
        angle = torch.functional.F.tanh((x[:, 1])) * torch.pi

        return torch.stack((strength, angle), dim=1)


class angleActivation(nn.Module):
    def __init__(self):
        super(angleActivation, self).__init__()
        pass

    def forward(self, x):
        return torch.functional.F.tanh(x) * torch.pi
