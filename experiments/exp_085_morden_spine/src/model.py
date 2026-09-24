"""Compact MORDEN-style mask and spine network for a 16 GB T4.

The design keeps the published MORDEN ingredients that its ablation found most
useful for solar filaments: 5x5 convolutions, instance normalization, direct
image branches at every scale, channel gates, and multiscale pyramid context.
The original H100-sized model is reduced to one head and fewer channels.  A
second output learns the supplied filament spines so instance identity does not
have to be inferred from semantic connected components alone.

Architecture adapted from https://github.com/irisaltHu/MORDEN (MIT license,
Copyright (c) 2026 Hu Jiarui).  See third_party/MORDEN_LICENSE.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel: int = 5):
        super().__init__()
        padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel, padding=padding, bias=False),
            nn.InstanceNorm2d(outputs, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv2d(outputs, outputs, kernel, padding=padding, bias=False),
            nn.InstanceNorm2d(outputs, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ImageBranch(nn.Module):
    """Inject the original local-contrast image at an encoder scale."""

    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inputs, outputs, 5, padding=2, bias=False),
            nn.InstanceNorm2d(outputs, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
        )

    def forward(self, image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return self.conv(F.adaptive_avg_pool2d(image, size))


class PyramidContext(nn.Module):
    """Low-cost context at four spatial scales."""

    def __init__(self, channels: int):
        super().__init__()
        branch = max(8, channels // 4)
        self.sizes = (1, 2, 4, 8)
        self.branches = nn.ModuleList(
            [nn.Conv2d(channels, branch, 1, bias=False) for _ in self.sizes]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels + branch * len(self.sizes), channels, 1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        context = [x]
        for output_size, branch in zip(self.sizes, self.branches):
            pooled = F.adaptive_avg_pool2d(x, output_size)
            context.append(F.interpolate(branch(pooled), size=size, mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(context, dim=1))


class MordenSpineNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super().__init__()
        channels = [base_channels * (2**i) for i in range(5)]
        self.stem = ConvBlock(in_channels, channels[0])
        self.pool = nn.MaxPool2d(2)
        self.image_branches = nn.ModuleList(
            [ImageBranch(in_channels, max(4, channels[i] // 4)) for i in range(1, 5)]
        )
        self.down = nn.ModuleList(
            [ConvBlock(channels[i - 1] + max(4, channels[i] // 4), channels[i]) for i in range(1, 5)]
        )
        self.down_gates = nn.ModuleList([ChannelGate(channels[i]) for i in range(1, 5)])
        self.context = PyramidContext(channels[-1])

        self.up = nn.ModuleList()
        self.decode = nn.ModuleList()
        for i in range(4, 0, -1):
            self.up.append(nn.ConvTranspose2d(channels[i], channels[i - 1], 2, stride=2))
            self.decode.append(ConvBlock(channels[i - 1] * 2, channels[i - 1]))
        self.output_gate = ChannelGate(channels[0])
        self.mask_head = nn.Conv2d(channels[0], 1, 1)
        self.spine_head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = [self.stem(image)]
        x = skip[0]
        for branch, block, gate in zip(self.image_branches, self.down, self.down_gates):
            x = self.pool(x)
            x = block(torch.cat([x, branch(image, x.shape[-2:])], dim=1))
            x = gate(x)
            skip.append(x)
        x = self.context(skip[-1])
        for index, (up, block) in enumerate(zip(self.up, self.decode), start=1):
            x = up(x)
            lateral = skip[-1 - index]
            if x.shape[-2:] != lateral.shape[-2:]:
                x = F.interpolate(x, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, lateral], dim=1))
        x = self.output_gate(x)
        return self.mask_head(x), self.spine_head(x)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
