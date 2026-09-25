"""A small refinement U-Net that repairs one proposal mask at a time.

It reuses the ingredients that already proved themselves on this data in
exp085: 5x5 convolutions, instance normalisation and channel gates. The task is
different though. exp085 had to find filaments in a whole 2048 photograph and
failed at instance decomposition; this network is handed the instance and only
has to decide where its edge belongs, so it is deliberately small.
"""
from __future__ import annotations

import torch
from torch import nn

from experiments.exp_085_morden_spine.src.model import ChannelGate, ConvBlock


class RefinerNet(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 24, depth: int = 3):
        super().__init__()
        self.depth = depth
        widths = [base * 2 ** level for level in range(depth + 1)]
        self.down = nn.ModuleList()
        previous = in_channels
        for width in widths[:-1]:
            self.down.append(nn.ModuleList([ConvBlock(previous, width), ChannelGate(width)]))
            previous = width
        self.middle = ConvBlock(previous, widths[-1])
        self.up = nn.ModuleList()
        for level in range(depth - 1, -1, -1):
            self.up.append(nn.ModuleList([
                nn.ConvTranspose2d(widths[level + 1], widths[level], 2, stride=2),
                ConvBlock(widths[level] * 2, widths[level]),
            ]))
        self.head = nn.Conv2d(widths[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block, gate in self.down:
            x = gate(block(x))
            skips.append(x)
            x = self.pool(x)
        x = self.middle(x)
        for (upsample, block), skip in zip(self.up, reversed(skips)):
            x = upsample(x)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
