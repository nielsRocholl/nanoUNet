"""Two-stream shared-encoder UNet with Difference Weighting at every skip
(LongiSeg). x = [FU(3ch); BL(3ch)]; null baseline = duplicated FU ⇒ identity DWB."""

from __future__ import annotations

import torch
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from torch import nn


class LongiResEncUNet(nn.Module):
    def __init__(self, base: nn.Module, n_stream_channels: int):
        super().__init__()
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.n = n_stream_channels
        ch = base.encoder.output_channels
        self.dwb = nn.ModuleList([nn.InstanceNorm3d(c, affine=False) for c in ch])

    def forward(self, x: torch.Tensor):
        a = x[:, : self.n]
        b = x[:, self.n :]
        sa = self.encoder(a)
        sb = self.encoder(b)
        fused = [t + t * self.dwb[i](t - s) for i, (t, s) in enumerate(zip(sa, sb))]
        return self.decoder(fused)

    @staticmethod
    def initialize(m: nn.Module) -> None:
        ResidualEncoderUNet.initialize(m)

    def compute_conv_feature_map_size(self, ps):
        return (
            2 * self.encoder.compute_conv_feature_map_size(ps)
            + self.decoder.compute_conv_feature_map_size(ps)
        )
