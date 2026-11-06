"""TGNet - Trust-Gated guided depth refinement network (~1.3 M params).

Two-branch encoder:
  depth branch : classical init depth (normalised), validity mask, log2(scale) plane
  guide branch : RGB + trust map + boundary band (from the classical stage)
The guide features are multiplied by a learned gate computed from the trust map, so RGB
structure is injected mostly where the classical stage found real depth boundaries
(prevents texture copying). A small U-Net decodes a depth residual and a per-pixel
fusion gate alpha:   D = D_classical + alpha * residual.
alpha is the adaptive "select / combine" mechanism between classical and learned output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.c2(self.act(self.c1(x)))


def down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.LeakyReLU(0.1, True), ResBlock(cout))


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin + cskip, cout, 3, padding=1), nn.LeakyReLU(0.1, True), ResBlock(cout))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class TGNet(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.d_in = nn.Sequential(nn.Conv2d(3, c, 3, padding=1), nn.LeakyReLU(0.1, True), ResBlock(c))
        self.g_in = nn.Sequential(nn.Conv2d(5, c, 3, padding=1), nn.LeakyReLU(0.1, True), ResBlock(c))
        # trust-conditioned gate on guidance features
        self.g_gate = nn.Sequential(nn.Conv2d(2 + c, c, 3, padding=1), nn.Sigmoid())
        self.fuse = nn.Sequential(nn.Conv2d(2 * c, c, 3, padding=1), nn.LeakyReLU(0.1, True), ResBlock(c))
        self.e1 = down(c, 2 * c)
        self.e2 = down(2 * c, 3 * c)
        self.e3 = down(3 * c, 4 * c)
        self.mid = nn.Sequential(ResBlock(4 * c), ResBlock(4 * c))
        self.u3 = Up(4 * c, 3 * c, 3 * c)
        self.u2 = Up(3 * c, 2 * c, 2 * c)
        self.u1 = Up(2 * c, c, c)
        self.head = nn.Sequential(ResBlock(c), nn.Conv2d(c, 2, 3, padding=1))
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, init, valid, scale_plane, rgb, trust, band, return_aux=False):
        fd = self.d_in(torch.cat([init, valid, scale_plane], 1))
        fg = self.g_in(torch.cat([rgb, trust, band], 1))
        gate = self.g_gate(torch.cat([trust, band, fg], 1))
        x0 = self.fuse(torch.cat([fd, fg * gate], 1))
        x1 = self.e1(x0)
        x2 = self.e2(x1)
        x3 = self.mid(self.e3(x2))
        y = self.u3(x3, x2)
        y = self.u2(y, x1)
        y = self.u1(y, x0)
        out = self.head(y)
        residual = 0.5 * torch.tanh(out[:, :1])     # bounded correction (normalised depth units)
        alpha = torch.sigmoid(out[:, 1:2].clamp(-20, 20) + 2.0)    # starts ~0.88 -> trusts the network early
        pred = init + alpha * residual
        if return_aux:
            return pred, {"residual": residual, "alpha": alpha, "guide_gate": gate.mean(1, keepdim=True)}
        return pred


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    net = TGNet()
    print("params", count_params(net))
    x = torch.rand(1, 1, 64, 64)
    y = net(x, x, x, torch.rand(1, 3, 64, 64), x, x)
    print(y.shape)
