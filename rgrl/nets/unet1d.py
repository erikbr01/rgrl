"""ConditionalUnet1D from Diffusion Policy (Chi et al. 2023), self-contained.

Matches the reference architecture (FiLM-conditioned residual 1D conv blocks)
so hyperparameters and results transfer from existing repos.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResBlock1d(nn.Module):
    """Two conv blocks with FiLM (scale, bias) conditioning after the first."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.block1 = Conv1dBlock(in_ch, out_ch, kernel_size, n_groups)
        self.block2 = Conv1dBlock(out_ch, out_ch, kernel_size, n_groups)
        self.cond = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_ch * 2))
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        scale, bias = self.cond(cond).unsqueeze(-1).chunk(2, dim=1)
        out = out * scale + bias
        out = self.block2(out)
        return out + self.residual(x)


class ConditionalUnet1D(nn.Module):
    """Denoises an action sequence (B, T, act_dim) given (timestep, global_cond)."""

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        diffusion_step_embed_dim: int = 128,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        dsed = diffusion_step_embed_dim
        self.step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed)
        )
        cond_dim = dsed + global_cond_dim

        dims = (input_dim, *down_dims)
        in_out = list(zip(dims[:-1], dims[1:]))

        self.down = nn.ModuleList()
        for i, (ci, co) in enumerate(in_out):
            last = i == len(in_out) - 1
            self.down.append(
                nn.ModuleList(
                    [
                        ConditionalResBlock1d(ci, co, cond_dim, kernel_size, n_groups),
                        ConditionalResBlock1d(co, co, cond_dim, kernel_size, n_groups),
                        nn.Conv1d(co, co, 3, stride=2, padding=1) if not last else nn.Identity(),
                    ]
                )
            )

        mid = down_dims[-1]
        self.mid = nn.ModuleList(
            [
                ConditionalResBlock1d(mid, mid, cond_dim, kernel_size, n_groups),
                ConditionalResBlock1d(mid, mid, cond_dim, kernel_size, n_groups),
            ]
        )

        # note: both up levels upsample (down path has len-1 downsamples), and
        # the first down level's skip is intentionally unused — both quirks
        # match the reference implementation exactly.
        self.up = nn.ModuleList()
        for ci, co in reversed(in_out[1:]):
            self.up.append(
                nn.ModuleList(
                    [
                        ConditionalResBlock1d(co * 2, ci, cond_dim, kernel_size, n_groups),
                        ConditionalResBlock1d(ci, ci, cond_dim, kernel_size, n_groups),
                        nn.ConvTranspose1d(ci, ci, 4, stride=2, padding=1),
                    ]
                )
            )

        self.final = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D); timestep: (B,) or scalar; global_cond: (B, C)."""
        x = x.movedim(-1, -2)  # (B, D, T)
        if timestep.ndim == 0:
            timestep = timestep.expand(x.shape[0])
        cond = torch.cat([self.step_encoder(timestep), global_cond], dim=-1)

        skips = []
        for res1, res2, down in self.down:
            x = res2(res1(x, cond), cond)
            skips.append(x)
            x = down(x)
        for block in self.mid:
            x = block(x, cond)
        for res1, res2, up in self.up:
            x = torch.cat([x, skips.pop()], dim=1)
            x = res2(res1(x, cond), cond)
            x = up(x)
        return self.final(x).movedim(-2, -1)
