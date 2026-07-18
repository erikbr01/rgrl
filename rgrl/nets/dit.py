"""Action-chunk DiT, ported from ABC-DiT (https://github.com/amazon-far/abc,
abc_minimal/dit.py) and adapted to rgrl's conditioning: ABC cross-attends
each block to DINOv3 vision tokens and folds [state, CLIP task vec,
timestep] into the AdaLN conditioning; rgrl's ObsEncoder already reduces the
whole obs window (images + states) to one flat vector, so the vision
cross-attention path (and its 3 extra modulation terms) is dropped and
blocks are standard AdaLN-Zero DiT blocks (Peebles & Xie) — self-attention
over action tokens + MLP, 6-way modulation.

What makes this the right backbone for train-time RTC: the conditioning `c`
may be per-token, (B, T, hidden) — TimestepEmbedder accepts (B,) *or* (B, T)
timesteps, and modulate()/gate_residual() broadcast either shape — so a
conditioning action prefix is expressed exactly as in ABC: those positions
get t=0 (clean) while the rest carry the real flow time, no mask-channel
workaround needed (compare rgrl/policies/diffusion.py's UNet variant).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


def gate_residual(gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    if gate.ndim == 2:
        gate = gate.unsqueeze(1)
    return gate * residual


def sincos_pos_embed_1d(embed_dim: int, length: int) -> torch.Tensor:
    """(length, embed_dim) fixed sin/cos position embedding."""
    omega = torch.arange(embed_dim // 2, dtype=torch.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    out = torch.arange(length, dtype=torch.float64)[:, None] * omega[None, :]
    return torch.cat([out.sin(), out.cos()], dim=1).float()


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + MLP; t may be (B,) or (B, T) — the
    per-position form is what carries prefix conditioning (t=0 at committed
    positions)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_shape = t.shape
        args = t.reshape(-1).float()[:, None] * self.freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype)).reshape(*t_shape, -1)


class DiTAttention(nn.Module):
    """Self-attention over action tokens (timm-equivalent, qkv_bias=True)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class DiTMlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """AdaLN-Zero DiT block (6-way modulation; ABC's is 9-way, the extra 3
    gating its vision cross-attention, which rgrl conditions via the flat
    obs vector instead)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiTAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = DiTMlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_residual(gate_msa, self.attn(modulate(self.norm1(x), shift_msa, scale_msa)))
        x = x + gate_residual(gate_mlp, self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, action_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class ActionDiT(nn.Module):
    """Velocity predictor over an action chunk: x (B, T, act_dim) and
    conditioning c (B, hidden) or (B, T, hidden) -> (B, T, act_dim)."""

    def __init__(
        self,
        action_dim: int,
        chunk_length: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.y_embedder = nn.Linear(action_dim, hidden_size)
        self.register_buffer(
            "pos_embed", sincos_pos_embed_1d(hidden_size, chunk_length)[None], persistent=False
        )
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.final_layer = FinalLayer(hidden_size, action_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Standard DiT init: trunc-normal linears, then zero every adaLN
        modulation projection and the final output layer (AdaLN-Zero —
        blocks start as identity, training is stable from step 0)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z = self.y_embedder(x) + self.pos_embed[:, : x.shape[1]]
        for block in self.blocks:
            z = block(z, c)
        return self.final_layer(z, c)
