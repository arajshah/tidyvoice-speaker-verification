# wespeaker/models/ssl_pooling.py
# Minimal S1 speaker embedder for SSL features (e.g., WavLM Base via s3prl frontend)
#
# Expected input:
#   x: (B, T, F) frame-level SSL features produced by the frontend
# Output:
#   emb: (B, embed_dim) utterance embedding for speaker classification / scoring

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SSLStatsPooling(nn.Module):
    """
    Simple, strong baseline:
      x (B,T,F) -> (optional LayerNorm) -> frame MLP -> stats pooling (mean+std over T)
      -> linear -> embedding (B, embed_dim)

    This is intentionally minimal:
      - no masking/variable-length handling (assumes your pipeline chunks to fixed length)
      - no attention pooling
      - optional L2 norm on output embedding
    """

    def __init__(
        self,
        feat_dim: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        l2_norm: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        if feat_dim is None or int(feat_dim) <= 0:
            raise ValueError(f"feat_dim must be a positive int, got {feat_dim}")

        self.feat_dim = int(feat_dim)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.l2_norm = bool(l2_norm)
        self.eps = float(eps)

        self.ln = nn.LayerNorm(self.feat_dim) if use_layernorm else None

        self.frame_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)) if dropout and float(dropout) > 0.0 else nn.Identity(),
        )

        # mean+std => 2*hidden_dim
        self.out = nn.Linear(2 * self.hidden_dim, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            emb: (B, embed_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input (B,T,F), got shape={tuple(x.shape)}")

        if self.ln is not None:
            x = self.ln(x)

        h = self.frame_mlp(x)            # (B,T,H)
        mean = h.mean(dim=1)             # (B,H)
        var = (h - mean.unsqueeze(1)).pow(2).mean(dim=1)  # (B,H)
        std = torch.sqrt(var + self.eps) # (B,H)

        pooled = torch.cat([mean, std], dim=-1)  # (B,2H)
        emb = self.out(pooled)                   # (B,E)

        if self.l2_norm:
            emb = F.normalize(emb, p=2.0, dim=-1)

        return emb
