"""
weighting.py — Dynamic Source Weighting for Multi-Source Domain Adaptation
===========================================================================
Weighting & Evaluation Strategist

Two mechanisms:
  1. CosineWeighter  : closed-form, no learnable params, fast & interpretable
  2. AttentionWeighter: small cross-attention module, end-to-end differentiable (WOW extra)

Usage
-----
    from evaluation.weighting import CosineWeighter, AttentionWeighter

    weighter = CosineWeighter(temperature=0.5)
    w1, w2 = weighter(target_embeddings, centroid_s1, centroid_s2)
    # w1, w2 are scalar tensors summing to 1, shape []

    att_weighter = AttentionWeighter(embed_dim=512).to(device)
    w1, w2 = att_weighter(target_embeddings, centroid_s1, centroid_s2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1. Cosine-Similarity Weighter (closed-form, no parameters)
# ──────────────────────────────────────────────────────────────────────────────

class CosineWeighter:
    """
    Computes per-batch weights (w1, w2) for two sources via cosine similarity
    between the mean target embedding and each source centroid.

    Steps
    -----
      1. avg_tgt  = mean(target_embeddings, dim=0)          # [D]
      2. sim_i    = cosine_similarity(avg_tgt, centroid_i)  # scalar ∈ [-1,1]
      3. (w1, w2) = softmax([sim_1, sim_2] / T)            # T = temperature

    Args
    ----
    temperature : float
        Sharpness of the softmax. Lower → more peaked weights. Default 1.0.
    eps : float
        Numerical stability for cosine similarity. Default 1e-8.
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-8):
        self.temperature = temperature
        self.eps = eps

    def __call__(
        self,
        target_embeddings: torch.Tensor,   # [B, D]
        centroid_s1: torch.Tensor,         # [D]
        centroid_s2: torch.Tensor,         # [D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        w1, w2 : scalar tensors (float32) summing to 1.
        """
        avg_tgt = target_embeddings.mean(dim=0)  # [D]

        sim1 = F.cosine_similarity(avg_tgt.unsqueeze(0),
                                   centroid_s1.unsqueeze(0),
                                   eps=self.eps).squeeze()   # scalar
        sim2 = F.cosine_similarity(avg_tgt.unsqueeze(0),
                                   centroid_s2.unsqueeze(0),
                                   eps=self.eps).squeeze()   # scalar

        logits = torch.stack([sim1, sim2]) / self.temperature  # [2]
        weights = torch.softmax(logits, dim=0)                  # [2]
        return weights[0], weights[1]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Attention-based Weighter  🌟 WOW extra
# ──────────────────────────────────────────────────────────────────────────────

class AttentionWeighter(nn.Module):
    """
    Differentiable cross-attention source weighter.

    Architecture
    ------------
    The target batch acts as QUERY; the two source centroids are KEYS/VALUES.
    A single-head scaled dot-product attention produces (w1, w2).

    All projections are shared (same W_q, W_k), keeping the parameter count
    very small (~3 * embed_dim^2 // reduction^2 params).

    Args
    ----
    embed_dim : int   — dimension of feature embeddings (e.g. 512)
    proj_dim  : int   — internal projection dimension (default embed_dim // 4)
    temperature : float — additional softmax temperature (default 1.0)
    """

    def __init__(
        self,
        embed_dim: int,
        proj_dim: int | None = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        proj_dim = proj_dim or max(embed_dim // 4, 16)
        self.temperature = temperature
        self.scale = proj_dim ** -0.5

        # Shared projections
        self.W_q = nn.Linear(embed_dim, proj_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, proj_dim, bias=False)

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)

    def forward(
        self,
        target_embeddings: torch.Tensor,  # [B, D]
        centroid_s1: torch.Tensor,        # [D]
        centroid_s2: torch.Tensor,        # [D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        w1, w2 : scalar tensors summing to 1 (differentiable).
        """
        # Query: mean-pooled target batch  [1, proj_dim]
        query = self.W_q(target_embeddings.mean(dim=0, keepdim=True))  # [1, P]

        # Keys: source centroids  [2, proj_dim]
        keys = self.W_k(torch.stack([centroid_s1, centroid_s2], dim=0))  # [2, P]

        # Scaled dot-product attention  [1, 2]
        attn_logits = (query @ keys.T) * self.scale / self.temperature   # [1, 2]
        weights = torch.softmax(attn_logits, dim=-1).squeeze(0)          # [2]

        return weights[0], weights[1]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Centroid Tracker  (maintains running centroids per source)
# ──────────────────────────────────────────────────────────────────────────────

class CentroidTracker:
    """
    Maintains an exponential-moving-average centroid for each source domain.

    Call `update(embeddings, source_id)` after each forward pass; retrieve
    the current centroid via `centroids[source_id]`.

    Args
    ----
    embed_dim  : int   — feature dimension
    momentum   : float — EMA momentum (default 0.99, high → slow update)
    num_sources: int   — number of source domains (default 2)
    """

    def __init__(
        self,
        embed_dim: int,
        momentum: float = 0.99,
        num_sources: int = 2,
    ):
        self.momentum = momentum
        self.centroids: list[torch.Tensor | None] = [None] * num_sources
        self.embed_dim = embed_dim

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, source_id: int):
        """
        embeddings : [B, D] — batch of L2-normalised or raw embeddings
        source_id  : int    — 0 = source 1, 1 = source 2
        """
        batch_mean = embeddings.mean(dim=0).cpu()  # [D]
        if self.centroids[source_id] is None:
            self.centroids[source_id] = batch_mean
        else:
            self.centroids[source_id] = (
                self.momentum * self.centroids[source_id]
                + (1.0 - self.momentum) * batch_mean
            )

    def get(self, source_id: int, device: torch.device | None = None) -> torch.Tensor:
        c = self.centroids[source_id]
        if c is None:
            raise RuntimeError(f"Centroid {source_id} not yet initialised — call update() first.")
        return c.to(device) if device is not None else c