"""
Differential privacy utilities shared across training scripts.

This module centralizes where DP is applied so experiments can switch between:
- no DP
- client-side pre-server DP
- server-side post-aggregation DP
- both stages
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .notation import Payload, SyntheticMinibatch, UniversumSet


DP_VARIANTS = ("none", "pre_server", "post_server", "both")


@dataclass(frozen=True)
class DPConfig:
    variant: str = "post_server"
    sigma: float = 1.0
    clip_min: float = -1.0
    clip_max: float = 1.0

    def is_enabled(self) -> bool:
        return self.variant != "none" and self.sigma > 0

    def use_pre_server(self) -> bool:
        return self.is_enabled() and self.variant in {"pre_server", "both"}

    def use_post_server(self) -> bool:
        return self.is_enabled() and self.variant in {"post_server", "both"}


def resolve_dp_config(dp_enabled: bool, dp_sigma: float, dp_variant: str) -> DPConfig:
    variant = str(dp_variant).strip().lower()
    if variant not in DP_VARIANTS:
        raise ValueError(f"Invalid dp_variant '{dp_variant}'. Expected one of {DP_VARIANTS}.")
    if not dp_enabled:
        variant = "none"
    return DPConfig(variant=variant, sigma=float(dp_sigma))


def _noisify_features(X: np.ndarray, cfg: DPConfig, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if len(X) == 0 or not cfg.is_enabled():
        return X
    if rng is None:
        rng = np.random.default_rng()
    X_clip = np.clip(X, cfg.clip_min, cfg.clip_max)
    noise = rng.normal(0.0, cfg.sigma, X_clip.shape)
    return X_clip + noise


def apply_pre_server_dp(payloads: list[Payload], cfg: DPConfig, rng: Optional[np.random.Generator] = None) -> List[Payload]:
    if not cfg.use_pre_server():
        return payloads
    out: List[Payload] = []
    for pl in payloads:
        syn = pl.synthetic
        uni = pl.universum
        syn_new = SyntheticMinibatch(
            X=_noisify_features(syn.X, cfg, rng),
            A=syn.A.copy(),
            Y=syn.Y.copy(),
            q_a0y0=syn.q_a0y0,
            q_a0y1=syn.q_a0y1,
            q_a1y0=syn.q_a1y0,
            q_a1y1=syn.q_a1y1,
            Delta_s=syn.Delta_s,
        )
        uni_new = UniversumSet(
            X=_noisify_features(uni.X, cfg, rng),
            A=uni.A.copy(),
        )
        out.append(Payload(synthetic=syn_new, universum=uni_new))
    return out


def apply_post_server_dp(X_agg: np.ndarray, cfg: DPConfig, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if not cfg.use_post_server():
        return X_agg
    return _noisify_features(X_agg, cfg, rng)
