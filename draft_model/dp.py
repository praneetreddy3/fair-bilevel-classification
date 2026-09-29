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
    """Build a DPConfig from CLI args, forcing variant='none' when dp_enabled is False."""
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


def analytic_gaussian_epsilon(sigma: float, clip_min: float, clip_max: float,
                              d: int, delta: float = 1e-5) -> float:
    """Single-release (epsilon, delta) for the Gaussian mechanism used in _noisify_features.

    Sensitivity: one client's feature row changing (the standard "remove/replace one
    record" neighboring-dataset definition) can move the released vector by at most the
    L2 diameter of the per-dimension clip box, i.e. sqrt(d) * (clip_max - clip_min).
    Uses the standard analytic Gaussian-mechanism bound:
        epsilon = (sensitivity / sigma) * sqrt(2 * ln(1.25 / delta))
    This is the textbook (Dwork & Roth) bound, not the tighter numeric/moments-accountant
    analysis -- adequate for reporting a defensible order-of-magnitude budget, not a
    publication-grade tight accountant.
    """
    if sigma <= 0:
        return float("inf")
    sensitivity = np.sqrt(d) * (clip_max - clip_min)
    return float((sensitivity / sigma) * np.sqrt(2.0 * np.log(1.25 / delta)))


def composed_epsilon_basic(epsilon_per_release: float, num_releases: int) -> float:
    """Basic (linear, non-tight) composition: epsilon_total = num_releases * epsilon_per_release.

    Reported alongside the tighter advanced-composition bound for context; basic
    composition always upper-bounds the true privacy loss, so it is a safe (if loose)
    number to quote.
    """
    return float(epsilon_per_release * num_releases)


def composed_epsilon_advanced(epsilon_per_release: float, num_releases: int,
                              delta_prime: float = 1e-5) -> float:
    """Advanced composition (Dwork, Rothblum, Vadhan 2010 style) bound:
        epsilon_total ~= sqrt(2*k*ln(1/delta')) * epsilon + k*epsilon*(e^epsilon - 1)
    where k = num_releases. Tighter than basic composition for many releases.
    """
    k = num_releases
    eps = epsilon_per_release
    return float(np.sqrt(2.0 * k * np.log(1.0 / delta_prime)) * eps + k * eps * (np.exp(eps) - 1.0))


def report_privacy_budget(cfg: DPConfig, d: int, num_rounds: int, num_clients: int,
                          delta: float = 1e-5) -> dict:
    """Full budget summary for a T-round, K-client run releasing one noised payload
    per client per round (T*K total releases under pre_server DP; K under post_server;
    T*K + K under both -- see cfg.variant)."""
    if not cfg.is_enabled():
        return {"enabled": False}
    eps1 = analytic_gaussian_epsilon(cfg.sigma, cfg.clip_min, cfg.clip_max, d, delta=delta)
    if cfg.variant == "pre_server":
        k = num_rounds * num_clients
    elif cfg.variant == "post_server":
        k = num_rounds
    else:  # "both"
        k = num_rounds * num_clients + num_rounds
    return {
        "enabled": True,
        "variant": cfg.variant,
        "sigma": cfg.sigma,
        "clip_range": [cfg.clip_min, cfg.clip_max],
        "delta": delta,
        "epsilon_per_release": eps1,
        "num_releases": k,
        "epsilon_basic_composition": composed_epsilon_basic(eps1, k),
        "epsilon_advanced_composition": composed_epsilon_advanced(eps1, k, delta_prime=delta),
        "note": "Analytic Gaussian-mechanism bound (Dwork & Roth), not a tight moments "
                "accountant; basic composition is a safe upper bound, advanced composition "
                "is tighter for large num_releases.",
    }
