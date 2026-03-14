"""
Data structures matching the draft paper notation (Tables 1-2).
Used by minibatch_design, bilevel_al, and server modules.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class ClientOriginalData:
    """D^o_k: client k's private dataset with (x, a, y), a,y ∈ {0,1}."""
    X: np.ndarray   # (Nk, d) features
    A: np.ndarray   # (Nk,) sensitive attribute
    Y: np.ndarray   # (Nk,) label
    Nk: int = 0
    N_oa0y0: int = 0
    N_oa0y1: int = 0
    N_oa1y0: int = 0
    N_oa1y1: int = 0
    pi_a0y0: float = 0.0
    pi_a0y1: float = 0.0
    pi_a1y0: float = 0.0
    pi_a1y1: float = 0.0
    pi_plus: float = 0.0

    def __post_init__(self):
        self.Nk = len(self.Y)
        self.N_oa0y0 = int(np.sum((self.A == 0) & (self.Y == 0)))
        self.N_oa0y1 = int(np.sum((self.A == 0) & (self.Y == 1)))
        self.N_oa1y0 = int(np.sum((self.A == 1) & (self.Y == 0)))
        self.N_oa1y1 = int(np.sum((self.A == 1) & (self.Y == 1)))
        self.pi_a0y0 = self.N_oa0y0 / max(1, self.Nk)
        self.pi_a0y1 = self.N_oa0y1 / max(1, self.Nk)
        self.pi_a1y0 = self.N_oa1y0 / max(1, self.Nk)
        self.pi_a1y1 = self.N_oa1y1 / max(1, self.Nk)
        self.pi_plus = (self.N_oa0y1 + self.N_oa1y1) / max(1, self.Nk)


@dataclass
class OriginalMinibatch:
    """B_{k,t}: stratified minibatch from client k at round t."""
    X: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    q_a0y0: int = 0
    q_a0y1: int = 0
    q_a1y0: int = 0
    q_a1y1: int = 0
    n0: int = 0
    n1: int = 0
    Delta: int = 0  # label imbalance = n0 - n1

    @property
    def B_plus(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract true positives (y=1) for EO computation."""
        mask = self.Y == 1
        return self.X[mask], self.A[mask]


@dataclass
class SyntheticMinibatch:
    """D^s_{k,t}: synthetic minibatch with fixed (a,y) labels and learnable features."""
    X: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    q_a0y0: int = 0
    q_a0y1: int = 0
    q_a1y0: int = 0
    q_a1y1: int = 0
    Delta_s: int = 0

    def size(self) -> int:
        return len(self.Y)


@dataclass
class UniversumSet:
    """U^{(Y)}_{k,t}: pseudo-positive points (y=1 always) for minority-group fairness."""
    X: np.ndarray
    A: np.ndarray

    def size(self) -> int:
        return len(self.A)


@dataclass
class Payload:
    """S̃_{k,t} = D^s ∪ U: client-to-server message (no raw data)."""
    synthetic: SyntheticMinibatch
    universum: UniversumSet

    def total_size(self) -> int:
        return self.synthetic.size() + self.universum.size()
