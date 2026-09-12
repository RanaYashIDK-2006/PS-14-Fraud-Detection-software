"""Differential privacy for the federated prototype (architecture section 19).

Client-level DP FedAvg (McMahan et al., "Federated Learning with Differential
Privacy"): each per-client weight UPDATE is clipped to a norm bound S, the
clipped deltas are averaged (weighted by local size, as in FedAvg), and the
coordinator adds Gaussian noise calibrated so the whole multi-round run
satisfies a total (epsilon, delta)-DP guarantee.

Privacy accounting is Rényi differential privacy (RDP, Mironov 2017), which
composes tightly over the simulation's rounds: each round's Gaussian
mechanism contributes alpha / (2 sigma^2) Rényi divergence at order alpha,
and the RDP->(eps, delta) conversion is applied to the summed budget.

Honest scope (mirrors the federated report's caveats):
  * CENTRAL DP: the coordinator adds the noise. Individual clipped updates
    still cross the boundary, so a real deployment pairs this with secure
    aggregation so the server never observes them.
  * delta is a global failure probability; with the demo's tiny client count
    (3) the guarantee is weak in an absolute sense - production would use
    delta ~ 1/N_clients or smaller and a battle-tested accountant (e.g.
    Google's DP library / Opacus).
  * sigma is reported normalized to sensitivity 1; the applied noise standard
    deviation is sigma * S (clip norm).
"""

from __future__ import annotations

import numpy as np

# RDP orders to search when converting a target epsilon to a noise scale.
ALPHA_GRID = np.arange(2.0, 65.0, 1.0)


def clip_delta(delta: np.ndarray, clip_norm: float) -> np.ndarray:
    """Clip a concatenated [dw; db] update vector to Euclidean norm S."""
    norm = float(np.linalg.norm(delta))
    if norm <= clip_norm or norm == 0.0:
        return delta.copy()
    return delta * (clip_norm / norm)


def rdp_gaussian_eps(sigma: float, rounds: int, alpha: float) -> float:
    """RDP budget (order alpha) of `rounds` Gaussian mechanisms of scale
    sigma (sensitivity 1): rounds * alpha / (2 sigma^2)."""
    return rounds * alpha / (2.0 * sigma * sigma)


def eps_from_rdp(rdp: float, delta: float, alpha: float) -> float:
    """Convert an (alpha, rdp)-RDP guarantee into (eps, delta)-DP
    (Mironov 2017 conversion for alpha > 1)."""
    return rdp + (np.log(1.0 / delta) + (alpha - 1.0) * np.log(1.0 - 1.0 / alpha)
                  - np.log(alpha)) / (alpha - 1.0)


def noise_scale_for_eps(epsilon: float, delta: float, rounds: int,
                        alpha_grid: np.ndarray | None = None) -> float:
    """Minimal Gaussian noise scale sigma (normalized: sensitivity 1) such
    that the RDP-composed `rounds`-round run satisfies (epsilon, delta)-DP.
    epsilon = inf -> 0.0 (the clean FedAvg baseline). Binary search over
    sigma, minimizing over the RDP order grid."""
    if epsilon == float("inf"):
        return 0.0
    grid = ALPHA_GRID if alpha_grid is None else np.asarray(alpha_grid, dtype=float)
    grid = grid[grid > 1.0]

    def eps_at(sigma: float) -> float:
        best = float("inf")
        for a in grid:
            e = eps_from_rdp(rdp_gaussian_eps(sigma, rounds, a), delta, a)
            if e < best:
                best = e
        return best

    lo, hi = 1e-6, 200.0
    for _ in range(70):  # 70 bisections -> ~1e-17 relative precision
        mid = 0.5 * (lo + hi)
        if eps_at(mid) <= epsilon:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def dp_aggregate(global_w: np.ndarray, global_b: float,
                 updates: list[tuple[np.ndarray, float, int]],
                 clip_norm: float, sigma: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """DP-FedAvg aggregation: clip each client's update delta to S, average
    the clipped deltas (weighted by local size), and add Gaussian noise with
    std = sigma * S. Returns the noisy (w, b). With sigma = 0 this is exactly
    plain FedAvg (deltas sum to the same average as absolute weights)."""
    dw = global_w.shape[0]
    total = sum(n for _, _, n in updates)
    acc = np.zeros(dw + 1)
    for w_i, b_i, n in updates:
        delta = np.concatenate([np.asarray(w_i) - global_w, [b_i - global_b]])
        acc += clip_delta(delta, clip_norm) * n
    acc /= total
    if sigma > 0.0 and clip_norm > 0.0:
        acc += rng.normal(0.0, sigma * clip_norm, size=acc.shape)
    new = np.concatenate([np.asarray(global_w), [global_b]]) + acc
    return new[:-1].astype(float), float(new[-1])


def dp_aggregate_flat(global_flat: np.ndarray,
                      updates: list[tuple[np.ndarray, int]],
                      clip_norm: float, sigma: float,
                      rng: np.random.Generator) -> np.ndarray:
    """DP-FedAvg over FLAT parameter vectors (the MLP path): clip each
    client's delta to S, weighted-average, add Gaussian noise std = sigma*S.
    Same mechanism as `dp_aggregate`, but the state is one flat vector
    (concatenated layers) instead of a (w, b) pair."""
    total = sum(n for _, n in updates)
    acc = np.zeros(np.asarray(global_flat).shape[0])
    for f_i, n in updates:
        acc += clip_delta(np.asarray(f_i) - global_flat, clip_norm) * n
    acc /= total
    if sigma > 0.0 and clip_norm > 0.0:
        acc += rng.normal(0.0, sigma * clip_norm, size=acc.shape)
    return (np.asarray(global_flat) + acc).astype(float)
