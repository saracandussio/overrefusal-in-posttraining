"""Geometry of the three groups along the refusal axis.

Every quantity here is a function of five centroids, computed per
(checkpoint, layer, position):

    mu_harmless, mu_harmful
    mu_ref, mu_acc     pseudo-harmful prompts the model refused / answered
    mu_pseudo          their weighted mean

Directions
    v_ref   = mu_harmful - mu_harmless        the refusal axis
    v_over  = mu_pseudo  - mu_harmless        where pseudo-harmful sit
    v_beh   = mu_ref     - mu_acc             what separates refused from answered pseudo

Measures (the old scripts computed the same things under different names)
    t         projection of mu_pseudo on the axis: 0 = harmless, 1 = harmful.
              old `boundary_margin_n` == 2t - 1 (compute_entanglement.py)
              old `t`                            (centroid_axis.py)
    off_axis  distance of mu_pseudo from the axis, in axis lengths. If large,
              "pseudo sits between harmless and harmful" is the wrong picture.
    entanglement  cos(v_ref, v_over)
              old `cos_pseudo_harmless`          (explore_comprehension_decision.py)
    cos(v_beh, v_ref), cos(v_beh, v_over), and cos(v_beh, v_over_orth) where
    v_over_orth is v_over with its v_ref component removed. If the last one is
    ~0, refusal of pseudo-harmful prompts runs along v_ref alone.

Because they depend only on centroids, confidence intervals come from a
stratified bootstrap of the centroids (resampling weights, never copying rows).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.full_like(v, np.nan)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(unit(a) @ unit(b))


def remove_component(v: np.ndarray, along: np.ndarray) -> np.ndarray:
    u = unit(along)
    return v - (v @ u) * u


def axis_position(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    """(t, off_axis) of `point` relative to the segment start -> end."""
    axis = end - start
    length2 = float(axis @ axis)
    rel = point - start
    t = float(rel @ axis) / length2
    off = float(np.linalg.norm(rel - t * axis)) / np.sqrt(length2)
    return t, off


def measures(mu_harmless, mu_harmful, mu_ref, mu_acc, n_ref: int, n_acc: int) -> dict:
    mu_pseudo = (n_ref * mu_ref + n_acc * mu_acc) / (n_ref + n_acc)
    v_ref = mu_harmful - mu_harmless
    v_over = mu_pseudo - mu_harmless
    v_beh = mu_ref - mu_acc

    t, off = axis_position(mu_pseudo, mu_harmless, mu_harmful)
    t_ref, _ = axis_position(mu_ref, mu_harmless, mu_harmful)
    t_acc, _ = axis_position(mu_acc, mu_harmless, mu_harmful)
    return {
        "t": t, "off_axis": off, "t_refused": t_ref, "t_answered": t_acc,
        "entanglement": cos(v_ref, v_over),
        "cos_vbeh_vref": cos(v_beh, v_ref),
        "cos_vbeh_vover": cos(v_beh, v_over),
        "cos_vbeh_vover_orth": cos(v_beh, remove_component(v_over, v_ref)),
    }


def nearest_side(X: np.ndarray, mu_harmless: np.ndarray, mu_harmful: np.ndarray) -> np.ndarray:
    """1 where a row is closer to the harmful centroid (= its own t > 0.5)."""
    return (np.linalg.norm(X - mu_harmful, axis=1) < np.linalg.norm(X - mu_harmless, axis=1))


def _boot_means(X: np.ndarray, n_boot: int, rng) -> np.ndarray:
    n = len(X)
    W = rng.multinomial(n, np.full(n, 1 / n), size=n_boot).astype(np.float32) / n
    return W @ X


def cell(X_harmless, X_harmful, X_ref, X_acc, n_boot: int = 0, seed: int = 0) -> dict:
    """All measures for one cell, with 95% bootstrap intervals if n_boot > 0."""
    means = [X.mean(0) for X in (X_harmless, X_harmful, X_ref, X_acc)]
    out = measures(*means, len(X_ref), len(X_acc))
    out.update(n_harmless=len(X_harmless), n_harmful=len(X_harmful),
               n_refused=len(X_ref), n_answered=len(X_acc),
               pseudo_read_harmful=float(np.mean(nearest_side(
                   np.vstack([X_ref, X_acc]), means[0], means[1]))))
    if n_boot:
        rng = np.random.default_rng(seed)
        boots = [_boot_means(X, n_boot, rng) for X in (X_harmless, X_harmful, X_ref, X_acc)]
        draws = [measures(*(b[i] for b in boots), len(X_ref), len(X_acc)) for i in range(n_boot)]
        for key in draws[0]:
            lo, hi = np.nanpercentile([d[key] for d in draws], [2.5, 97.5])
            out[f"{key}_lo"], out[f"{key}_hi"] = float(lo), float(hi)
    return out


def centroid_drift(centroids: dict[str, np.ndarray], pairs: list[tuple[str, str]]) -> dict:
    """cos between the same group's centroid in two checkpoints."""
    return {f"{a}->{b}": cos(centroids[a], centroids[b]) for a, b in pairs}
