"""Where a prompt sits relative to the genuinely harmless and genuinely harmful
prompts, measured in the full activation space.

Reference groups (groups.py): `harmless` and `harmful`, defined by the
datasets' labels, never by the model's behaviour. Pseudo-harmful prompts are
never used to build anything here; they are only measured.

Per (checkpoint, layer, position) cell:

1. Standardise. Each dimension is centred and divided by its pooled
   within-group standard deviation (computed inside harmless and inside
   harmful, then averaged). Within-group, so that the gap between the two
   groups does not inflate the scale of the dimensions that separate them.
   A floor on the std keeps near-constant dimensions from exploding.

2. Two centroids, the means of the two reference groups in that space.

3. Distances. For every prompt, its Euclidean distance to each centroid,
   divided by sqrt(n_dims): the root-mean-square distance per dimension, in
   within-group standard deviations. Comparable across layers and families.

4. Spread, as a control. For the members of each reference group, the
   distribution of their distance to their own centroid (mean, sd,
   quantiles). This says what "close" means: a pseudo-harmful prompt at the
   harmless group's typical distance is as close as a typical harmless prompt.
   z_<group> = (d_<group> - mean member distance) / sd member distance.

Reference prompts are measured out of fold: scaler and centroids are refitted
without the fold being measured, so no prompt contributes to the centroid it
is compared with (in-sample distances to one's own centroid are biased low).

Note: with two centroids, "closer to harmful than to harmless" is decided by
one hyperplane. The distances themselves are not projections on that axis:
they also count everything off it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from overrefusal import activations

REFERENCE = ("harmless", "harmful")
STD_FLOOR = 1e-3  # times the median std: dimensions below it are not amplified


@dataclass(frozen=True)
class Reference:
    shift: np.ndarray                  # per-dimension centre
    scale: np.ndarray                  # pooled within-group std, floored
    centroids: dict[str, np.ndarray]   # in standardised coordinates

    def standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.shift) / self.scale

    def distances(self, X: np.ndarray) -> dict[str, np.ndarray]:
        Z = self.standardise(X)
        root_d = np.sqrt(Z.shape[1])
        return {g: np.linalg.norm(Z - mu, axis=1) / root_d for g, mu in self.centroids.items()}

    def separation(self) -> float:
        """Distance between the two centroids, same units as `distances`."""
        a, b = (self.centroids[g] for g in REFERENCE)
        return float(np.linalg.norm(a - b) / np.sqrt(len(a)))


def fit_reference(X: np.ndarray, groups: np.ndarray) -> Reference:
    """Scaler and centroids from the reference rows of X (other rows ignored)."""
    parts = {g: X[groups == g].astype(np.float64) for g in REFERENCE}
    for g, P in parts.items():
        if len(P) < 2:
            raise ValueError(f"need at least 2 {g} prompts, got {len(P)}")
    shift = np.mean([P.mean(0) for P in parts.values()], axis=0)  # midpoint: balance-free
    var = sum(((P - P.mean(0)) ** 2).sum(0) for P in parts.values())
    dof = sum(len(P) - 1 for P in parts.values())
    scale = np.sqrt(var / dof)
    scale = np.maximum(scale, STD_FLOOR * np.median(scale))
    centroids = {g: (P.mean(0) - shift) / scale for g, P in parts.items()}
    return Reference(shift, scale, centroids)


def measure(X: np.ndarray, groups: np.ndarray, folds: int = 5, seed: int = 0
            ) -> tuple[pd.DataFrame, dict]:
    """Distances of every row to both centroids, and the spread of each group.

    Returns
      per_row  one row per row of X: d_harmless, d_harmful, z_harmless,
               z_harmful, closer_to ("harmless"/"harmful"), out_of_fold (bool)
      summary  separation, and for each reference group its member distance
               mean, sd, median, q05, q95, plus counts
    """
    groups = np.asarray(groups)
    is_ref = np.isin(groups, REFERENCE)
    d = {g: np.full(len(X), np.nan) for g in REFERENCE}

    # reference rows: out of fold
    ref_idx = np.flatnonzero(is_ref)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(ref_idx, groups[ref_idx]):
        ref = fit_reference(X[ref_idx[train]], groups[ref_idx[train]])
        for g, v in ref.distances(X[ref_idx[test]]).items():
            d[g][ref_idx[test]] = v

    # everything else: one reference fitted on all reference rows
    full = fit_reference(X, groups)
    if (~is_ref).any():
        for g, v in full.distances(X[~is_ref]).items():
            d[g][~is_ref] = v

    summary = {"separation": full.separation(), "n_dims": int(X.shape[1])}
    own = {}
    for g in REFERENCE:
        member = d[g][groups == g]
        own[g] = (member.mean(), member.std(ddof=1))
        q05, med, q95 = np.percentile(member, [5, 50, 95])
        summary |= {f"n_{g}": int(len(member)), f"{g}_mean": float(own[g][0]),
                    f"{g}_sd": float(own[g][1]), f"{g}_median": float(med),
                    f"{g}_q05": float(q05), f"{g}_q95": float(q95)}

    per_row = pd.DataFrame({
        "d_harmless": d["harmless"], "d_harmful": d["harmful"],
        "z_harmless": (d["harmless"] - own["harmless"][0]) / own["harmless"][1],
        "z_harmful": (d["harmful"] - own["harmful"][0]) / own["harmful"][1],
        "closer_to": np.where(d["harmful"] < d["harmless"], "harmful", "harmless"),
        "out_of_fold": is_ref,
    })
    return per_row, summary


def measure_frame(df: pd.DataFrame, column: str, folds: int = 5, seed: int = 0
                  ) -> tuple[pd.DataFrame, dict]:
    """`measure` on one activation column of a frame from activations.load.

    Rows without a vector in `column` are skipped; per_row keeps the frame's
    index and its `group` column, so it joins back onto df (and its metadata).
    """
    rows = activations.has(df, column)
    per_row, summary = measure(activations.matrix(df, column), df.loc[rows, "group"].to_numpy(),
                               folds=folds, seed=seed)
    per_row.index = df.index[rows]
    per_row.insert(0, "group", df.loc[rows, "group"])
    summary["n_missing"] = int((~rows).sum())
    return per_row, summary
