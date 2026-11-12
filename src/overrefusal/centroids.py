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

Scaler and centroids are fitted once, on all reference prompts. A reference
prompt therefore contributes to its own centroid, which shrinks its distance
to it by exactly a factor (1 - 1/n), n = size of its group: with about 1,700
to 1,900 prompts per group, about 0.05%, invisible at two decimals. (Until
Nov 2026 reference prompts were measured with 5 folds instead; that is only
needed when groups are small.)

Note: with two centroids, "closer to harmful than to harmless" is decided by
one hyperplane, i.e. it is a linear (one-direction) comparison. The two
distances together add one number (how far from both groups), nothing more.

5. Neighbours, the non-linear counterpart (`neighbours`). For every prompt,
   its k nearest reference prompts in the same standardised space (a
   reference prompt is never its own neighbour: it is removed from its own
   candidate list), and how many are harmful. No axis, no hyperplane: the boundary between the
   harmless and the harmful region can take any shape. The share is
   balanced for group size: knn_k = 0.5 means the neighbourhood is as
   harmful as the reference set as a whole, 1 all harmful, 0 all harmless.
   Several k at once, as a robustness check. In thousands of dimensions
   distances concentrate; the optional PCA (fitted on reference prompts
   only) is the second check: conclusions should not depend on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from overrefusal import activations

REFERENCE = ("harmless", "harmful")
KNN_KS = (10, 20, 50)
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
        """Distance between the two centroids, same units as `distances`.
        Equals the root mean square of `per_dimension_d`."""
        a, b = (self.centroids[g] for g in REFERENCE)
        return float(np.linalg.norm(a - b) / np.sqrt(len(a)))

    def per_dimension_d(self) -> np.ndarray:
        """For every dimension, (mean harmful - mean harmless) / pooled
        within-group std: a Cohen's d per dimension."""
        return self.centroids["harmful"] - self.centroids["harmless"]


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


def measure(X: np.ndarray, groups: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Distances of every row to both centroids, and the spread of each group.

    Scaler and centroids come from all reference rows (see the module note on
    why including a prompt in its own centroid does not matter at our sizes).

    Returns
      per_row  one row per row of X: d_harmless, d_harmful, z_harmless,
               z_harmful, closer_to ("harmless"/"harmful")
      summary  separation; the distribution of |d| over dimensions (Cohen's d
               per dimension: mean, sd, median, q90, q99, max, and how many
               dimensions exceed 0.2, 0.5, 0.8); and for each reference group
               its member distance mean, sd, median, q05, q95, plus counts
    """
    groups = np.asarray(groups)
    ref = fit_reference(X, groups)
    d = ref.distances(X)

    summary = {"separation": ref.separation(), "n_dims": int(X.shape[1])}
    d_abs = np.abs(ref.per_dimension_d())
    q50, q90, q99 = np.percentile(d_abs, [50, 90, 99])
    summary |= {"d_abs_mean": float(d_abs.mean()), "d_abs_sd": float(d_abs.std(ddof=1)),
                "d_abs_median": float(q50), "d_abs_q90": float(q90), "d_abs_q99": float(q99),
                "d_abs_max": float(d_abs.max()),
                "n_dims_d_over_0.2": int((d_abs > 0.2).sum()),
                "n_dims_d_over_0.5": int((d_abs > 0.5).sum()),
                "n_dims_d_over_0.8": int((d_abs > 0.8).sum())}
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
    })
    return per_row, summary


def measure_frame(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict]:
    """`measure` on one activation column of a frame from activations.load.

    Rows without a vector in `column` are skipped; per_row keeps the frame's
    index and its `group` column, so it joins back onto df (and its metadata).
    """
    rows = activations.has(df, column)
    per_row, summary = measure(activations.matrix(df, column), df.loc[rows, "group"].to_numpy())
    per_row.index = df.index[rows]
    per_row.insert(0, "group", df.loc[rows, "group"])
    summary["n_missing"] = int((~rows).sum())
    return per_row, summary


def _harmful_share(Q: np.ndarray, R: np.ndarray, r_groups: np.ndarray, ks,
                   self_index: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """Balanced share of harmful prompts among the k nearest rows of R, for each
    row of Q. self_index[i] = position in R of query i itself (-1 if it is not
    in R); that row is excluded from its own candidates."""
    kmax = max(ks)
    if kmax > len(R) - 1:
        raise ValueError(f"k = {kmax} but only {len(R)} reference prompts")
    is_harmful = r_groups == "harmful"
    n_h, n_s = is_harmful.sum(), (~is_harmful).sum()
    r_sq = (R ** 2).sum(1)
    out = {k: np.empty(len(Q)) for k in ks}
    for start in range(0, len(Q), 1024):  # bounded memory
        q = Q[start:start + 1024]
        d = (q ** 2).sum(1)[:, None] - 2 * q @ R.T + r_sq[None, :]
        if self_index is not None:
            own = self_index[start:start + 1024]
            rows = np.flatnonzero(own >= 0)
            d[rows, own[rows]] = np.inf
        idx = np.argpartition(d, kmax - 1, axis=1)[:, :kmax]
        idx = np.take_along_axis(idx, np.take_along_axis(d, idx, 1).argsort(1), 1)
        hit = is_harmful[idx]
        for k in ks:
            h = hit[:, :k].sum(1) / n_h
            s_ = (k - hit[:, :k].sum(1)) / n_s
            out[k][start:start + 1024] = h / (h + s_)
    return out


def _space(X_ref: np.ndarray, g_ref: np.ndarray, pca: int | None):
    """Standardise (and optionally reduce) with statistics of the reference rows only."""
    ref = fit_reference(X_ref, g_ref)
    if pca is None:
        return lambda X: ref.standardise(X).astype(np.float32)
    proj = PCA(n_components=pca, random_state=0).fit(ref.standardise(X_ref))
    return lambda X: proj.transform(ref.standardise(X)).astype(np.float32)


def neighbours(X: np.ndarray, groups: np.ndarray, ks=KNN_KS, pca: int | None = None) -> pd.DataFrame:
    """knn_<k> for every row of X: balanced harmful share among its k nearest
    reference prompts, in the space standardised on all reference prompts.
    A reference prompt is removed from its own candidate list."""
    groups = np.asarray(groups)
    is_ref = np.isin(groups, REFERENCE)
    ref_pos = np.full(len(X), -1)
    ref_pos[is_ref] = np.arange(is_ref.sum())
    f = _space(X[is_ref], groups[is_ref], pca)
    shares = _harmful_share(f(X), f(X[is_ref]), groups[is_ref], ks, self_index=ref_pos)
    return pd.DataFrame({f"knn_{k}": v for k, v in shares.items()})


def neighbours_frame(df: pd.DataFrame, column: str, ks=KNN_KS, pca: int | None = None) -> pd.DataFrame:
    """`neighbours` on one activation column; keeps the frame's index."""
    rows = activations.has(df, column)
    out = neighbours(activations.matrix(df, column), df.loc[rows, "group"].to_numpy(), ks=ks, pca=pca)
    out.index = df.index[rows]
    return out
