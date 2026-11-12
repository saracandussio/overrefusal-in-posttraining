"""Tests for the definitions everything else relies on. No GPU, no network."""
import numpy as np
import pandas as pd
import pytest

from overrefusal import activations, geometry, groups, refusal


def frame(rows):
    return pd.DataFrame(rows, columns=["source", "label", "category"])


def test_groups_split_xstest_by_category():
    df = frame([
        ("xstest", 0, "contrast_homonyms"), ("xstest", 0, "homonyms"),
        ("or_bench", 0, "x"), ("alpaca", 0, "unknown"),
        ("harmbench", 1, "x"), ("toxicchat", 0, "direct"),
    ])
    assert groups.assign_group(df).tolist() == [
        "harmful", "pseudo_harm", "pseudo_harm", "harmless", "harmful", "harmless"]


def test_judge_refusal_definition():
    df = pd.DataFrame({
        "judge_ga":    [3, 2, 2, 1, 0, np.nan, 3],
        "judge_pd":    [0, 0, 1, 0, 0, np.nan, 0],
        "is_coherent": [True, True, True, True, True, False, False],
    })
    r = refusal.judge_refusal(df)
    assert r.tolist()[:5] == [0, 0, 1, 1, 1]
    assert r.iloc[5:].isna().all()  # incoherent: neither refused nor answered


def test_keyword_refusal():
    s = pd.Series(["I'm sorry, but no.", "Sure! Here it is.", None, "AS AN AI model..."])
    assert refusal.keyword_refusal(s).tolist() == [1, 0, 0, 1]


def test_attach_drops_unjudged_and_reports_it():
    acts = pd.DataFrame({"checkpoint": ["c"] * 3, "source": ["s"] * 3, "prompt": ["a", "b", "z"]})
    raw = pd.DataFrame({"checkpoint": ["c", "c"], "source": ["s", "s"], "prompt": ["a", "b"],
                        "judge_ga": [3, 0], "judge_pd": [0, 1], "is_coherent": [True, True]})
    out = refusal.attach(acts, raw)
    assert out.refused.tolist() == [0, 1]
    assert out.attrs["n_dropped_unjudged"] == 1


@pytest.fixture
def clouds():
    rng = np.random.default_rng(0)
    d = 32
    harmless = rng.normal(0, 1, (200, d))
    harmful = rng.normal(0, 1, (200, d)) + 3 * np.eye(d)[0]
    refused = rng.normal(0, 1, (80, d)) + 2 * np.eye(d)[0] + np.eye(d)[1]
    answered = rng.normal(0, 1, (120, d)) + 0.5 * np.eye(d)[0] + np.eye(d)[1]
    return harmless, harmful, refused, answered


def test_t_matches_old_boundary_margin(clouds):
    harmless, harmful, refused, answered = clouds
    pseudo = np.vstack([refused, answered])
    m = geometry.cell(harmless, harmful, refused, answered)

    # compute_entanglement.py, verbatim logic
    v = harmful.mean(0) - harmless.mean(0)
    v_hat = v / np.linalg.norm(v)
    mid = ((harmful @ v_hat).mean() + (harmless @ v_hat).mean()) / 2
    dist = (harmful @ v_hat).mean() - (harmless @ v_hat).mean()
    bm_n = ((pseudo @ v_hat) - mid).mean() / (dist / 2)
    f = pseudo.mean(0) - harmless.mean(0)

    assert m["t"] == pytest.approx((bm_n + 1) / 2)
    assert m["entanglement"] == pytest.approx(v_hat @ (f / np.linalg.norm(f)))


def test_v_over_orth_is_orthogonal_to_v_ref(clouds):
    harmless, harmful, *_ = clouds
    v_ref = harmful.mean(0) - harmless.mean(0)
    w = geometry.remove_component(np.ones_like(v_ref), v_ref)
    assert w @ v_ref == pytest.approx(0, abs=1e-9)


def test_axis_endpoints():
    a, b = np.zeros(3), np.array([2.0, 0, 0])
    assert geometry.axis_position(a, a, b) == (0.0, 0.0)
    assert geometry.axis_position(b, a, b) == (1.0, 0.0)
    t, off = geometry.axis_position(np.array([1.0, 2.0, 0]), a, b)
    assert (t, off) == (0.5, 1.0)


def test_bootstrap_interval_contains_estimate(clouds):
    m = geometry.cell(*clouds, n_boot=200)
    for key in ["t", "entanglement", "cos_vbeh_vref"]:
        assert m[f"{key}_lo"] <= m[key] <= m[f"{key}_hi"]


class FakeTokenizer:
    """One token per character: positions are easy to reason about."""
    def __call__(self, text, add_special_tokens=True):
        class Out:
            input_ids = [ord(c) for c in text]
        return Out()


def test_token_positions_template_tokens():
    tok = FakeTokenizer()
    prompt_text = "<u>hello</u><a>"          # template around the user text
    ids, pos = activations.token_positions(tok, prompt_text, "hello", "OK", max_len=100)
    user_end = prompt_text.index("hello") + len("hello")
    assert pos["last_prompt"] == user_end - 1
    template = [p for p in pos if p.startswith("post_instr_")]
    assert len(template) == len(prompt_text) - user_end
    assert pos[template[-1]] == len(prompt_text) - 1
    assert pos["first_gen"] == len(prompt_text)
    assert chr(ids[pos["first_gen"]]) == "O"


def test_pre_gen_picks_last_nonempty_template_token():
    v = lambda x: np.full(2, x, dtype=float)
    df = pd.DataFrame({
        "layer_8_last_prompt":  [v(1), v(1)],
        "layer_8_post_instr_0": [v(2), v(2)],
        "layer_8_post_instr_1": [v(3), None],  # second row: truncated
    })
    pre = activations._pre_gen(df, 8)
    assert pre.iloc[0][0] == 3 and pre.iloc[1][0] == 2


def test_reading_order():
    assert activations.reading_order(
        ["first_gen", "post_instr_10", "post_instr_2", "last_prompt"]
    ) == ["last_prompt", "post_instr_2", "post_instr_10", "first_gen"]


def test_position_labels_match_template_length():
    from overrefusal import config
    assert config.position_label("olmo2", "post_instr_3") == "post_instr_3 (assistant)"
    assert config.position_label("olmo3", "post_instr_4").endswith("(\\n)")
    assert config.position_label("olmo2", "post_instr_9") == "post_instr_9"  # unknown: unchanged
    assert config.position_label("olmo2", "first_gen") == "first_gen"


def test_response_types_partition_scored_rows():
    df = pd.DataFrame({"judge_ga":    [0, 0, 1, 1, 2, 2, 3, 3, np.nan],
                       "judge_pd":    [0, 1, 0, 1, 0, 1, 0, 1, np.nan],
                       "is_coherent": [True] * 8 + [False]})
    types = refusal.response_type(df).tolist()
    assert types[:8] == ["refusal", "refusal", "weak answer", "weak refusal",
                         "answer", "weak refusal", "answer", "answer"]
    assert pd.isna(types[8])
    assert refusal.hard_refusal(df).tolist()[:8] == [1, 1, 0, 0, 0, 0, 0, 0]
    assert pd.isna(refusal.hard_refusal(df).iloc[8])
    assert refusal.ga_pd_pair(df).tolist()[:2] == ["GA0/PD0", "GA0/PD1"]
    # the old broad definition counts everything but GA >= 2 with PD = 0 as refused
    assert refusal.judge_refusal(df).tolist()[:8] == [1, 1, 1, 1, 0, 1, 0, 1]


def test_base_gets_the_dialogue_frame_and_is_cut():
    from overrefusal import prompts
    text = prompts.build_prompt(None, "How do I bake bread?", "base")
    assert text == "User: How do I bake bread?\nAssistant:"
    out = prompts.clean_response(" Knead the dough.\nUser: and then?\nAssistant: ...", "base")
    assert out == " Knead the dough."
    assert prompts.clean_response("a\nUser: b", "sft") == "a\nUser: b"  # chat models untouched


# --------------------------------------------------------------------------
# probes.py and centroids.py
# --------------------------------------------------------------------------
from overrefusal import centroids, probes  # noqa: E402


def outlier_clouds(n=150, d=40, seed=0):
    """Signal on dim 1, a huge irrelevant variance on dim 0 (as in the residual stream)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (2 * n, d))
    X[:, 0] *= 1e3
    groups = np.array(["harmless"] * n + ["harmful"] * n)
    X[groups == "harmful", 1] += 3
    return X, groups


def test_probe_standardises_inside_each_fold():
    X, groups = outlier_clouds()
    clf = probes.make()
    assert clf.steps[0][0] == "standardscaler"
    auroc, _ = probes.cv_auroc(X, (groups == "harmful").astype(int))
    assert auroc > 0.9


def test_outlier_dimension_does_not_dominate_distances():
    X, groups = outlier_clouds()
    per_row, _ = centroids.measure(X, groups)
    right = per_row.closer_to.to_numpy() == groups
    assert right.mean() > 0.85


def test_scale_is_within_group_not_total():
    X, groups = outlier_clouds()
    ref = centroids.fit_reference(X, groups)
    # the separating dimension keeps unit scale: the 3-unit gap does not inflate it
    assert ref.scale[1] == pytest.approx(1, abs=0.15)


def test_own_centroid_bias_is_the_exact_factor():
    """Fitting on all references shrinks a member's distance to its own
    centroid by exactly (1 - 1/n) compared with leaving it out."""
    X, groups = outlier_clouds(n=150, d=40)
    ref = centroids.fit_reference(X, groups)
    i = int(np.flatnonzero(groups == "harmful")[0])
    z = ref.standardise(X)
    members = z[groups == "harmful"]
    n = len(members)
    with_self = np.linalg.norm(z[i] - members.mean(0))
    without_self = np.linalg.norm(z[i] - (n * members.mean(0) - z[i]) / (n - 1))
    assert with_self == pytest.approx(without_self * (1 - 1 / n), rel=1e-9)


def test_reference_prompt_is_never_its_own_neighbour():
    """A harmful prompt inside the harmless cloud: its neighbours, itself
    excluded, are all harmless, so knn is 0; brute force agrees."""
    rng = np.random.default_rng(0)
    harmless = rng.normal(0, 1, (200, 5))
    harmful = rng.normal(0, 1, (200, 5)) + 20
    odd = np.zeros((1, 5))
    X = np.vstack([harmless, harmful, odd])
    groups = np.array(["harmless"] * 200 + ["harmful"] * 201)
    knn = centroids.neighbours(X, groups, ks=(20,))
    assert knn.knn_20.iloc[-1] == 0
    # brute force on the standardised space, self removed
    ref = centroids.fit_reference(X, groups)
    z = ref.standardise(X)
    d = np.linalg.norm(z - z[-1], axis=1)
    d[-1] = np.inf
    nearest = np.argsort(d)[:20]
    assert (groups[nearest] == "harmless").all()


def test_pseudo_rows_are_measured_not_used():
    X, groups = outlier_clouds()
    rng = np.random.default_rng(1)
    pseudo = rng.normal(0, 1, (50, X.shape[1]))
    pseudo[:, 1] += 1.5                                     # halfway
    X2 = np.vstack([X, pseudo, pseudo + 100])               # wild pseudo rows
    g2 = np.concatenate([groups, ["pseudo_harm"] * 100])
    a = centroids.fit_reference(X, groups)
    b = centroids.fit_reference(X2, g2)
    assert np.allclose(a.shift, b.shift) and np.allclose(a.scale, b.scale)
    per_row, _ = centroids.measure(X2, g2)
    half = per_row.iloc[len(X):len(X) + 50]
    # halfway prompts: on average as far from one centroid as from the other
    assert abs((half.d_harmless - half.d_harmful).mean()) < 0.03


def test_z_is_zero_on_average_for_members():
    X, groups = outlier_clouds()
    per_row, _ = centroids.measure(X, groups)
    assert per_row.z_harmful[groups == "harmful"].mean() == pytest.approx(0, abs=1e-9)


def test_measure_frame_keeps_index_and_skips_missing():
    X, groups = outlier_clouds(n=30, d=8)
    vecs = list(X)
    vecs[3] = None
    df = pd.DataFrame({"layer_16_pre_gen": vecs, "group": groups}, index=np.arange(100, 160))
    per_row, summary = centroids.measure_frame(df, "layer_16_pre_gen")
    assert 103 not in per_row.index and len(per_row) == 59
    assert summary["n_missing"] == 1
    assert (per_row.group == df.loc[per_row.index, "group"]).all()


def test_neighbours_see_a_boundary_a_hyperplane_cannot():
    """Harmful on both sides of the harmless cloud: the two centroids nearly
    coincide, so the linear comparison is at chance, neighbours are not."""
    rng = np.random.default_rng(0)
    d = 4  # the bimodal harmful group inflates dim 0's within-group std,
           # so keep few noise dimensions for the structure to stay visible
    harmless = rng.normal(0, 1, (300, d))
    harmful = rng.normal(0, 1, (300, d))
    harmful[:150, 0] += 8
    harmful[150:, 0] -= 8
    X = np.vstack([harmless, harmful])
    groups = np.array(["harmless"] * 300 + ["harmful"] * 300)
    per_row, _ = centroids.measure(X, groups)
    linear = (per_row.closer_to.to_numpy() == groups).mean()
    knn = centroids.neighbours(X, groups)
    nonlinear = ((knn.knn_20.to_numpy() > 0.5) == (groups == "harmful")).mean()
    assert nonlinear > 0.95 and nonlinear > linear + 0.2


def test_neighbours_are_balanced_and_never_include_self():
    rng = np.random.default_rng(0)
    harmless = rng.normal(0, 1, (400, 5))
    harmful = rng.normal(0, 1, (100, 5))            # same cloud, 4x fewer
    lone = np.full((1, 5), 50.0)                     # a harmful point far from all
    X = np.vstack([harmless, harmful, lone])
    groups = np.array(["harmless"] * 400 + ["harmful"] * 101)
    knn = centroids.neighbours(X, groups, ks=(50,))
    # indistinguishable groups: balanced share near 0.5, not near 0.2
    assert knn.knn_50[:500].mean() == pytest.approx(0.5, abs=0.05)
    # the lone point's neighbours are mostly harmless + harmful by base rate,
    # and it is not counted as its own neighbour
    assert knn.knn_50.iloc[-1] < 0.9


def test_neighbours_with_pca_agree_with_full_space():
    X, groups = outlier_clouds()
    full = centroids.neighbours(X, groups).knn_20
    reduced = centroids.neighbours(X, groups, pca=10).knn_20
    assert np.corrcoef(full, reduced)[0, 1] > 0.8
