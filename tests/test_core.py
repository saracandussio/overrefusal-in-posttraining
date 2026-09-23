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
