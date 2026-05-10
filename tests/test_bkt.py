from __future__ import annotations

import pytest

from src.bkt import BKTParams, BKTPredictor, decay_mastery
from src.bkt.predictor import update


def make_params(L0=0.1, slip=0.1, guess=0.2, transit=0.15) -> BKTParams:
    return BKTParams(L0=L0, slip=slip, guess=guess, transit=transit)


# ---------- params ---------------------------------------------------


def test_invalid_param_raises():
    with pytest.raises(ValueError):
        BKTParams(L0=1.5, slip=0.1, guess=0.1, transit=0.1)
    with pytest.raises(ValueError):
        BKTParams(L0=0.1, slip=-0.1, guess=0.1, transit=0.1)


# ---------- single-step update --------------------------------------


def test_update_correct_increases_mastery():
    p = update(0.5, correct=True, params=make_params())
    assert p > 0.5


def test_update_incorrect_decreases_mastery_relative_to_correct():
    p_correct = update(0.5, correct=True, params=make_params())
    p_incorrect = update(0.5, correct=False, params=make_params())
    assert p_incorrect < p_correct


def test_update_stays_in_unit_interval():
    params = make_params()
    for prior in [0.0, 0.01, 0.5, 0.99, 1.0]:
        for correct in (True, False):
            p = update(prior, correct, params)
            assert 0.0 <= p <= 1.0


def test_update_invalid_prior_raises():
    with pytest.raises(ValueError):
        update(1.5, True, make_params())


def test_update_handles_degenerate_priors():
    # slip=0, guess=0 — incorrect from a fully-mastered student is impossible
    # under the model. Update should not blow up.
    params = make_params(slip=0.0, guess=0.0, transit=0.0)
    p = update(1.0, correct=False, params=params)
    assert 0.0 <= p <= 1.0


# ---------- decay ----------------------------------------------------


def test_decay_half_life():
    decayed = decay_mastery(0.8, days_elapsed=7, half_life_days=7)
    assert decayed == pytest.approx(0.4, abs=1e-9)


def test_decay_zero_days_is_noop():
    assert decay_mastery(0.7, 0.0, 7.0) == pytest.approx(0.7)


def test_decay_strictly_decreasing_in_days():
    p1 = decay_mastery(0.9, 1, 7)
    p2 = decay_mastery(0.9, 2, 7)
    p3 = decay_mastery(0.9, 7, 7)
    assert p1 > p2 > p3


def test_decay_invalid_inputs():
    with pytest.raises(ValueError):
        decay_mastery(0.5, -1.0, 7.0)
    with pytest.raises(ValueError):
        decay_mastery(0.5, 1.0, 0.0)
    with pytest.raises(ValueError):
        decay_mastery(1.5, 1.0, 7.0)


def test_decay_bounds():
    assert decay_mastery(0.0, 1, 7) == 0.0
    assert decay_mastery(1.0, 0, 7) == 1.0


# ---------- predictor ------------------------------------------------


def test_predictor_init_uses_L0():
    p = BKTPredictor(make_params(L0=0.42))
    p.init_topic("x")
    assert p.mastery("x") == pytest.approx(0.42)


def test_predictor_observe_updates_mastery():
    p = BKTPredictor(make_params(L0=0.3))
    p.init_topic("x")
    before = p.mastery("x")
    after = p.observe("x", correct=True)
    assert after > before
    assert p.mastery("x") == pytest.approx(after)


def test_predictor_unknown_topic_raises():
    p = BKTPredictor(make_params())
    with pytest.raises(KeyError):
        p.observe("missing", True)


def test_predictor_decay_reduces_mastery():
    p = BKTPredictor(make_params(L0=0.9))
    p.init_topic("x")
    before = p.mastery("x")
    after = p.apply_decay("x", days_elapsed=7, half_life_days=7)
    assert after < before
    assert after == pytest.approx(before * 0.5)


def test_predictor_double_init_is_idempotent():
    p = BKTPredictor(make_params(L0=0.5))
    p.init_topic("x")
    p._mastery["x"] = 0.9
    p.init_topic("x")
    assert p.mastery("x") == pytest.approx(0.9)
