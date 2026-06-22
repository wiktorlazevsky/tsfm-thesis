"""CPU unit tests for clean.py (no WRDS, no network)."""

import numpy as np
import pytest

from tsfm_cal import clean, config


def _dates(n, start="2020-01-01"):
    return np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(n, "D"))


# --------------------------------------------------------------------------- #
# Return conversion
# --------------------------------------------------------------------------- #
def test_crsp_ret_to_log():
    ret = np.array([0.0, 0.10, -0.05])
    out = clean.returns_from_crsp_ret(ret)
    assert np.allclose(out, np.log1p(ret))


def test_levels_to_log():
    lv = np.array([100.0, 110.0, 99.0])
    out = clean.returns_from_levels(lv)
    assert np.allclose(out, np.diff(np.log(lv)))


# --------------------------------------------------------------------------- #
# clean_asset: CRSP path
# --------------------------------------------------------------------------- #
def test_clean_crsp_basic_alignment():
    n = 50
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.01, n)
    res = clean.clean_asset(_dates(n), ret, kind="crsp_ret", source="crsp")
    # crsp_ret: one return per date -> same length out (no NaNs here).
    assert res["log_returns"].shape == (n,)
    assert res["dates"].shape == (n,)
    assert np.allclose(res["log_returns"], np.log1p(ret))
    assert res["report"]["n_obs"] == n


def test_clean_drops_nan_ret_not_fills():
    ret = np.array([0.01, np.nan, -0.02, 0.005])
    res = clean.clean_asset(_dates(4), ret, kind="crsp_ret", source="crsp")
    assert res["log_returns"].shape == (3,)          # nan dropped, NOT filled
    assert res["report"]["n_nan_dropped"] == 1


def test_clean_dedup_dates():
    d = np.array(["2020-01-01", "2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    ret = np.array([0.01, 0.02, 0.03])
    res = clean.clean_asset(d, ret, kind="crsp_ret", source="crsp")
    assert res["report"]["n_dup_dates"] == 1
    assert res["log_returns"].shape == (2,)


# --------------------------------------------------------------------------- #
# Tail preservation: a genuine extreme is FLAGGED but KEPT
# --------------------------------------------------------------------------- #
def test_extreme_flagged_but_preserved():
    n = 100
    rng = np.random.default_rng(1)
    ret = rng.normal(0, 0.01, n)
    ret[50] = -0.30  # a real -30% crash day (|log|~0.357 > 0.5? no). use bigger
    ret[50] = -0.55  # |log1p(-0.55)|=0.799 > 0.5 -> flagged
    res = clean.clean_asset(_dates(n), ret, kind="crsp_ret", source="crsp",
                            threshold=config.EXTREME_LOGRET_THRESHOLD)
    assert res["report"]["n_flagged"] == 1
    # The extreme is still in the series (not winsorized/dropped).
    assert res["log_returns"].shape == (n,)
    assert res["log_returns"].min() < -0.5


def test_glitch_reason_includes_bad_prc():
    n = 10
    ret = np.zeros(n)
    ret[5] = 0.8  # extreme
    prc = np.full(n, 100.0)
    prc[5] = -1.0  # bad price on the extreme day
    res = clean.clean_asset(_dates(n), ret, kind="crsp_ret", source="crsp",
                            aux={"prc": prc}, threshold=0.5)
    assert res["report"]["n_flagged"] == 1
    assert "bad prc" in res["flagged"][0]["reason"]


# --------------------------------------------------------------------------- #
# clean_asset: level path (FX/crypto)
# --------------------------------------------------------------------------- #
def test_clean_level_path():
    n = 20
    lv = 100 * np.cumprod(1 + np.full(n, 0.001))
    res = clean.clean_asset(_dates(n), lv, kind="level", source="wrds_fx")
    assert res["log_returns"].shape == (n - 1,)      # diff drops one
    assert res["dates"].shape == (n - 1,)
    assert np.all(np.isfinite(res["log_returns"]))


def test_report_keys_present():
    res = clean.clean_asset(_dates(30), np.random.default_rng(2).normal(0, 0.01, 30),
                            kind="crsp_ret", source="crsp")
    for k in ("asset", "source", "n_obs", "date_start", "date_end", "n_flagged",
              "excess_kurtosis", "ann_vol", "min_log_ret", "max_log_ret"):
        assert k in res["report"]
