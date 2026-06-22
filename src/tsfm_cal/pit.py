"""PIT computation from the native 9 deciles, plus the per-asset run loop.

Two PIT estimators (HANDOFF §9):
  * **discrete** — ``searchsorted(deciles, x) / 9`` (PRIMARY; matches Phase-1).
  * **interp**   — linear CDF interpolation across the 9 decile knots (smoother).

Only ``compute_pit_for_asset`` / ``run_all_assets`` touch the model object; the
two estimator helpers are pure numpy and CPU-unit-tested.
"""

from __future__ import annotations

import numpy as np

from . import config, io_utils

# Cumulative-probability level of each native decile channel (q10..q90).
_LEVELS = np.asarray(config.NATIVE_QUANTILES, dtype=float)  # [0.1 .. 0.9]


def pit_discrete(quantiles: np.ndarray, realized: float) -> float:
    """PIT via right-count of crossed deciles, divided by 9.

    ``quantiles`` is the 9-vector ``[q10..q90]`` (already sorted). Returns a value
    in ``{0, 1/9, ..., 1}``. This is the validated Phase-1 estimator.
    """
    q = np.sort(np.asarray(quantiles, dtype=float))
    return float(np.searchsorted(q, realized)) / len(q)


def pit_interp(quantiles: np.ndarray, realized: float) -> float:
    """PIT via linear interpolation of the CDF across the 9 decile knots.

    Inside ``[q10, q90]`` this returns a smooth value in ``[0.1, 0.9]``. Outside,
    ``np.interp`` clamps to the edge levels (0.1 / 0.9) — we do NOT extrapolate
    into the tails (extrapolation there would import a tail assumption into the
    exact region under study; HANDOFF §3.1). Tail behaviour is therefore read off
    the discrete estimator's 0 and 1 buckets instead.
    """
    q = np.sort(np.asarray(quantiles, dtype=float))
    return float(np.interp(realized, q, _LEVELS))


def compute_pit_for_asset(
    model,
    rets: np.ndarray,
    context: int = config.CONTEXT,
    progress_every: int = 500,
    label: str = "",
):
    """Roll a 1-step forecast across ``rets`` and capture full per-step output.

    Returns a dict with arrays aligned by forecast step ``t`` (t in
    ``[context, len(rets)-1)``):
      * ``pit``        — discrete PIT, shape (T,)
      * ``pit_interp`` — interpolated PIT, shape (T,)
      * ``quantiles``  — the 9 deciles per step, shape (T, 9)
      * ``realized``   — realized return per step, shape (T,)
      * ``idx``        — the index into ``rets`` of each realized value, shape (T,)

    Saving the full quantile matrix + realized means every downstream diagnostic
    and figure is recomputed offline without re-running the model.
    """
    rets = np.asarray(rets, dtype=float).flatten()
    n = len(rets)
    pit, pit_i, qmat, real, idx = [], [], [], [], []

    for t in range(context, n - 1):
        ctx = rets[t - context:t]
        realized = rets[t]
        _, q = model.forecast(horizon=1, inputs=[ctx])
        q_vals = np.asarray(q[0, 0, 1:], dtype=float)  # channels 1..9 = q10..q90
        pit.append(pit_discrete(q_vals, realized))
        pit_i.append(pit_interp(q_vals, realized))
        qmat.append(q_vals)
        real.append(realized)
        idx.append(t)
        if progress_every and (t % progress_every == 0):
            print(f"    [{label}] t={t}/{n - 1}")

    return dict(
        pit=np.asarray(pit),
        pit_interp=np.asarray(pit_i),
        quantiles=np.asarray(qmat),
        realized=np.asarray(real),
        idx=np.asarray(idx),
    )


def run_all_assets(
    model,
    run_dir,
    tickers: dict[str, str] | None = None,
    start: str = config.START,
    end: str = config.END,
    kind: str = config.RETURNS_TYPE,
    context: int = config.CONTEXT,
):
    """Loop the asset universe, computing + **saving each npz inside the loop**.

    Returns ``{asset_name: result_dict}`` for in-session convenience, but the
    authoritative artifacts are the per-asset npz files written as we go — so a
    Kaggle session loss never discards a completed asset (HANDOFF §8).
    """
    from . import data

    tickers = tickers or config.TICKERS
    results: dict[str, dict] = {}

    for ticker, name in tickers.items():
        print(f"Running {name} ({ticker})...")
        try:
            dates, rets = data.download_returns(ticker, start, end, kind)
        except Exception as e:  # noqa: BLE001 — keep the loop alive on one bad fetch
            print(f"  ! download failed for {ticker}: {e}")
            continue

        if len(rets) < context + 50:
            print(f"  ! skipping {ticker}: only {len(rets)} returns (< context+50)")
            continue

        res = compute_pit_for_asset(model, rets, context=context, label=name)
        # Align dates to forecast steps (dates of the realized values).
        res_dates = dates[res["idx"]] if len(dates) == len(rets) else None

        io_utils.save_pit_npz(
            run_dir,
            name,
            pit=res["pit"],
            pit_interp=res["pit_interp"],
            quantiles=res["quantiles"],
            realized=res["realized"],
            dates=res_dates,
        )
        results[name] = res
        print(f"  saved {name}: {len(res['pit'])} PIT values")

    return results
