# tsfm-thesis

Distributional calibration & arbitrary tail-quantile extraction from **TimesFM 2.5**
on financial returns. MSc thesis (Wiktor Lazewski, Maastricht University).

**Phase 1 (this code):** reproducible zero-shot PIT + coverage audit across 7
assets, log returns, `force_flip_invariance=False` (asymmetry preserved), with
every artifact saved. See `HANDOFF_1_JUNE_22_2026.md` in the parent project for
full context.

## Layout

```
src/tsfm_cal/
  config.py     tickers, dates, CONTEXT, NATIVE_QUANTILES, explicit ForecastConfig flags
  data.py       yfinance download + cache, log/simple returns (.flatten, auto_adjust)
  pit.py        PIT (discrete + CDF-interp), per-asset roll, saves npz IN-loop
  eval.py       KS, coverage, tail mass, chi2, conditional KS, VaR/ES backtests
  plot.py       PIT histogram, 2x4 grid, KS-vs-kurtosis
  io_utils.py   npz/csv/json save+load, run folders, Kaggle/local path autodetect
  heads.py      Phase-3 STUB (IQN / skew-t heads)
  finetune.py   Phase-3 STUB (saving convention for adapters/heads)
notebooks/01_zeroshot_pit.ipynb   thin Kaggle driver
tests/test_eval.py                CPU unit tests
```

## Local (CPU, no model)

```bash
pip install -e ".[dev]"
pytest                       # PIT math + diagnostics + VaR backtest sanity
```

## Kaggle (T4) — run the audit

First cells of `notebooks/01_zeroshot_pit.ipynb` (also paste-able):

```python
!pip install "git+https://github.com/google-research/timesfm.git#egg=timesfm[torch]" -q
!pip install yfinance -q
!pip install -e "git+https://github.com/<user>/tsfm-thesis.git#egg=tsfm_cal" -q
# >>> RESTART KERNEL before importing timesfm <<<
```

```python
import torch, timesfm
from tsfm_cal import config, data, pit, eval, plot, io_utils

torch.set_float32_matmul_precision("high")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(timesfm.ForecastConfig(**config.FORECAST_FLAGS))   # flip OFF, explicit

rid = config.run_id(flip=False)
run_dir = io_utils.zeroshot_run_dir(rid)
io_utils.save_json(run_dir / "config.json", {
    "run_id": rid, "flags": config.FORECAST_FLAGS, "returns_type": config.RETURNS_TYPE,
    "start": config.START, "end": config.END, "context": config.CONTEXT,
})

results = pit.run_all_assets(model, run_dir)              # saves npz per asset in-loop
rows = eval.summarize_run(run_dir, from_disk=True)
io_utils.save_table_csv(run_dir / "tables" / "pit_summary.csv", rows)

pits = {r["asset"]: io_utils.load_pit_npz(run_dir, r["asset"])["pit"] for r in rows}
ks = {r["asset"]: r["ks"] for r in rows}
plot.pit_grid(pits, ks, save_path=run_dir / "figures" / "pit_grid.png")
plot.ks_vs_kurtosis(rows, save_path=run_dir / "figures" / "ks_vs_kurtosis.png")
```

`outputs/` resolves to `/kaggle/working/outputs` on Kaggle (downloadable / push as
a Kaggle Dataset) and to `./outputs` locally.

### Conventions (locked)

- **Log returns** everywhere (the old simple-return KS table is superseded).
- 9 native deciles for zero-shot PIT; **discrete** PIT is primary, CDF-interp reported alongside.
- `ForecastConfig` flags set **explicitly** (`force_flip_invariance=False`,
  `infer_is_positive=False`, `fix_quantile_crossing=True`,
  `use_continuous_quantile_head=True`, `normalize_inputs=True`) — written into every `config.json`.
- Each asset's npz is saved **inside** the loop (Kaggle-session-loss safe).
- To run the symmetrized comparison on SPY: `config.FORECAST_FLAGS_FLIP` + `config.run_id(flip=True)`.
