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
  config.py     ASSETS registry + sources, dates, CONTEXT, RISK_QUANTILES, explicit ForecastConfig flags
  data.py       load_returns (clean npz, Kaggle/local autodetect) + yfinance fallback
  wrds_data.py  LOCAL: CRSP daily total-return + WRDS FX pulls -> clean dataset (lazy `wrds`)
  clean.py      log returns, no-forward-fill, tail-preserving error screen, quality report
  pit.py        zero-shot PIT (discrete + CDF-interp), per-asset roll, saves npz IN-loop
  eval.py       KS, coverage, tail mass, chi2, conditional KS, VaR/ES backtests, per-τ reliability
  plot.py       PIT histogram, 2x4 grid, KS-vs-kurtosis
  io_utils.py   npz/csv/json save+load, run/clean/finetune dirs, Kaggle/local autodetect
  backbone.py   Phase 3: load native (torch_compile=False), freeze, cache embeddings
  heads.py      Phase 3: SkewStudentTHead (Hansen) + IQNHead, arbitrary-τ quantiles/PIT/VaR/ES
  finetune.py   Phase 3: head-only training (NLL / sampled-τ pinball) on cached embeddings
notebooks/
  01_zeroshot_pit.ipynb     Phase 1 zero-shot PIT audit (Kaggle)
  02_cache_embeddings.ipynb Phase 3 backbone embedding cache (Kaggle, expensive once)
  03_train_heads.ipynb      Phase 3 train skew-t + IQN, tail VaR/ES eval (Kaggle)
tests/  test_eval.py  test_clean.py  test_heads.py   CPU unit tests
```

## Data prep (LOCAL — WRDS/CRSP, run once)

Yahoo adjusted close is weak for academic work (the EURUSD `=X` series produced a
kurtosis≈105 artifact). Canonical returns come from CRSP/WRDS:

```python
from tsfm_cal import wrds_data
db = wrds_data.connect()              # prompts for WRDS login (local only)
wrds_data.build_clean_dataset(db)     # -> outputs/data/clean/<ASSET>.npz + report + manifest
db.close()
```

- CRSP daily total return (`ret`, split/div/delisting-correct) for SPY/TLT/GLD/USO/EEM.
- WRDS FRB H.10 daily FX for EUR/USD. BTC from a documented source (yfinance), flagged.
- Inspect `outputs/data/data_quality_report.csv` (esp. EURUSD kurtosis, now sane) and
  `flagged.json` (extreme returns kept for manual inspection — fat tails are the study object).

### Ship to Kaggle (one zip)

```python
from tsfm_cal import io_utils
io_utils.make_clean_bundle()     # -> outputs/data/tsfm-clean-data.zip (clean/*.npz + report + manifest)
```

1. Kaggle → **Datasets → New Dataset** → upload `outputs/data/tsfm-clean-data.zip`
   (Kaggle auto-extracts it) → name it **`tsfm-clean-data`**.
2. In a notebook → **Add Data** → your `tsfm-clean-data` dataset.
3. `data.load_returns` finds it automatically via `io_utils.clean_dataset_dir` (handles both the
   extracted `clean/` folder and a still-zipped bundle). To refresh: rebuild the zip, then
   **Datasets → your dataset → New Version**.

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
# Replace YOURNAME with your GitHub username:
!pip install "tsfm_cal @ git+https://github.com/YOURNAME/tsfm-thesis.git" -q
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

## Phase 3 — head replacement (Kaggle T4)

The native head only emits `[mean, q10..q90]`; arbitrary tail quantiles (1/5/95/99%)
+ Expected Shortfall need a **replacement head on the frozen backbone**.

1. **`notebooks/02_cache_embeddings.ipynb`** — load TimesFM with `torch_compile=False`,
   freeze the backbone, cache the last-patch embedding + ReVIN stats + normalized 1-step
   target per rolling window (expensive, once per asset → `outputs/finetune/<ASSET>/emb_cache.npz`).
2. **`notebooks/03_train_heads.ipynb`** — train **Skew-Student-t** (Hansen 1994; NLL) and
   **IQN** (sampled-τ pinball) head-only on the caches, all 7 assets. Evaluate before/after
   PIT/KS, per-τ reliability, and VaR/ES at 1/5/95/99% (Kupiec, Christoffersen, FZ loss, ES±CI).

**τ is a query-time argument** — both heads emit any τ post-hoc (skew-t via analytic PPF, IQN
via its τ input); `RISK_QUANTILES` is only the eval grid. Deep tails (≤1%) are sample-limited —
report with the bootstrap ES bands. Heads share one API so `eval.py` treats them uniformly:
`heads.head_quantiles / head_pit / head_var / head_es`.
