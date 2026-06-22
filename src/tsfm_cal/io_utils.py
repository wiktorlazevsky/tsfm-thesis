"""Artifact I/O: run-folder layout, npz/csv/json save+load, Kaggle/local paths.

Folder tree (per HANDOFF §8)::

    outputs/
      zeroshot/<run_id>/
        config.json
        pit/<ASSET>.npz
        tables/{pit_summary.csv, ks_table.json}
        figures/{pit_grid.png, ks_vs_kurtosis.png}
      finetune/<ASSET>/{adapter.pt, head.pt, pit_before.npz, pit_after.npz}

All saving for the zero-shot loop is keyed off a single run folder so re-running
``eval``/``plot`` from saved arrays never needs the model again.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def base_dir() -> Path:
    """Resolve the outputs base: ``/kaggle/working/outputs`` on Kaggle, else
    ``./outputs`` relative to the repo root (cwd when the notebook runs)."""
    if os.path.isdir("/kaggle/working"):
        return Path("/kaggle/working/outputs")
    # Allow override for local testing.
    env = os.environ.get("TSFM_CAL_OUTPUTS")
    if env:
        return Path(env)
    return Path("outputs")


def zeroshot_run_dir(run_id: str, create: bool = True) -> Path:
    """Return ``outputs/zeroshot/<run_id>/`` and create its subtree."""
    root = base_dir() / "zeroshot" / run_id
    if create:
        for sub in ("pit", "tables", "figures"):
            (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def finetune_dir(asset: str, create: bool = True) -> Path:
    """Return ``outputs/finetune/<asset>/`` (Phase 3 scaffold)."""
    root = base_dir() / "finetune" / _safe(asset)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def data_clean_dir(create: bool = True) -> Path:
    """Return ``outputs/data/clean/`` (canonical cleaned returns)."""
    root = base_dir() / "data" / "clean"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def make_clean_bundle(out_zip: str | os.PathLike | None = None) -> Path:
    """Zip the whole cleaned-data tree into ONE file for easy Kaggle upload.

    Produces ``outputs/data/tsfm-clean-data.zip`` containing::

        clean/<ASSET>.npz ...      (the 7 cleaned return series)
        data_quality_report.csv
        manifest.json
        flagged.json

    Upload this single zip as a Kaggle Dataset; Kaggle auto-extracts it, and the
    loader (``clean_dataset_dir``) finds the ``clean/`` folder automatically.
    """
    import zipfile

    data_dir = base_dir() / "data"
    out = Path(out_zip) if out_zip else data_dir / "tsfm-clean-data.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("clean",):
            for p in (data_dir / sub).glob("*.npz"):
                z.write(p, arcname=f"{sub}/{p.name}")
        for meta in ("data_quality_report.csv", "manifest.json", "flagged.json"):
            mp = data_dir / meta
            if mp.exists():
                z.write(mp, arcname=meta)
    return out


def clean_dataset_dir() -> Path | None:
    """Resolve the mounted Kaggle clean-data Dataset, if present.

    Returns the first dir under ``/kaggle/input/`` containing clean npz files
    (checking both the dir and its ``clean/`` subfolder). If a dataset shipped as
    a still-zipped bundle, it is extracted to ``/kaggle/working`` first. Returns
    None off Kaggle / when nothing is found.
    """
    from . import config

    inp = Path("/kaggle/input")
    candidates = [
        inp / config.KAGGLE_CLEAN_DATASET,
        inp / config.KAGGLE_CLEAN_DATASET / "clean",
    ]
    if inp.is_dir():
        for d in inp.iterdir():
            candidates += [d, d / "clean"]
    for c in candidates:
        if c.is_dir() and list(c.glob("*.npz")):
            return c

    # Fallback: a still-zipped bundle somewhere under /kaggle/input -> extract.
    if inp.is_dir():
        for zp in inp.glob("**/*.zip"):
            dest = Path("/kaggle/working") / "tsfm_clean_extracted"
            dest.mkdir(parents=True, exist_ok=True)
            import zipfile

            with zipfile.ZipFile(zp) as z:
                z.extractall(dest)
            for c in (dest / "clean", dest):
                if c.is_dir() and list(c.glob("*.npz")):
                    return c
    return None


# --------------------------------------------------------------------------- #
# Primitive savers / loaders                                                  #
# --------------------------------------------------------------------------- #
def save_json(path: str | os.PathLike, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | os.PathLike) -> dict:
    with open(path) as f:
        return json.load(f)


def save_pit_npz(
    run_dir: str | os.PathLike,
    asset: str,
    *,
    pit: np.ndarray,
    pit_interp: np.ndarray,
    quantiles: np.ndarray,
    realized: np.ndarray,
    dates: np.ndarray | None = None,
) -> Path:
    """Persist one asset's full PIT artifacts. Called *inside* the asset loop so
    a Kaggle session loss never destroys completed assets (HANDOFF §8)."""
    out = Path(run_dir) / "pit" / f"{_safe(asset)}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        pit=np.asarray(pit),
        pit_interp=np.asarray(pit_interp),
        quantiles=np.asarray(quantiles),
        realized=np.asarray(realized),
    )
    if dates is not None:
        payload["dates"] = np.asarray(dates)
    np.savez_compressed(out, **payload)
    return out


def load_pit_npz(run_dir: str | os.PathLike, asset: str) -> dict:
    npz = np.load(Path(run_dir) / "pit" / f"{_safe(asset)}.npz", allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def list_pit_assets(run_dir: str | os.PathLike) -> list[str]:
    pit_dir = Path(run_dir) / "pit"
    if not pit_dir.is_dir():
        return []
    return sorted(p.stem for p in pit_dir.glob("*.npz"))


def save_table_csv(path: str | os.PathLike, rows: list[dict]) -> None:
    """Write a list-of-dicts as CSV without requiring pandas at call sites that
    only have stdlib. (pandas is fine too — this just keeps the dep optional.)"""
    import csv

    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_clean_npz(
    asset: str,
    *,
    dates: np.ndarray,
    log_returns: np.ndarray,
    simple_returns: np.ndarray,
    source: str,
    rf: np.ndarray | None = None,
    clean_dir: str | os.PathLike | None = None,
) -> Path:
    """Persist one asset's canonical cleaned returns under outputs/data/clean/."""
    root = Path(clean_dir) if clean_dir is not None else data_clean_dir()
    out = root / f"{_safe(asset)}.npz"
    payload = dict(
        dates=np.asarray(dates),
        log_returns=np.asarray(log_returns, dtype=float),
        simple_returns=np.asarray(simple_returns, dtype=float),
        source=np.asarray(source),
    )
    if rf is not None:
        payload["rf"] = np.asarray(rf, dtype=float)
    np.savez_compressed(out, **payload)
    return out


def load_clean_npz(asset: str, clean_dir: str | os.PathLike | None = None) -> dict:
    """Load one asset's cleaned returns.

    Resolution order: explicit ``clean_dir`` -> mounted Kaggle Dataset
    (``clean_dataset_dir``) -> local ``outputs/data/clean/``.
    """
    if clean_dir is not None:
        root = Path(clean_dir)
    else:
        root = clean_dataset_dir() or data_clean_dir(create=False)
    npz = np.load(Path(root) / f"{_safe(asset)}.npz", allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def _safe(name: str) -> str:
    """Filesystem-safe asset token (e.g. 'EURUSD=X' -> 'EURUSD_X')."""
    return name.replace("=", "_").replace("/", "_").replace(" ", "_")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
