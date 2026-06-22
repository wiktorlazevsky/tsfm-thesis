"""WRDS / CRSP data pulls. LOCAL ONLY (needs WRDS credentials; not on Kaggle).

Pulls raw source series and hands them to ``clean.clean_asset``. The orchestrator
``build_clean_dataset`` loops the asset universe, cleans, and writes the canonical
``outputs/data/clean/`` tree + a quality report + a manifest — which is then
uploaded to Kaggle as a Dataset.

``wrds`` is imported lazily so the rest of the package stays importable on a
machine without it. Exact CRSP/FRB table + column names can vary by WRDS
subscription; the defaults below are the common ones and are recorded in the
manifest. Override via the function kwargs if your library names differ.

Usage (local)::

    from tsfm_cal import wrds_data
    db = wrds_data.connect()                       # prompts for WRDS login
    wrds_data.build_clean_dataset(db)              # writes outputs/data/clean/
    db.close()
"""

from __future__ import annotations

import datetime as _dt

import numpy as np

from . import clean, config, io_utils


def connect(username: str | None = None):
    """Open a WRDS connection.

    ``username`` defaults to the ``WRDS_USERNAME`` env var if set, else the wrds
    library prompts. Setting it (and a ``~/.pgpass`` via
    ``wrds.Connection(wrds_username=...).create_pgpass_file()`` once) avoids the
    repeated login prompts.
    """
    import os

    import wrds  # lazy

    username = username or os.environ.get("WRDS_USERNAME")
    return wrds.Connection(wrds_username=username) if username else wrds.Connection()


def discover_fx_tables(db):
    """Print candidate FX/exchange-rate libraries + tables in your WRDS account.

    The FRB FX table name varies by subscription. Run this once, eyeball the
    output, then pass the right ``table``/``rate_col`` to ``wrds_fx_levels`` (or
    set them as defaults). Looks at frb-ish libraries and any table whose name
    hints at exchange rates.
    """
    libs = db.list_libraries()
    hits = [l for l in libs if any(k in l.lower() for k in ("frb", "fx", "exch", "fed"))]
    print("Candidate libraries:", hits or "(none — try db.list_libraries())")
    for lib in hits:
        try:
            tables = db.list_tables(library=lib)
        except Exception as e:  # noqa: BLE001
            print(f"  {lib}: <cannot list: {e}>")
            continue
        fx = [t for t in tables if any(k in t.lower() for k in ("fx", "exch", "rate", "h10"))]
        print(f"  {lib}: {fx or tables[:20]}")
    print("\nInspect a table's columns with: db.describe_table(library=..., table=...)")


# --------------------------------------------------------------------------- #
# CRSP daily total return (equities / ETFs)                                   #
# --------------------------------------------------------------------------- #
def crsp_permnos_for_ticker(db, ticker: str, names_table: str = "crsp.stocknames") -> list[int]:
    """Resolve PERMNO(s) historically associated with a ticker."""
    sql = f"SELECT DISTINCT permno FROM {names_table} WHERE ticker = %(t)s"
    df = db.raw_sql(sql, params={"t": ticker})
    return sorted(int(p) for p in df["permno"].dropna().unique())


def crsp_daily(
    db,
    ticker: str,
    start: str = config.START,
    end: str = config.END,
    dsf_table: str = "crsp.dsf",
    delist_table: str = "crsp.dsedelist",
    names_table: str = "crsp.stocknames",
):
    """Pull CRSP daily total return for one ticker.

    Returns a dict: dates, ret (holding-period, delisting-merged), prc, vol,
    permno(s). ``ret`` already accounts for splits/dividends; delisting returns
    are merged in on the delist date (compounded with any same-day ret).
    """
    permnos = crsp_permnos_for_ticker(db, ticker, names_table)
    if not permnos:
        raise ValueError(f"No PERMNO found for ticker {ticker!r} in {names_table}")
    plist = ",".join(str(p) for p in permnos)

    df = db.raw_sql(
        f"""SELECT permno, date, ret, prc, vol
            FROM {dsf_table}
            WHERE permno IN ({plist}) AND date BETWEEN %(s)s AND %(e)s
            ORDER BY date""",
        params={"s": start, "e": end},
    )
    # Delisting returns (rare for ETFs, but correct to merge).
    try:
        dl = db.raw_sql(
            f"""SELECT permno, dlstdt AS date, dlret
                FROM {delist_table}
                WHERE permno IN ({plist}) AND dlstdt BETWEEN %(s)s AND %(e)s""",
            params={"s": start, "e": end},
        )
    except Exception:  # noqa: BLE001 — table may be absent / named differently
        dl = None

    import pandas as pd

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if dl is not None and not dl.empty:
        dl["date"] = pd.to_datetime(dl["date"])
        df = df.merge(dl[["permno", "date", "dlret"]], on=["permno", "date"], how="left")
        # Compound delisting return with same-day ret: (1+ret)(1+dlret)-1.
        has_dl = df["dlret"].notna()
        df.loc[has_dl, "ret"] = (
            (1 + df.loc[has_dl, "ret"].fillna(0.0)) * (1 + df.loc[has_dl, "dlret"]) - 1
        )

    # If a ticker mapped to multiple permnos, prefer the one with the most obs.
    if df["permno"].nunique() > 1:
        best = df["permno"].value_counts().idxmax()
        df = df[df["permno"] == best]

    return {
        "dates": df["date"].values.astype("datetime64[ns]"),
        "ret": df["ret"].values.astype(float),
        "prc": np.abs(df["prc"].values.astype(float)),  # CRSP encodes bid/ask avg as negative
        "vol": df["vol"].values.astype(float),
        "permnos": permnos,
    }


# --------------------------------------------------------------------------- #
# WRDS FX (FRB H.10 daily exchange rate)                                       #
# --------------------------------------------------------------------------- #
def wrds_fx_levels(
    db,
    series_id: str = "EUR",
    start: str = config.START,
    end: str = config.END,
    table: str = "frb.fx_daily",
    date_col: str = "date",
    rate_col: str | None = None,
):
    """Pull a daily FX level series from the WRDS FRB library.

    Defaults target ``frb.fx_daily`` column ``dexuseu`` = US$ per Euro (the
    standard EUR/USD quote, ~1.0-1.6 over the sample; FRB H.10, euro starts
    1999-01-04). ``dexeuus`` is the inverse. Pass ``table``/``rate_col`` to use a
    different pair. The exact table/column are recorded in the manifest.
    """
    col = rate_col or "dexuseu"
    df = db.raw_sql(
        f"SELECT {date_col} AS date, {col} AS level FROM {table} "
        f"WHERE {date_col} BETWEEN %(s)s AND %(e)s ORDER BY {date_col}",
        params={"s": start, "e": end},
    )
    import pandas as pd

    df = df.dropna(subset=["level"])
    return {
        "dates": pd.to_datetime(df["date"]).values.astype("datetime64[ns]"),
        "level": df["level"].values.astype(float),
        "table": table,
        "column": col,
    }


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #
def build_clean_dataset(
    db=None,
    assets: dict | None = None,
    start: str = config.START,
    end: str = config.END,
):
    """Pull + clean every asset, writing the canonical clean-data tree.

    ``db`` may be None if the universe contains only crypto (yfinance). CRSP/FX
    assets require an open WRDS connection.
    """
    assets = assets or config.ASSETS
    reports, manifest, flagged_all, failed = [], {}, {}, {}

    for key, meta in assets.items():
        source, sid = meta["source"], meta["source_id"]
        print(f"[{key}] source={source} id={sid}")
        try:
            if source == "crsp":
                raw = crsp_daily(db, sid, start, end)
                res = clean.clean_asset(
                    raw["dates"], raw["ret"], kind="crsp_ret", source="crsp",
                    aux={"prc": raw["prc"], "vol": raw["vol"]},
                )
                manifest[key] = {"source": "crsp", "permnos": raw["permnos"], "ticker": sid}

            elif source == "wrds_fx":
                raw = wrds_fx_levels(db, sid, start, end)
                res = clean.clean_asset(
                    raw["dates"], raw["level"], kind="level", source="wrds_fx",
                )
                manifest[key] = {"source": "wrds_fx", "table": raw["table"], "column": raw["column"]}

            elif source == "crypto":
                from . import data  # yfinance fallback for BTC (documented exception)
                prices = data.download_prices(sid, start, end)
                res = clean.clean_asset(
                    np.asarray(prices.index, dtype="datetime64[ns]"),
                    np.asarray(prices.values, dtype=float),
                    kind="level", source="crypto(yfinance)",
                )
                manifest[key] = {"source": "crypto", "provider": "yfinance", "ticker": sid}

            else:
                print(f"  ! unknown source {source!r}, skipping")
                continue
        except Exception as e:  # noqa: BLE001 — one bad asset shouldn't lose the rest
            print(f"  ! FAILED {key}: {type(e).__name__}: {e}")
            failed[key] = f"{type(e).__name__}: {e}"
            continue

        res["report"]["asset"] = key
        reports.append(res["report"])
        flagged_all[key] = res["flagged"]
        io_utils.save_clean_npz(
            key,
            dates=res["dates"],
            log_returns=res["log_returns"],
            simple_returns=res["simple_returns"],
            source=res["report"]["source"],
        )
        print(f"  saved {key}: n={res['report']['n_obs']} "
              f"kurt={res['report']['excess_kurtosis']:.1f} flagged={res['report']['n_flagged']}")

    if failed:
        print(f"\n!! {len(failed)} asset(s) failed: {list(failed)} — see 'failed' in manifest")
        manifest["_failed"] = failed

    # Reports + manifest + flagged log.
    clean_dir = io_utils.data_clean_dir()
    io_utils.save_table_csv(clean_dir.parent / "data_quality_report.csv", reports)
    io_utils.save_json(clean_dir.parent / "manifest.json", {
        "built": _dt.datetime.now().isoformat(timespec="seconds"),
        "window": {"start": start, "end": end},
        "assets": manifest,
    })
    io_utils.save_json(clean_dir.parent / "flagged.json", flagged_all)
    print(f"\nWrote clean data + report to {clean_dir.parent}")
    return reports
