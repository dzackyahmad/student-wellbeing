# DataCleaning.py
# Cleaning: missing + outlier
# Outlier detector default = Z-score (otomatis), dengan 6 mode penanganan.

from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd

# -------- util kecil --------
def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def _coerce_numeric_like(df: pd.DataFrame, thresh: float = 0.7) -> pd.DataFrame:
    """Kolom non-numerik yang > thresh bisa diparse ke angka → konversi."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        parsed = pd.to_numeric(out[c], errors="coerce")
        # jika mayoritas berhasil → paksa jadi numerik
        if parsed.notna().mean() >= thresh:
            out[c] = parsed
    return out

def _normalize_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Seragamkan nilai kosong (string kosong, 'NA', 'null', '-') jadi NaN."""
    return df.replace(
        to_replace=[r'^\s*$', r'^\-$', r'^(na|n/a|NA|N/A|null|NULL)$'],
        value=np.nan,
        regex=True,
    )

# -------- missing handler --------
def _handle_missing(
    df: pd.DataFrame,
    method: str = "median",
    fill_value: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    work = _normalize_nans(df.copy())
    work = _coerce_numeric_like(work)

    log = f"[Missing] method={method}"
    if method == "drop":
        before = len(work)
        work = work.dropna()
        log += f" | drop {before - len(work)} baris"
        return work, log

    if method == "mean":
        # numerik → mean; non-numerik → mode
        means = work.mean(numeric_only=True)
        for c in means.index:
            work[c] = work[c].fillna(means[c])
        cats = [c for c in work.columns if c not in means.index]
        for c in cats:
            try:
                m = work[c].mode(dropna=True).iloc[0]
            except Exception:
                m = "Unknown"
            work[c] = work[c].fillna(m)
        log += " | numeric→mean, categorical→mode"
        return work, log

    if method == "median":
        # numerik → median; non-numerik → mode
        meds = work.median(numeric_only=True)
        for c in meds.index:
            work[c] = work[c].fillna(meds[c])
        cats = [c for c in work.columns if c not in meds.index]
        for c in cats:
            try:
                m = work[c].mode(dropna=True).iloc[0]
            except Exception:
                m = "Unknown"
            work[c] = work[c].fillna(m)
        log += " | numeric→median, categorical→mode"
        return work, log

    if method == "mode":
        try:
            work = work.fillna(work.mode().iloc[0])
        except Exception:
            # jika ada kolom all-NaN
            for c in work.columns:
                m = work[c].mode()
                if not m.empty:
                    work[c] = work[c].fillna(m.iloc[0])
        log += " | all→mode"
        return work, log

    if method == "constant":
        try:
            v = pd.to_numeric(pd.Series([fill_value]), errors="coerce").iloc[0]
            v = v if pd.notna(v) else fill_value
        except Exception:
            v = fill_value
        work = work.fillna(v)
        log += f" | all→constant({v})"
        return work, log

    # default
    return work, log + " | dilewati"

# -------- outlier: deteksi Z-score + 6 mode aksi --------
def _z_stats(col: pd.Series) -> Tuple[float, float]:
    x = pd.to_numeric(col, errors="coerce")
    mu = float(np.nanmean(x.values))
    sd = float(np.nanstd(x.values, ddof=0))
    return mu, sd

def _detect_outlier_z(col: pd.Series, z_thr: float) -> Tuple[pd.Series, float, float]:
    """kembalikan (mask, lo, hi)"""
    x = pd.to_numeric(col, errors="coerce")
    mu, sd = _z_stats(x)
    if not np.isfinite(sd) or sd == 0:
        mask = pd.Series(False, index=col.index)
        return mask, np.nan, np.nan
    z = (x - mu) / sd
    mask = z.abs() > z_thr
    lo, hi = mu - z_thr * sd, mu + z_thr * sd
    return mask, lo, hi

def _apply_outlier_action(
    df: pd.DataFrame,
    masks: Dict[str, pd.Series],
    bounds: Dict[str, Tuple[float, float]],
    mode: str = "mark",
    fill_value_outlier: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    mode:
      - clip            : potong ke [lo, hi]
      - remove          : drop baris yang outlier di kolom manapun
      - mark            : tambah kolom *_is_outlier = True/False
      - set_nan         : nilai outlier → NaN
      - impute_median   : set_nan lalu isi median
      - impute_constant : set_nan lalu isi konstanta
    """
    work = df.copy()
    mode = (mode or "mark").lower()
    union_mask = pd.Series(False, index=work.index)

    for c, mask in masks.items():
        if mask is None or mask.sum() == 0:
            # tidak ada outlier di kolom ini
            if mode == "mark":
                work[c + "_is_outlier"] = False
            continue

        lo, hi = bounds.get(c, (np.nan, np.nan))
        col = pd.to_numeric(work[c], errors="coerce")

        if mode == "clip":
            if np.isfinite(lo) and np.isfinite(hi):
                work[c] = col.clip(lo, hi)
        elif mode == "remove":
            union_mask = union_mask | mask
        elif mode == "mark":
            work[c + "_is_outlier"] = mask
        elif mode == "set_nan":
            work.loc[mask, c] = np.nan
        elif mode == "impute_median":
            work.loc[mask, c] = np.nan
            med = work[c].median(skipna=True)
            work[c] = work[c].fillna(med)
        elif mode == "impute_constant":
            work.loc[mask, c] = np.nan
            try:
                k = pd.to_numeric(pd.Series([fill_value_outlier]), errors="coerce").iloc[0]
                k = k if pd.notna(k) else fill_value_outlier
            except Exception:
                k = fill_value_outlier
            work[c] = work[c].fillna(k)

    if mode == "remove":
        dropped = int(union_mask.sum())
        work = work.loc[~union_mask].copy()
        return work, f"[Outlier] remove: drop {dropped} baris (union semua kolom)"

    return work, f"[Outlier] mode={mode} diterapkan"

def clean_data(
    df: pd.DataFrame,
    missing_method: str = "median",
    fill_value: Optional[str] = None,
    # Outlier: selalu deteksi Z-score (auto). IQR opsional bisa ditambah nanti.
    outlier_method: str = "zscore",
    z_threshold: float = 3.0,
    outlier_mode: str = "mark",                 # clip/remove/mark/set_nan/impute_median/impute_constant
    fill_value_outlier: Optional[str] = None,   # untuk impute_constant
    target_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Pipeline:
      1) Missing handling (drop/mean/median/mode/constant)
      2) Outlier detection (Z-score otomatis) + action

    Params penting untuk outlier:
      - z_threshold: ambang |Z|
      - outlier_mode: aksi sesudah deteksi (lihat di atas)
      - target_cols: jika None → semua kolom numerik
    """
    if df is None or len(df) == 0:
        return df, "Data kosong – cleaning dilewati."

    # --- Step 1: Missing ---
    work, log_miss = _handle_missing(df, method=missing_method, fill_value=fill_value)

    # --- Step 2: Outlier (Z-score) ---
    if outlier_method != "zscore":
        # tetap izinkan override, tapi default kita Z-score
        pass

    cols_apply = target_cols if target_cols else _numeric_cols(work)
    masks: Dict[str, pd.Series] = {}
    bounds: Dict[str, Tuple[float, float]] = {}
    counts: Dict[str, int] = {}

    for c in cols_apply:
        mask, lo, hi = _detect_outlier_z(work[c], z_threshold)
        masks[c] = mask
        bounds[c] = (lo, hi)
        counts[c] = int(mask.sum())

    work2, log_out = _apply_outlier_action(
        work, masks, bounds,
        mode=outlier_mode,
        fill_value_outlier=fill_value_outlier,
    )

    # ringkasan per kolom
    summary = ", ".join([f"{c}:{counts[c]}" for c in cols_apply]) if cols_apply else "-"
    log = f"{log_miss}\n[Outlier] detector=Z-score | |Z|>{z_threshold} | kolom={cols_apply}\n[Outlier] jumlah terdeteksi per kolom: {summary}\n{log_out}"

    return work2, log
