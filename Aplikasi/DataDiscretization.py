# DataDiscretization.py
# Diskretisasi Student Well-Being berbasis Z-score (relatif terhadap cohort).
# Anti-NaN: Z-score & label selalu terisi (default 'Menengah' bila nilai kosong).
from typing import List, Tuple
import numpy as np
import pandas as pd

ASPECTS = ["Education", "Financial", "Physical", "Psychological", "Relational"]
LABELS = ("Tidak Sejahtera", "Menengah", "Sejahtera")

# ---------- helpers ----------
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _find_item_cols(df: pd.DataFrame, aspect: str) -> List[str]:
    pref = f"{aspect}_"
    return [c for c in df.columns if str(c).startswith(pref)]

def _ensure_aspect_means(df: pd.DataFrame, aspects: List[str]) -> pd.DataFrame:
    """
    Pastikan kolom mean per aspek ada. Jika tidak, hitung dari item <Aspect>_1..6.
    """
    out = df.copy()
    for a in aspects:
        if a not in out.columns:
            items = _find_item_cols(out, a)
            if len(items) >= 2:
                out[a] = out[items].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            else:
                out[a] = np.nan  # tidak cukup item → nanti dinetralkan saat Z
    return out

def _ensure_aspect_z(df: pd.DataFrame, aspects: List[str]) -> pd.DataFrame:
    """
    Pastikan kolom Z per aspek ada. Jika tidak, hitung dari mean aspek.
    Anti-NaN: x yang NaN diisi μ; jika σ=0/invalid → Z=0.
    """
    out = _ensure_aspect_means(df, aspects).copy()
    for a in aspects:
        zc = f"{a}_Z"
        if zc not in out.columns:
            x = _num(out[a])
            mu = float(np.nanmean(x))
            sd = float(np.nanstd(x, ddof=0))
            # isikan NaN ke mean agar Z tidak NaN
            if np.isfinite(mu):
                x_filled = x.fillna(mu)
            else:
                # kalau mean pun tidak finite (semua NaN), pakai 0 agar stabil
                x_filled = x.fillna(0.0)
                mu = 0.0
            if not np.isfinite(sd) or sd == 0:
                out[zc] = 0.0
            else:
                out[zc] = (x_filled - mu) / sd
    return out

def _label_from_value(v: float, low_thr: float, high_thr: float) -> str:
    # Selalu kembalikan string label (anti-NaN)
    try:
        x = float(v)
    except Exception:
        return LABELS[1]
    if not np.isfinite(x):
        return LABELS[1]
    if x <= low_thr:  return LABELS[0]
    if x >= high_thr: return LABELS[2]
    return LABELS[1]

# ---------- API utama ----------
def discretize_pipeline_z(
    df: pd.DataFrame,
    aspects: List[str] = ASPECTS,
    low_thr: float = -0.5,
    high_thr: float = 0.5,
) -> Tuple[pd.DataFrame, str]:
    """
    1) Pastikan ada kolom Z per aspek (atau hitung dari mean aspek/item).
    2) Buat label per-aspek dari Z.
    3) WB_Index_Z = rata-rata 5 Z (anti-NaN).
    4) WB_Label dari WB_Index_Z.

    Output kolom baru:
      - Label_<Aspect> untuk tiap aspek
      - WB_Index_Z, WB_Label
    """
    work = _ensure_aspect_z(df.copy(), aspects)

    # label per-aspek
    for a in aspects:
        zc = f"{a}_Z"
        if zc not in work.columns:
            # safety (harusnya sudah dibuat di _ensure_aspect_z)
            work[zc] = 0.0
        work[f"Label_{a}"] = work[zc].apply(lambda v: _label_from_value(v, low_thr, high_thr))

    # index keseluruhan (anti-NaN → isi yang hilang dengan 0 sebelum rata-rata)
    zcols = [f"{a}_Z" for a in aspects]
    Z = work[zcols].copy()
    Z = Z.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    work["WB_Index_Z"] = Z.mean(axis=1)
    work["WB_Label"] = work["WB_Index_Z"].apply(lambda v: _label_from_value(v, low_thr, high_thr))

    log = (
        f"[Discretization|Z] aspects={aspects}, thr=({low_thr},{high_thr}); "
        f"Z ensured; labels per-aspek & WB_Label dibuat tanpa NaN."
    )
    return work, log
