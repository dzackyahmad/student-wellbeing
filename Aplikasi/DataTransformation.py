#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess Kuesioner → Excel out (2–3 sheet, TANPA RAW):
- Bagan_2_Hasil : [Education, Financial, Physical, Psychological, Relational,
                   Education_Z, Financial_Z, Physical_Z, Psychological_Z, Relational_Z]
- Bagan_3_Discretized : per-butir *_kategori + per-mean *_MeanKategori
- (opsional) Log_Transform : catatan proses
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# ====== KONFIG ======
INPUT_FILE = "Kuesioner Kesejahteraan Mahasiswa Unpad (Jawaban).xlsx"
DOMAINS = ["Education", "Financial", "Physical", "Psychological", "Relational"]
EXCLUDE_CONTAINS = ["Semester", "Timestamp", "Nama", "Jenis Kelamin", "Jurusan"]  # agar tak ikut Likert
_LIKERT_VALUES = {1, 2, 3, 4, 5}
WRITE_LOG_SHEET = True  # set False kalau tak mau sheet log

# ====== UTIL TAMBAHAN ======
def _fix_timestamp_if_any(df: pd.DataFrame) -> pd.DataFrame:
    if "Timestamp" not in df.columns:
        return df
    s = pd.to_numeric(df["Timestamp"], errors="coerce")
    if s.notna().sum() == 0:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        return df
    med = float(s.dropna().median())
    unit = None
    if med > 1e17:      unit = "ns"
    elif med > 1e12:    unit = "ms"
    elif med > 1e9:     unit = "s"
    df["Timestamp"] = pd.to_datetime(s, unit=unit, errors="coerce") if unit else pd.to_datetime(df["Timestamp"], errors="coerce")
    return df

# ====== LOGIKA DATA TRANSFORMATION (AUTO) ======
def _z_series(x: pd.Series) -> pd.Series:
    mu = float(np.nanmean(x.values))
    sigma = float(np.nanstd(x.values, ddof=0))
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sigma

def _detect_likert_columns(
    df: pd.DataFrame,
    exclude_contains: Optional[List[str]] = None,
    min_share_in_set: float = 0.70,
    max_unique: int = 5,
) -> List[str]:
    exclude_contains = exclude_contains or ["Semester"]
    candidates: List[str] = []
    for col in df.columns:
        ser_num = pd.to_numeric(df[col], errors="coerce")
        non_na = ser_num.notna().sum()
        if non_na == 0:
            continue
        ser_int = ser_num.dropna().round().astype(int)
        share = (ser_int.isin(_LIKERT_VALUES).sum()) / max(1, non_na)
        if share >= min_share_in_set and ser_int.nunique() <= max_unique:
            if any(ex and ex.lower() in str(col).lower() for ex in exclude_contains):
                continue
            candidates.append(col)
    return [c for c in df.columns if c in set(candidates)]

def _group_domains_by_six(likert_cols: List[str], domains: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    start = 0
    for d in domains:
        groups[d] = likert_cols[start:start + 6]
        start += 6
    return groups

def aggregate_aspects_auto(
    df: pd.DataFrame,
    domains: Optional[List[str]] = None,
    exclude_contains: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], str]:
    if df is None or len(df) == 0:
        return df, {}, "Data kosong – agregasi domain dilewati."
    domains = domains or DOMAINS
    likert_cols = _detect_likert_columns(df, exclude_contains=exclude_contains)
    groups = _group_domains_by_six(likert_cols, domains)

    work = df.copy()
    logs = [f"[Deteksi Likert] total kandidat: {len(likert_cols)} kolom"]
    # Mean per domain
    for d in domains:
        cols = groups.get(d, [])
        cols = [c for c in cols if c in work.columns]  # filter aman
        if cols:
            work[d] = pd.to_numeric(work[cols], errors="coerce").mean(axis=1)
            logs.append(f"[Mean] {d}: mean dari {cols}")
        else:
            work[d] = np.nan
            logs.append(f"[Mean] {d}: kolom tidak cukup (butuh 6).")
    # Z per domain
    for d in domains:
        work[f"{d}_Z"] = _z_series(work[d])
        logs.append(f"[Z-score] {d}_Z dibuat dari {d}")
    logs.append(f"Kolom hasil: {domains + [f'{d}_Z' for d in domains]}")
    logs.append(f"Shape akhir: {work.shape[0]} x {work.shape[1]}")
    return work, groups, "\n".join(logs)

# ====== DISKRETISASI (opsional) ======
def _likert_to_category(val) -> Optional[str]:
    try:
        x = float(val)
    except Exception:
        return np.nan
    if 1.0 <= x <= 2.0: return "Rendah"
    if x == 3.0:       return "Sedang"
    if 4.0 <= x <= 5.0:return "Tinggi"
    return np.nan

def discretize_likert_items(df: pd.DataFrame, likert_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if likert_cols is None:
        likert_cols = _detect_likert_columns(df, exclude_contains=EXCLUDE_CONTAINS)
    out = pd.DataFrame(index=df.index)
    for c in likert_cols:
        out[f"{c}_kategori"] = pd.to_numeric(df[c], errors="coerce").apply(_likert_to_category)
    return out

def discretize_domain_means(df: pd.DataFrame, domains: Optional[List[str]] = None) -> pd.DataFrame:
    domains = domains or DOMAINS
    out = pd.DataFrame(index=df.index)
    for d in domains:
        rounded = pd.to_numeric(df[d], errors="coerce").round().clip(lower=1, upper=5)
        out[f"{d}_MeanKategori"] = rounded.apply(_likert_to_category)
    return out

# ====== MAIN ======
def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"File tidak ditemukan: {INPUT_FILE}")

    # 1) Load & rapikan header
    xls = pd.ExcelFile(INPUT_FILE)
    df = pd.read_excel(INPUT_FILE, sheet_name=xls.sheet_names[0])
    df.columns = [(" ".join(str(c).strip().split())) for c in df.columns]

    # 2) Perbaiki Timestamp bila ada (di memory saja)
    df_raw = _fix_timestamp_if_any(df.copy())

    # 3) Transformasi otomatis (mean + z per domain)
    df_transformed, groups, log_text = aggregate_aspects_auto(
        df_raw, domains=DOMAINS, exclude_contains=EXCLUDE_CONTAINS
    )

    # 4) Susun Bagan_2_Hasil: semua Mean dulu, lalu semua Z
    df_hasil = pd.concat(
        [df_transformed[d] for d in DOMAINS] + [df_transformed[f"{d}_Z"] for d in DOMAINS],
        axis=1
    )
    df_hasil.columns = DOMAINS + [f"{d}_Z" for d in DOMAINS]

    # 5) Bagan_3: Discretized (per-butir + per-mean)
    likert_cols = _detect_likert_columns(df_raw, exclude_contains=EXCLUDE_CONTAINS)
    df_items_kat = (discretize_likert_items(df_raw[likert_cols])
                    if likert_cols else pd.DataFrame({"Info": ["Tidak ada kolom Likert terdeteksi"]}))
    df_mean_kat  = discretize_domain_means(df_transformed, domains=DOMAINS)
    df_discretized = (pd.concat([df_items_kat, df_mean_kat], axis=1)
                      if "Info" not in df_items_kat.columns else df_items_kat)

    # 6) Tulis Excel — HANYA HASIL (tanpa Bagan_1_RAW)
    out_name = os.path.splitext(os.path.basename(INPUT_FILE))[0] + "__HASIL_ONLY.xlsx"
    engine = "xlsxwriter"
    try:
        __import__("xlsxwriter")
    except Exception:
        engine = "openpyxl"

    with pd.ExcelWriter(out_name, engine=engine) as writer:
        df_hasil.to_excel(writer, sheet_name="Bagan_2_Hasil", index=False)
        df_discretized.to_excel(writer, sheet_name="Bagan_3_Discretized", index=False)
        if WRITE_LOG_SHEET:
            log_df = pd.DataFrame({"Log": log_text.split("\n")})
            log_df.to_excel(writer, sheet_name="Log_Transform", index=False)

        for sh in ["Bagan_2_Hasil", "Bagan_3_Discretized"] + (["Log_Transform"] if WRITE_LOG_SHEET else []):
            ws = writer.sheets.get(sh)
            try:
                ws.freeze_panes(1, 0)
            except Exception:
                pass

    print("=== DONE ===")
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {out_name}")
    print("Sheets: Bagan_2_Hasil, Bagan_3_Discretized" + (", Log_Transform" if WRITE_LOG_SHEET else ""))
    print("Kolom Bagan_2_Hasil: [Means…] + [Z-scores…]")

if __name__ == "__main__":
    main()
