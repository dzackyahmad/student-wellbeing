# DataIntegration.py
# Gabung baris (append/union) dua DataFrame

from typing import Tuple
import pandas as pd

def append_rows(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    mode: str = "union",          # "union" (semua kolom) atau "intersection" (hanya kolom sama)
    add_source: bool = True,      # tambahkan kolom _source untuk jejak asal data
    left_name: str = "A",
    right_name: str = "B",
) -> Tuple[pd.DataFrame, str]:
    """
    Menggabungkan dua dataset secara vertikal (stack baris).
    - mode="union"        : semua kolom dari kiri ∪ kanan, yang tidak ada → NaN.
    - mode="intersection" : hanya kolom yang sama pada kedua dataset.
    """
    if df_left is None or df_right is None:
        return df_left, "⚠️ Salah satu dataset kosong — append dibatalkan."

    mode = (mode or "union").lower()
    if mode not in {"union", "intersection"}:
        mode = "union"

    if mode == "intersection":
        common_cols = [c for c in df_left.columns if c in df_right.columns]
        L = df_left[common_cols].copy()
        R = df_right[common_cols].copy()
    else:
        all_cols = sorted(list(set(df_left.columns) | set(df_right.columns)))
        L = df_left.reindex(columns=all_cols).copy()
        R = df_right.reindex(columns=all_cols).copy()

    if add_source:
        L["_source"] = left_name
        R["_source"] = right_name

    out = pd.concat([L, R], axis=0, ignore_index=True, sort=False)

    log = (
        f"🧩 Append rows | mode={mode} | "
        f"kiri={df_left.shape[0]} baris, kanan={df_right.shape[0]} baris "
        f"→ hasil={out.shape[0]} baris × {out.shape[1]} kolom."
    )
    if mode == "intersection":
        log += f"\nKolom yang dipakai (intersection): {list(L.columns)}"
    return out, log
