# app.py
# Streamlit 1.11.0 compatible; no sklearn; UI only – logic lives in the modules.
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Import modul logika (dengan fallback aman) ---
try:
    import DataCleaning as DC
except Exception as e:
    DC = None

try:
    import DataIntegration as DI
except Exception as e:
    DI = None

try:
    import DataTransformation as DT
except Exception as e:
    DT = None

try:
    import DataReduction as DR
except Exception as e:
    DR = None

try:
    import DataDiscretization as DD
except Exception as e:
    DD = None

# --- Import modul Visualisasi (controller) ---
try:
    from Visualisasi import Visualisasi as VIS  # jika folder Visualisasi adalah package
except Exception:
    try:
        import Visualisasi.Visualisasi as VIS   # alternatif path
    except Exception:
        VIS = None



# ---------- Utils ----------
def line():
    st.markdown('---')

def init_state():
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    if "df_work" not in st.session_state:
        st.session_state.df_work = None
    if "logs" not in st.session_state:
        st.session_state.logs = []

def log(msg):
    st.session_state.logs.append(str(msg))

def preview_df(df, title="Preview Data"):
    st.subheader(title)
    if df is None:
        st.info("Belum ada data.")
        return
    # kasih key unik berdasarkan judul
    key = f"slider_{title.replace(' ', '_')}"
    n_rows = st.slider("Tampilkan n baris awal", 5, 1000, 20, key=key)
    st.write(df.head(n_rows))
    st.caption(f"Shape: {df.shape[0]} baris × {df.shape[1]} kolom")


def numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def cat_columns(df: pd.DataFrame):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

# --- helper UI kartu rapi ---
def render_wisdom_cards(wisdom_list, prof_df, basis_name,
                        aspect_order=("Education","Financial","Physical","Psychological","Relational")):
    import streamlit as st
    st.markdown("""
    <style>
      .wiz-card{background:#111418;border:1px solid #2a2f36;border-radius:16px;padding:16px 18px;margin:14px 0}
      .wiz-title{font-weight:700;font-size:1.1rem;margin-bottom:.35rem}
      .chips{display:flex;flex-wrap:wrap;gap:.4rem .6rem;margin:.3rem 0 .6rem 0}
      .chip{padding:4px 10px;border-radius:999px;font-size:.86rem;border:1px solid #3a404a;background:#1a1f27}
      .hi{background:#16261b;border-color:#317a43}
      .lo{background:#2a1b1b;border-color:#7c3535}
      .ne{background:#1a1f27;border-color:#3a404a}
      .sec{font-weight:600;margin:.4rem 0 .2rem 0}
      .muted{color:#aab1ba;font-size:.9rem}
      .pin{margin:.2rem 0 .2rem 0}
    </style>
    """, unsafe_allow_html=True)

    if str(basis_name).lower().startswith("z"):
        low_thr, high_thr = -0.4, 0.4; fmt = lambda v: f"{v:+.2f}"
    else:
        low_thr, high_thr = 3.0, 3.5; fmt = lambda v: f"{v:.2f}"

    for w in wisdom_list:
        cid = w["cluster"]
        try: row = prof_df.loc[int(cid)]
        except Exception: row = prof_df.loc[cid]

        label_txt = f" — {w['label']}" if w.get("label") else ""
        st.markdown('<div class="wiz-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="wiz-title">Cluster {cid}{label_txt} — Anggota: {int(w["n"])} · Basis: {w["basis"]}</div>', unsafe_allow_html=True)

        if w.get("primary_action"): st.markdown(f'<div class="pin"><b>📌 Rekomendasi Utama:</b> {w["primary_action"]}</div>', unsafe_allow_html=True)
        if w.get("primary_cause"):  st.markdown(f'<div class="pin"><b>🧠 Penyebab Utama:</b> {w["primary_cause"]}</div>', unsafe_allow_html=True)
        if w.get("why"):            st.markdown(f'<div class="muted">{w["why"]}</div>', unsafe_allow_html=True)

        chips = ['<div class="chips">']
        for a in aspect_order:
            col = a + "_Z" if (a + "_Z") in row.index else (a if a in row.index else None)
            if col is None: continue
            val = float(row[col])
            cls = "hi" if val >= high_thr else ("lo" if val <= low_thr else "ne")
            chips.append(f'<span class="chip {cls}">{a}: {fmt(val)}</span>')
        chips.append('</div>')
        st.markdown("".join(chips), unsafe_allow_html=True)

        if w.get("linkages"):
            st.markdown('<div class="sec">Keterkaitan Antar-Domain:</div>', unsafe_allow_html=True)
            for t in w["linkages"]: st.markdown(f"- {t}")
        if w.get("causes"):
            st.markdown('<div class="sec">Asumsi Penyebab (tambahan):</div>', unsafe_allow_html=True)
            for c in w["causes"]: st.markdown(f"- {c}")
        if w.get("recommendations"):
            st.markdown('<div class="sec">Saran Solusi (lengkap):</div>', unsafe_allow_html=True)
            for r in w["recommendations"]: st.markdown(f"- {r}")

        if w.get("strengths"):
            st.markdown(f'<div class="muted">Modal yang bisa diskalakan: {"; ".join(w["strengths"])}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- App ----------
st.set_page_config(page_title="Preprocessing App", layout="wide")
init_state()

st.title("Data Kesejahteraan Mahasiswa FMIPA Unpad")

# ========== Sidebar (dua level, tanpa radio) ==========
with st.sidebar:
    st.header("🧭 Navigasi")

    preprocess_pages = [
        "1) Upload Data",
        "2) Data Cleaning",
        "3) Data Integration",
        "4) Data Transformation",
        "5) Data Reduction",
        "6) Data Discretization",
        "7) Log & Unduh",
    ]
    visual_pages = [
        "8) Visualisasi Clustering",
        "9) Insight Cluster",
        "10) Action Plan (Wisdom)",
    ]

    # --- state awal & last selections (untuk deteksi perubahan selectbox) ---
    prev_page = st.session_state.get("page_selected", preprocess_pages[0])
    st.session_state.setdefault("sel_pre_last", preprocess_pages[0])
    st.session_state.setdefault("sel_vis_last", visual_pages[0])

    # tentukan grup aktif saat ini dari prev_page
    current_group = "Preprocess" if prev_page in preprocess_pages else "Visualisasi"

    # expand hanya grup aktif
    with st.expander("Preprocessing", expanded=(current_group == "Preprocess")):
        sel_pre = st.selectbox("Menu", preprocess_pages,
                               index=(preprocess_pages.index(prev_page) if prev_page in preprocess_pages else 0),
                               key="sel_pre")

    with st.expander("Visualisasi", expanded=(current_group == "Visualisasi")):
        sel_vis = st.selectbox("Menu", visual_pages,
                               index=(visual_pages.index(prev_page) if prev_page in visual_pages else 0),
                               key="sel_vis")

    # --- tentukan page berdasarkan selectbox yang terakhir berubah ---
    changed_group = None
    if sel_pre != st.session_state["sel_pre_last"]:
        changed_group = "Preprocess"
    if sel_vis != st.session_state["sel_vis_last"]:
        changed_group = "Visualisasi"

    if changed_group == "Preprocess":
        page = sel_pre
        current_group = "Preprocess"
    elif changed_group == "Visualisasi":
        page = sel_vis
        current_group = "Visualisasi"
    else:
        # tidak ada perubahan; pakai page sebelumnya
        page = prev_page

    # simpan last selections & page terpilih
    st.session_state["sel_pre_last"] = sel_pre
    st.session_state["sel_vis_last"] = sel_vis
    st.session_state["page_selected"] = page

    st.markdown("---")
    st.caption("Status Dataset")
    if st.session_state.get("df_work") is None:
        st.markdown('<span class="badge warn">Belum ada data</span>', unsafe_allow_html=True)
    else:
        r, c = st.session_state.df_work.shape
        st.markdown(f'<span class="badge ok">Work: {r}×{c}</span>', unsafe_allow_html=True)
        if st.session_state.get("df_raw") is not None:
            r0, c0 = st.session_state.df_raw.shape
            st.caption(f"Raw: {r0}×{c0}")
    if len(st.session_state.get("logs", [])) > 0:
        st.caption(f"Log: {len(st.session_state['logs'])} entri")


# 1) Upload Data
if page == "1) Upload Data":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Upload Dataset")
    st.write("Dukung: CSV, XLSX")
    file = st.file_uploader("Pilih file", type=["csv", "xlsx"])

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("🗑️ Hapus Data", key="btn_delete_all"):
            st.session_state.df_raw = None
            st.session_state.df_work = None
            st.session_state.logs = []
            st.success("✅ Semua data berhasil dihapus.")



    if file is not None:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        st.session_state.df_raw = df.copy()
        st.session_state.df_work = df.copy()
        log(f"Upload file: {file.name} – shape {df.shape}")

    st.markdown('</div>', unsafe_allow_html=True)
    if st.session_state.df_raw is not None:
        df0 = st.session_state.df_raw
        n_num = sum(pd.api.types.is_numeric_dtype(df0[c]) for c in df0.columns)
        n_cat = df0.shape[1] - n_num
        miss_pct = round(float(df0.isna().mean().mean()*100),2)

        ca, cb, cc, cd = st.columns(4)
        ca.metric("Rows", f"{df0.shape[0]:,}")
        cb.metric("Cols", f"{df0.shape[1]:,}")
        cc.metric("Numeric", f"{n_num}")
        cd.metric("Avg Missing", f"{miss_pct}%")

    preview_df(st.session_state.df_raw, "Data Awal")

# 2) Data Cleaning
elif page == "2) Data Cleaning":
    st.subheader("Data Cleaning")
    df = st.session_state.df_work
    preview_df(df, "Sebelum Cleaning")
    target_cols = None  # Default: None, can be set by user if needed

    # === Reset khusus Cleaning ===
    col_reset1, col_reset2 = st.columns(2)
    with col_reset1:
        if st.button("Reset ke Data Awal", key="btn_reset_cleaning"):
            if st.session_state.df_raw is not None:
                st.session_state.df_work = st.session_state.df_raw.copy()
                log("Reset Data Cleaning: df_work dikembalikan ke df_raw.")
                st.success("Berhasil reset!")
            else:
                st.warning("Upload File Terlebih Dahulu!.")

    # === Hapus kolom tidak penting ===
    if df is not None:
        all_cols = df.columns.tolist()
        drop_cols = st.multiselect("Pilih kolom yang ingin dihapus:", all_cols, key="drop_cols_clean")
        if st.button("Hapus Kolom yang Dipilih", key="btn_drop_cols"):
            st.session_state.df_work = df.drop(columns=drop_cols, errors="ignore")
            log(f"Hapus kolom: {drop_cols}")
            st.success(f"Kolom {drop_cols} berhasil dihapus.")

    if df is not None:
        line()

        # ============ MISSING VALUE (CARD) ============
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧩 Missing Value</div>', unsafe_allow_html=True)

        strategy = st.selectbox(
            "Strategi penanganan missing",
            ["Drop", "Mean", "Median", "Mode", "Constant"],
            key="missing_strategy",
        )

        const_val = None
        if strategy == "Constant":
            c1, c2 = st.columns([2,1])
            with c1:
                const_val = st.text_input(
                    "Nilai konstanta (akan dicoba dikonversi numerik jika cocok)",
                    "0",
                    key="const_val"
                )
            with c2:
                st.caption("💡 Cocok untuk kolom kategorik.")

        st.markdown('</div>', unsafe_allow_html=True)  # /section-card

        line()

        # ============ OUTLIER (CARD) ============
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Outlier (Deteksi otomatis: Z-score)</div>', unsafe_allow_html=True)

        outlier_mode = st.selectbox(
            "Mode penanganan outlier",
            ["Clip", "Drop", "Mark", "Set_NaN", "Impute_Median", "Impute_Constant"],
            index=2,  # default: mark
            key="outlier_mode",
        )

        c1, c2 = st.columns([2,1])
        with c1:
            z_thr = st.slider("Ambang |Z|", 2.0, 5.0, 3.0, 0.1, key="z_thr_auto")
        with c2:
            fill_value_out = None
            if outlier_mode == "Impute_Constant":
                fill_value_out = st.text_input("Nilai konstanta imputasi", "0", key="const_outlier")



        # === Tombol Jalankan Cleaning ===
        if st.button("Jalankan Cleaning", key="btn_run_clean"):
            try:
                if DC and hasattr(DC, "clean_data"):
                    df_clean, clean_log = DC.clean_data(
                        df,
                        missing_method=strategy,
                        fill_value=const_val,
                        outlier_method="zscore",         # paksa deteksi Z-score otomatis
                        z_threshold=z_thr,
                        outlier_mode=outlier_mode,
                        fill_value_outlier=fill_value_out,
                        target_cols=target_cols if target_cols else None,
                    )
                    st.session_state.df_work = df_clean
                    log(clean_log if clean_log else "Cleaning selesai via DataCleaning.clean_data (Z-score).")
                else:
                    # ====== Fallback minimalis (tanpa modul DC) ======
                    work = df.copy()

                    # --- Normalisasi NaN & koersi numerik-like ---
                    def _normalize_nans_local(dfin):
                        return dfin.replace(
                            to_replace=[r'^\s*$', r'^\-$', r'^(na|n/a|NA|N/A|null|NULL)$'],
                            value=np.nan, regex=True
                        )
                    def _coerce_numeric_like_local(dfin, thresh=0.7):
                        out = dfin.copy()
                        for c in out.columns:
                            if pd.api.types.is_numeric_dtype(out[c]):
                                continue
                            parsed = pd.to_numeric(out[c], errors="coerce")
                            if parsed.notna().mean() >= thresh:
                                out[c] = parsed
                        return out

                    work = _normalize_nans_local(work)
                    work = _coerce_numeric_like_local(work)

                    # --- Missing handling ---
                    if strategy == "drop":
                        work = work.dropna()
                    elif strategy == "mean":
                        means = work.mean(numeric_only=True)
                        for c in means.index:
                            work[c] = work[c].fillna(means[c])
                        cat_cols_local = [c for c in work.columns if c not in means.index]
                        for c in cat_cols_local:
                            try:
                                m = work[c].mode(dropna=True).iloc[0]
                            except Exception:
                                m = "Unknown"
                            work[c] = work[c].fillna(m)
                    elif strategy == "median":
                        meds = work.median(numeric_only=True)
                        for c in meds.index:
                            work[c] = work[c].fillna(meds[c])
                        cat_cols_local = [c for c in work.columns if c not in meds.index]
                        for c in cat_cols_local:
                            try:
                                m = work[c].mode(dropna=True).iloc[0]
                            except Exception:
                                m = "Unknown"
                            work[c] = work[c].fillna(m)
                    elif strategy == "mode":
                        try:
                            work = work.fillna(work.mode().iloc[0])
                        except Exception:
                            for c in work.columns:
                                m = work[c].mode()
                                if not m.empty:
                                    work[c] = work[c].fillna(m.iloc[0])
                    elif strategy == "constant":
                        try:
                            v = pd.to_numeric(pd.Series([const_val]), errors="coerce").iloc[0]
                            v = v if pd.notna(v) else const_val
                        except Exception:
                            v = const_val
                        work = work.fillna(v)

                    # --- Outlier detection: Z-score (auto) + action ---
                    cols_apply = target_cols if target_cols else [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
                    union_mask = pd.Series(False, index=work.index)
                    counts = {}

                    for c in cols_apply:
                        col = pd.to_numeric(work[c], errors="coerce")
                        mu, sd = col.mean(), col.std(ddof=0)
                        if sd == 0 or pd.isna(sd):
                            counts[c] = 0
                            if outlier_mode == "mark":
                                work[c + "_is_outlier"] = False
                            continue
                        z = (col - mu) / sd
                        mask = z.abs() > z_thr
                        counts[c] = int(mask.sum())
                        lo, hi = mu - z_thr * sd, mu + z_thr * sd

                        if outlier_mode == "clip":
                            work[c] = col.clip(lo, hi)
                        elif outlier_mode == "remove":
                            union_mask = union_mask | mask
                        elif outlier_mode == "mark":
                            work[c + "_is_outlier"] = mask
                        elif outlier_mode == "set_nan":
                            work.loc[mask, c] = np.nan
                        elif outlier_mode == "impute_median":
                            work.loc[mask, c] = np.nan
                            med = work[c].median(skipna=True)
                            work[c] = work[c].fillna(med)
                        elif outlier_mode == "impute_constant":
                            work.loc[mask, c] = np.nan
                            try:
                                k = pd.to_numeric(pd.Series([fill_value_out]), errors="coerce").iloc[0]
                                k = k if pd.notna(k) else fill_value_out
                            except Exception:
                                k = fill_value_out
                            work[c] = work[c].fillna(k)

                    if outlier_mode == "remove":
                        dropped = int(union_mask.sum())
                        work = work.loc[~union_mask].copy()
                        log(f"[Outlier][Fallback] remove: drop {dropped} baris")
                    else:
                        log(f"[Outlier][Fallback] mode={outlier_mode} | |Z|>{z_thr} | per kolom: {counts}")

                    st.session_state.df_work = work
                    log(f"[Missing][Fallback] strategi={strategy}")

                st.success("Cleaning selesai.")
            except Exception as e:
                st.error(f"Gagal cleaning: {e}")

        preview_df(st.session_state.df_work, "Sesudah Cleaning")


# 3) Data Integration — APPEND (stack baris)
elif page == "3) Data Integration":
    st.subheader("Data Integration — Append (gabung baris)")
    left_df = st.session_state.df_work
    preview_df(left_df, "Dataset Awal")

    st.markdown("### Upload Dataset Tambahan")
    up_right = st.file_uploader(
        "Pilih file Tambahan (CSV/XLSX)",          # <-- label WAJIB
        type=["csv", "xlsx"],
        key="right_file_integration_append",
    )

    def _read_any(file_obj):
        if file_obj is None:
            return None
        name = file_obj.name.lower()
        try:
            return pd.read_csv(file_obj) if name.endswith(".csv") else pd.read_excel(file_obj)
        except Exception as e:
            st.error(f"Gagal membaca file Tambahan: {e}")
            return None

    right_df = _read_any(up_right) if up_right else None
    if right_df is not None:
        preview_df(right_df, "Dataset Tambahan")

    # Opsi append
    st.markdown("**Pengaturan Append**")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        mode = st.selectbox(
            "Metode Integrasi",
            ["Union (semua kolom)", "Intersection (hanya kolom sama)"],
            index=0,
            key="append_mode"
        )
    with col2:
        add_source = st.checkbox("Tambahkan kolom _source", value=True, key="append_add_source")
    with col3:
        left_name = st.text_input("Label sumber kiri", "A", key="append_left_name")
        right_name = st.text_input("Label sumber kanan", "B", key="append_right_name")

    # normalisasi ke nilai yang diharapkan modul: 'union' / 'intersection'
    mode_val = "union" if mode.lower().startswith("union") else "intersection"

    if (left_df is not None) and (right_df is not None):
        if st.button("Gabungkan (Append Baris)", key="btn_append_rows"):
            try:
                if DI and hasattr(DI, "append_rows"):
                    merged, logtxt = DI.append_rows(
                        left_df, right_df,
                        mode=mode_val,
                        add_source=add_source,
                        left_name=left_name,
                        right_name=right_name,
                    )
                else:
                    # fallback sederhana
                    if mode_val == "intersection":
                        common = [c for c in left_df.columns if c in right_df.columns]
                        L = left_df[common].copy(); R = right_df[common].copy()
                    else:
                        all_cols = sorted(list(set(left_df.columns) | set(right_df.columns)))
                        L = left_df.reindex(columns=all_cols).copy()
                        R = right_df.reindex(columns=all_cols).copy()
                    if add_source:
                        L["_source"] = left_name; R["_source"] = right_name
                    merged = pd.concat([L, R], axis=0, ignore_index=True, sort=False)
                    logtxt = f"[Fallback] Append mode={mode_val} → hasil {merged.shape[0]} baris × {merged.shape[1]} kolom."

                st.session_state.df_work = merged
                log(logtxt)
                st.success("Append selesai. Dataset Baru ditambahkan di bawah Dataset Awal.")
            except Exception as e:
                st.error(f"Gagal append: {e}")

    preview_df(st.session_state.df_work, "Hasil Integrasi (df_work)")


# 4) Data Transformation (AUTO: mean per aspek + z-score per aspek)
elif page == "4) Data Transformation":
    st.subheader("Data Transformation")
    df = st.session_state.df_work
    preview_df(df, "Data Sebelum Transformasi")

    # ---------- Helpers robust ----------
    def _make_unique_columns(df_in: pd.DataFrame) -> pd.DataFrame:
        """Pastikan semua nama kolom unik: Kolom, Kolom.1, Kolom.2, ..."""
        seen = {}
        new_cols = []
        for c in df_in.columns:
            name = str(c)
            if name not in seen:
                seen[name] = 0
                new_cols.append(name)
            else:
                seen[name] += 1
                new_cols.append(f"{name}.{seen[name]}")
        out = df_in.copy()
        out.columns = new_cols
        return out

    def _as_series(df_in: pd.DataFrame, colname: str) -> pd.Series:
        """Selalu kembalikan 1D Series meskipun ada kolom duplikat."""
        obj = df_in[colname]
        if isinstance(obj, pd.DataFrame):
            # ambil kolom pertama kalau duplikat
            obj = obj.iloc[:, 0]
        return obj

    def _find_timestamp_col(df_in: pd.DataFrame):
        ex = [c for c in df_in.columns if str(c).strip().lower() == "timestamp"]
        if ex:
            return ex[0]
        fuzzy = [c for c in df_in.columns if "timestamp" in str(c).lower()]
        return fuzzy[0] if fuzzy else None

    def _fix_timestamp_if_any(df_in: pd.DataFrame) -> pd.DataFrame:
        df_in = _make_unique_columns(df_in.copy())
        ts_col = _find_timestamp_col(df_in)
        if not ts_col:
            return df_in
        s1d = _as_series(df_in, ts_col)
        s_num = pd.to_numeric(s1d, errors="coerce")
        if s_num.notna().sum() == 0:
            df_in[ts_col] = pd.to_datetime(s1d, errors="coerce")
            return df_in
        med = float(s_num.dropna().median())
        if med > 1e17:   unit = "ns"
        elif med > 1e12: unit = "ms"
        elif med > 1e9:  unit = "s"
        else:            unit = None
        df_in[ts_col] = pd.to_datetime(s_num, unit=unit, errors="coerce") if unit else pd.to_datetime(s1d, errors="coerce")
        return df_in

    def _z_series_local(x: pd.Series) -> pd.Series:
        mu = float(np.nanmean(x.values))
        sd = float(np.nanstd(x.values, ddof=0))
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - mu) / sd

    def _detect_likert_columns_local(df_in, exclude_contains):
        LIKERT_VALUES = {1,2,3,4,5}
        cand = []
        for c in df_in.columns:
            s = pd.to_numeric(_as_series(df_in, c), errors="coerce")  # <-- selalu Series 1D
            nn = s.notna().sum()
            if nn == 0:
                continue
            si = s.dropna().round().astype(int)
            share = (si.isin(LIKERT_VALUES).sum()) / max(1, nn)
            if share >= 0.70 and si.nunique() <= 5 and all(ex.lower() not in str(c).lower() for ex in exclude_contains):
                cand.append(c)
        # pertahankan urutan asli
        return [c for c in df_in.columns if c in set(cand)]

    def _group_by_six_local(cols, domains):
        groups = {}; start = 0
        for d in domains:
            groups[d] = cols[start:start+6]
            start += 6
        return groups
    # ------------------------------------

    if df is not None:
        # Opsi diagnostik
        st.markdown("**Pengaturan Deteksi**")
        colA, colB, colC = st.columns(3)
        with colA:
            show_diag = st.checkbox("Diagnosis Kolom Likert", value=False, key="diag_aspek_auto")
        with colB:
            _default = "Semester,Timestamp,Nama,Jenis Kelamin,Jurusan"
            st.session_state.setdefault("exclude_aspek_auto", _default)

            show_adv = st.checkbox("Atur daftar pengecualian (opsional)", value=False, key="show_exc_adv")
            if show_adv:
                st.session_state["exclude_aspek_auto"] = st.text_input(
                    "Daftar kolom dipisah koma", value=st.session_state["exclude_aspek_auto"], key="exclude_aspek_auto_input"
                )

            ex_text = st.session_state["exclude_aspek_auto"]
            exclude_list = [s.strip() for s in ex_text.split(",") if s.strip()]

        with colC:
            force_internal = st.checkbox("Gunakan Engine Internal", value=True, key="force_internal_dt")

        st.caption(f"Duplikat kolom (sebelum proses): {df.columns[df.columns.duplicated()].tolist()}")

        st.markdown("**Proses Agregasi Otomatis**")
        if st.button("Hitung Skor Mean & Z-SCore", key="btn_aspek_auto"):
            try:
                # 1) dedup + perbaiki timestamp
                base = _make_unique_columns(df.copy())
                df_fixed = _fix_timestamp_if_any(base)

                domains = ["Education", "Financial", "Physical", "Psychological", "Relational"]

                # 2) Pakai DT kalau diizinkan & ada; kalau gagal -> fallback
                use_dt = (not force_internal) and (DT is not None) and hasattr(DT, "aggregate_aspects_auto")
                df_new = None; agg_log = ""

                if use_dt:
                    try:
                        res = DT.aggregate_aspects_auto(
                            df_fixed, domains=domains, exclude_contains=exclude_list
                        )
                        # dukung return (df, log) atau (df, groups, log)
                        if isinstance(res, tuple) and len(res) == 2:
                            df_new, agg_log = res
                        elif isinstance(res, tuple) and len(res) == 3:
                            df_new, _, agg_log = res
                        else:
                            raise ValueError("Format return DT.aggregate_aspects_auto tidak dikenali")
                    except Exception as e_dt:
                        st.info(f"Modul DT gagal ({e_dt}). Menggunakan engine internal.")
                        use_dt = False  # jatuh ke fallback

                if not use_dt:
                    # --- FALLBACK INTERNAL ---
                    likert_cols = _detect_likert_columns_local(df_fixed, exclude_list)
                    groups = _group_by_six_local(likert_cols, domains)
                    work = df_fixed.copy()
                    logs = [f"[Deteksi Likert] total kandidat: {len(likert_cols)} kolom"]
                    # Mean per domain
                    for dname in domains:
                        cols = groups[dname]
                        if cols:
                            work[dname] = work[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                            logs.append(f"[Mean] {dname}: mean dari {cols}")
                        else:
                            work[dname] = np.nan
                            logs.append(f"[Mean] {dname}: kolom tidak cukup (butuh 6).")
                    # Z per domain (pastikan 1D)
                    for dname in domains:
                        ser = _as_series(work, dname)
                        work[f"{dname}_Z"] = _z_series_local(ser)
                        logs.append(f"[Z-score] {dname}_Z dibuat dari {dname}")
                    # Urutan: [Means...] + [Z-scores...]
                    mean_cols = domains
                    z_cols = [f"{d}_Z" for d in domains]
                    df_new = work[mean_cols + z_cols]
                    agg_log = "\n".join(logs)
                    # --- END FALLBACK ---

                st.session_state.df_work = df_new
                log(agg_log if agg_log else "Agregasi aspek (mean) + Z-score selesai.")
                st.success("Agregasi Aspek (Mean) + Z-score selesai.")

            except Exception as e:
                st.error(f"Gagal proses agregasi otomatis: {e}")

        # Diagnostik (opsional)
        if show_diag and st.session_state.df_work is not None:
            try:
                df_diag = _fix_timestamp_if_any(_make_unique_columns(st.session_state.df_work.copy()))
                likert_cols = _detect_likert_columns_local(df_diag, exclude_list)
                st.markdown("**Kolom Likert Terdeteksi (urut):**")
                st.write(likert_cols[:60])

                groups_diag = _group_by_six_local(likert_cols, domains)
                st.markdown("**Mapping Domain → 6 Item:**")
                for dname in domains:
                    st.write(f"- {dname}: {groups_diag[dname]}")
            except Exception as e:
                st.warning(f"Diagnosis deteksi gagal: {e}")
                
        preview_df(st.session_state.df_work, "Data Sesudah Transformasi")


# 5) Data Reduction (PCA)
elif page.startswith("5) Data Reduction"):
    st.subheader("Data Reduction — PCA & UMAP")

    df = st.session_state.df_work
    preview_df(df, "Data Sebelum Reduksi")

    if df is None or len(df) == 0:
        st.info("Belum ada data. Upload data terlebih dahulu.")
    else:
        cols_num = numeric_columns(df)

        with st.expander("🔧 Pilih Kolom Fitur (numerik)", expanded=False):
            # dua preset:
            base_aspects = ["Education", "Financial", "Physical", "Psychological", "Relational"]
            preset_mean = base_aspects
            preset_z = [f"{a}_Z" for a in base_aspects]

            basis = st.selectbox(
                "Basis fitur",
                ["Mean (1–5)", "Z-Score"],
                index=0,
                key="red_basis_select"
            )

            # tentukan preset sesuai basis, lalu intersect dengan kolom yang tersedia
            wanted = preset_mean if basis.startswith("Mean") else preset_z
            present = [c for c in wanted if c in df.columns]
            missing = [c for c in wanted if c not in df.columns]

            use_preset = st.checkbox("Gunakan preset kolom sesuai basis", value=True, key="red_use_preset")

            if use_preset:
                # target_cols = kolom preset yang tersedia saja
                target_cols = present
                # info kecil
                st.caption(f"Kolom dipakai: {', '.join(present) if present else '(tidak ada yang tersedia)'}")
                if missing:
                    st.warning(f"Kolom tidak ditemukan & di-skip: {', '.join(missing)}")
            else:
                # custom: multiselect dengan default = preset yang tersedia
                target_cols = st.multiselect(
                    "Pilih kolom (kosongkan = semua numerik)",
                    options=cols_num + [c for c in wanted if c not in cols_num and c in df.columns],
                    default=present,
                    key="red_target_cols"
                )
                # jika user kosongkan, biarkan None? → biar pipeline lama pakai semua numerik
                if len(target_cols) == 0:
                    target_cols = None
                    st.caption(f"Tidak ada yang dipilih → semua kolom numerik akan diproses ({len(cols_num)} kolom).")


        # Pilih metode
        method = st.radio(
            "Metode reduksi",
            ["PCA", "UMAP"],
            index=0, key="red_method"
        )

        keep_src = st.checkbox("Keep source features (jangan drop fitur asli)", value=True, key="red_keep_src")

        # Parameter
        if method.startswith("PCA"):
            colA, colB, colC = st.columns([1,1,1])
            with colA:
                n_comp = st.number_input("n_components", 1, 10, 2, 1, key="pca_ncomp")
            with colB:
                scale = st.checkbox("Z-score sebelum PCA (Disarankan)", value=True, key="pca_scale")
            with colC:
                show_scree = st.checkbox("Hitung Scree Data", value=False, key="pca_scree")
        else:
            # UMAP
            c1, c2, c3, c4 = st.columns([1,1,1,1])
            with c1:
                n_comp = st.number_input("n_components", 2, 5, 2, 1, key="umap_ncomp")
            with c2:
                n_neighbors = st.slider("n_neighbors", 5, 100, 25, 1, key="umap_neighbors")
            with c3:
                min_dist = st.slider("min_dist", 0.0, 0.99, 0.15, 0.01, key="umap_mindist")
            with c4:
                metric = st.selectbox("metric", ["euclidean","manhattan","cosine","hamming"], index=0, key="umap_metric")
            scale = st.checkbox("Z-score sebelum UMAP (Disarankan)", value=True, key="umap_scale")
            # default tersembunyi: 42
            st.session_state.setdefault("umap_rs", 42)
            show_seed = st.checkbox("Atur random_state (opsional)", value=False, key="show_umap_seed")

            if show_seed:
                st.session_state["umap_rs"] = st.number_input("random_state", 0, 10_000, st.session_state["umap_rs"], 1, key="umap_rs_input")

            rand_state = st.session_state["umap_rs"]


        # Jalankan
        if st.button("🧭 Jalankan Reduksi", key="btn_run_reduction"):
            # simpan snapshot sebelum menimpa df_work
            st.session_state["_df_before_reduction"] = st.session_state.df_work.copy()
            try:
                df_in = df[target_cols] if target_cols else df[cols_num]
                if df_in is None or df_in.shape[1] == 0:
                    st.warning("Tidak ada kolom numerik yang dipilih/tersedia.")
                else:
                    if method.startswith("PCA"):
                        if DR and hasattr(DR, "pca_reduce"):
                            df_red, model, red_log, evr = DR.pca_reduce(df_in, n_components=int(n_comp), scale_zscore=bool(scale))
                        else:
                            # Fallback lokal sederhana (gunakan DataReduction di memori jika tidak terimport)
                            from numpy.linalg import svd
                            X = df_in.apply(pd.to_numeric, errors="coerce").fillna(0.0).values
                            if scale:
                                mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd==0]=1.0; X=(X-mu)/sd
                            U,S,VT = svd(X, full_matrices=False)
                            comps = VT[:int(n_comp)]
                            scores = X @ comps.T
                            cols = [f"PC{i+1}" for i in range(int(n_comp))]
                            df_red = pd.DataFrame(scores[:, :int(n_comp)], columns=cols, index=df.index)
                            evr = (S**2)/np.sum(S**2)
                            red_log = f"[PCA Fallback] OK: comp={n_comp}, Total EV={round(float(np.sum(evr[:int(n_comp)]))*100,2)}%."
                        # gabungkan hasil
                        if keep_src:
                            out = df.copy().join(df_red)
                        else:
                            # drop fitur input saja
                            out = df.drop(columns=df_in.columns, errors="ignore").join(df_red)
                        st.session_state.df_work = out
                        log(red_log)
                        st.success("PCA selesai. Komponen ditambahkan.")
                        # ---- Scree (UI yang enak dibaca) ----
                        if method.startswith("PCA") and show_scree:
                            try:
                                # 1) Ambil EVR
                                if DR and hasattr(DR, "scree_data"):
                                    comps, evr = DR.scree_data(df_in, scale_zscore=bool(scale))
                                else:
                                    # coba ambil dari model jika ada
                                    evr = None
                                    try:
                                        evr = getattr(model, "explained_variance_ratio_", None)
                                    except Exception:
                                        pass
                                    if evr is None:
                                        # fallback: hitung cepat (tanpa mengubah hasil di atas)
                                        from numpy.linalg import svd
                                        X = df_in.apply(pd.to_numeric, errors="coerce").fillna(0.0).values
                                        if scale:
                                            mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd==0] = 1.0; X = (X - mu) / sd
                                        U, S, VT = svd(X, full_matrices=False)
                                        evr = (S**2) / np.sum(S**2)
                                    comps = list(range(1, len(evr) + 1))

                                # 2) Tabel rapi + rekomendasi k
                                evr = list(evr)
                                cum = np.cumsum(evr)
                                df_scree = pd.DataFrame({
                                    "Komponen": comps,
                                    "EVR (Var %)": [f"{v*100:.1f}%" for v in evr],
                                    "EVR Kumulatif": [f"{v*100:.1f}%" for v in cum],
                                })

                                st.markdown("### 📈 Scree (Explained Variance)")
                                st.table(df_scree.head(10))  # v1.11: pakai table/dataframe tanpa use_container_width

                            except Exception as e:
                                st.warning(f"Gagal membuat scree: {e}")


                    else:
                        # UMAP
                        if DR and hasattr(DR, "umap_reduce"):
                            df_red, model, red_log, _ = DR.umap_reduce(
                                df_in,
                                n_components=int(n_comp),
                                n_neighbors=int(n_neighbors),
                                min_dist=float(min_dist),
                                metric=str(metric),
                                scale_zscore=bool(scale),
                                random_state=int(rand_state),
                            )
                        else:
                            # Fallback: gunakan PCA tapi beri nama UMAP*
                            from numpy.linalg import svd
                            X = df_in.apply(pd.to_numeric, errors="coerce").fillna(0.0).values
                            if scale:
                                mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd==0]=1.0; X=(X-mu)/sd
                            U,S,VT = svd(X, full_matrices=False)
                            comps = VT[:int(n_comp)]
                            scores = X @ comps.T
                            cols = [f"UMAP{i+1}" for i in range(int(n_comp))]
                            df_red = pd.DataFrame(scores[:, :int(n_comp)], columns=cols, index=df.index)
                            red_log = "[UMAP Fallback] 'umap-learn' tidak tersedia → pakai PCA dan dinamai UMAP*."
                        if keep_src:
                            out = df.copy().join(df_red)
                        else:
                            out = df.drop(columns=df_in.columns, errors="ignore").join(df_red)
                        st.session_state.df_work = out
                        log(red_log)
                        st.success("UMAP selesai. Embedding ditambahkan.")

            except Exception as e:
                st.error(f"Gagal reduksi: {e}")
        
        import re
        col_reset_a, col_reset_b = st.columns(2)

        # (1) Kembalikan ke snapshot sebelum reduksi
        with col_reset_a:
            if st.button("Kembalikan Data Sebelum Reduksi", key="btn_restore_pre_reduce"):
                snap = st.session_state.get("_df_before_reduction")
                if snap is None:
                    st.warning("Belum ada snapshot. Jalankan reduksi dulu agar snapshot tersimpan.")
                else:
                    st.session_state.df_work = snap.copy()
                    st.success("Data dikembalikan ke kondisi sebelum reduksi.")

        # (2) Hapus kolom hasil reduksi (PC*/UMAP*) saja
        with col_reset_b:
            if st.button("Hapus Kolom PC/UMAP dari Dataset", key="btn_drop_components"):
                dfc = st.session_state.df_work
                pat = re.compile(r"^(PC\d+|UMAP\d+)$", re.IGNORECASE)
                drop_cols = [c for c in dfc.columns if pat.match(str(c))]
                if not drop_cols:
                    st.info("Tidak ditemukan kolom PC/UMAP untuk dihapus.")
                else:
                    st.session_state.df_work = dfc.drop(columns=drop_cols, errors="ignore")
                    st.success(f"Kolom dihapus: {drop_cols}")


    preview_df(st.session_state.df_work, "Data Sesudah Reduksi")


# 6) Data Discretization (Z-score + Data-driven Quantile)
elif page == "6) Data Discretization":
    st.subheader("Data Discretization — Z-score & Quantile (anti-NaN)")

    df = st.session_state.df_work
    preview_df(df, "Sebelum Discretization")

    aspects = ["Education", "Financial", "Physical", "Psychological", "Relational"]

    # ── Pilih metode
    method_disc = st.selectbox(
        "Metode pelabelan",
        ["Z-score (manual threshold)", "Data-driven (Quantile)"],
        index=0, key="disc_method"
    )

    col1, col2 = st.columns(2)
    if method_disc.startswith("Z-score"):
        with col1:
            low_thr = st.number_input("Ambang bawah Z (Rendah)", value=-0.5, step=0.1, key="disc_z_low")
        with col2:
            high_thr = st.number_input("Ambang atas Z (Tinggi)", value=0.5, step=0.1, key="disc_z_high")
    else:
        with col1:
            q_low = st.number_input("Quantile rendah (0–1)", value=0.33, min_value=0.0, max_value=0.99, step=0.01, key="disc_q_low")
        with col2:
            q_high = st.number_input("Quantile tinggi (0–1)", value=0.67, min_value=0.01, max_value=1.0, step=0.01, key="disc_q_high")

    # ====== PREVIEW: Tabel Ambang (satu expander saja) ======
    with st.expander("📏 Tabel Ambang (Preview)", expanded=True):
        import numpy as np
        import pandas as pd

        aspects = ["Education", "Financial", "Physical", "Psychological", "Relational"]
        def _num(s): return pd.to_numeric(s, errors="coerce")

        # pastikan *_Z ada untuk hitung quantile/WB
        def _ensure_aspect_z_local(dfin, aspects_):
            out = dfin.copy()
            for a in aspects_:
                zc = f"{a}_Z"
                if zc not in out.columns:
                    x = _num(out.get(a, np.nan))
                    mu = float(np.nanmean(x)); sd = float(np.nanstd(x, ddof=0))
                    x_filled = x.fillna(mu if np.isfinite(mu) else 0.0)
                    if not np.isfinite(sd) or sd == 0:
                        out[zc] = 0.0
                    else:
                        out[zc] = (x_filled - (mu if np.isfinite(mu) else 0.0)) / sd
            return out

        df_prev = st.session_state.get("df_work")
        if df_prev is None or len(df_prev) == 0:
            st.info("Belum ada data untuk membuat tabel ambang.")
        else:
            work = _ensure_aspect_z_local(df_prev.copy(), aspects)
            zcols = [f"{a}_Z" for a in aspects]
            Z = work[zcols].apply(pd.to_numeric, errors="coerce")
            work["WB_Index_Z"] = Z.mean(axis=1)

            rows = []
            if st.session_state.get("disc_method", "Z-score (manual threshold)").startswith("Z-score"):
                low_thr = float(st.session_state.get("disc_z_low", -0.5))
                high_thr = float(st.session_state.get("disc_z_high", 0.5))
                rows += [
                    ["Z-score", "Tidak Sejahtera", f"≤ {low_thr:.2f}", "Berdasar Z tiap entri"],
                    ["Z-score", "Menengah",       f"({low_thr:.2f}, {high_thr:.2f})", "Di antara low & high"],
                    ["Z-score", "Sejahtera",      f"≥ {high_thr:.2f}", "Berdasar Z tiap entri"],
                    ["Z-score (WB_Index_Z)", "Tidak Sejahtera", f"≤ {low_thr:.2f}", "Rata-rata Z lima aspek"],
                    ["Z-score (WB_Index_Z)", "Menengah",       f"({low_thr:.2f}, {high_thr:.2f})", ""],
                    ["Z-score (WB_Index_Z)", "Sejahtera",      f"≥ {high_thr:.2f}", ""],
                ]
                df_cut = pd.DataFrame(rows, columns=["Basis", "Level", "Kriteria", "Catatan"])
                st.dataframe(df_cut)   # v1.11: jangan pakai use_container_width
            else:
                q_low = float(st.session_state.get("disc_q_low", 0.33))
                q_high = float(st.session_state.get("disc_q_high", 0.67))

                sWB = _num(work["WB_Index_Z"]).dropna()
                loWB = float(sWB.quantile(q_low)) if len(sWB) else np.nan
                hiWB = float(sWB.quantile(q_high)) if len(sWB) else np.nan
                df_cut_global = pd.DataFrame([
                    ["Quantile (WB_Index_Z)", "Tidak Sejahtera", f"≤ {loWB:.3f}", f"q_low={q_low:.2f}"],
                    ["Quantile (WB_Index_Z)", "Menengah",       f"({loWB:.3f}, {hiWB:.3f})", f"q_high={q_high:.2f}"],
                    ["Quantile (WB_Index_Z)", "Sejahtera",      f"≥ {hiWB:.3f}", ""],
                ], columns=["Basis", "Level", "Kriteria", "Catatan"])
                st.markdown("**Ambang Global (WB_Index_Z)**")
                st.dataframe(df_cut_global)

                per_aspect = []
                for a in aspects:
                    s = _num(work[f"{a}_Z"]).dropna()
                    loA = float(s.quantile(q_low)) if len(s) else np.nan
                    hiA = float(s.quantile(q_high)) if len(s) else np.nan
                    per_aspect.append([a, loA, hiA])
                df_cut_aspect = pd.DataFrame(per_aspect, columns=["Aspek", "Cut rendah (q_low)", "Cut tinggi (q_high)"])
                st.markdown("**Ambang Per Aspek (Quantile pada *_Z)**")
                st.dataframe(df_cut_aspect.round(3))

    # ====== RINGKASAN OPSIONAL (di LUAR expander, pakai checkbox) ======
    show_stats = st.checkbox("Tampilkan ringkasan WB_Index_Z per label (preview)", value=False, key="show_disc_stats")
    if show_stats:
        def _label_by_z(x, lo, hi):
            if not np.isfinite(x): return "Menengah"
            if x <= lo: return "Tidak Sejahtera"
            if x >= hi: return "Sejahtera"
            return "Menengah"
        def _label_by_q(x, lo, hi):
            if not np.isfinite(x): return "Menengah"
            if x <= lo: return "Tidak Sejahtera"
            if x >= hi: return "Sejahtera"
            return "Menengah"

        work2 = st.session_state.df_work.copy()
        # hitung WB_Index_Z ringan (jika belum ada)
        zcols = [c for c in work2.columns if c.endswith("_Z")]
        if "WB_Index_Z" not in work2.columns and zcols:
            work2["WB_Index_Z"] = pd.DataFrame({c: pd.to_numeric(work2[c], errors="coerce") for c in zcols}).mean(axis=1)

        if st.session_state.get("disc_method", "Z-score (manual threshold)").startswith("Z-score"):
            lo = float(st.session_state.get("disc_z_low", -0.5))
            hi = float(st.session_state.get("disc_z_high", 0.5))
            labs = work2["WB_Index_Z"].apply(lambda v: _label_by_z(float(v), lo, hi))
        else:
            # ambil cutoff yang dihitung di atas: kalau mau aman, hitung ulang singkat
            sWB = pd.to_numeric(work2["WB_Index_Z"], errors="coerce").dropna()
            q_low = float(st.session_state.get("disc_q_low", 0.33))
            q_high = float(st.session_state.get("disc_q_high", 0.67))
            loWB = float(sWB.quantile(q_low)) if len(sWB) else np.nan
            hiWB = float(sWB.quantile(q_high)) if len(sWB) else np.nan
            labs = work2["WB_Index_Z"].apply(lambda v: _label_by_q(float(v), loWB, hiWB))

        prev_tbl = pd.DataFrame({"WB_Index_Z": work2["WB_Index_Z"], "Label": labs})
        st.dataframe(prev_tbl.groupby("Label")["WB_Index_Z"].agg(["count","mean","std"]).round(3))

    # ── Tombol jalan
    if st.button("🔖 Buat Label", key="btn_disc_make"):
        try:
            if method_disc.startswith("Z-score") and DD and hasattr(DD, "discretize_pipeline_z"):
                # gunakan fungsi modul jika ada (tetap seperti sebelumnya)
                df_new, dlog = DD.discretize_pipeline_z(
                    df, aspects=aspects, low_thr=low_thr, high_thr=high_thr,
                )
            else:
                # ==== FALLBACK lokal (anti-NaN) ====
                def _num(s): return pd.to_numeric(s, errors="coerce")
                def _find_items(dfin, a): return [c for c in dfin.columns if str(c).startswith(f"{a}_")]
                def _ensure_aspect_means_local(dfin, aspects_):
                    out = dfin.copy()
                    for a in aspects_:
                        if a not in out.columns:
                            items = _find_items(out, a)
                            if len(items) >= 2:
                                out[a] = out[items].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                            else:
                                out[a] = np.nan
                    return out
                def _ensure_aspect_z_local(dfin, aspects_):
                    out = _ensure_aspect_means_local(dfin, aspects_).copy()
                    for a in aspects_:
                        zc = f"{a}_Z"
                        if zc not in out.columns:
                            x = _num(out[a])
                            mu = float(np.nanmean(x)); sd = float(np.nanstd(x, ddof=0))
                            x_filled = x.fillna(mu if np.isfinite(mu) else 0.0)
                            if not np.isfinite(sd) or sd == 0:
                                out[zc] = 0.0
                            else:
                                out[zc] = (x_filled - (mu if np.isfinite(mu) else 0.0)) / sd
                    return out

                work = _ensure_aspect_z_local(df.copy(), aspects)
                zcols = [f"{a}_Z" for a in aspects]
                Z = work[zcols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

                if method_disc.startswith("Z-score"):
                    lo, hi = float(low_thr), float(high_thr)
                    def _lab_z(v):
                        try: x = float(v)
                        except: return "Menengah"
                        if not np.isfinite(x): return "Menengah"
                        if x <= lo: return "Tidak Sejahtera"
                        if x >= hi: return "Sejahtera"
                        return "Menengah"
                    for a in aspects:
                        work[f"Label_{a}"] = work[f"{a}_Z"].apply(_lab_z)
                    work["WB_Index_Z"] = Z.mean(axis=1)
                    work["WB_Label"] = work["WB_Index_Z"].apply(_lab_z)
                    dlog = "[Fallback] Discretization Z selesai (anti-NaN)."

                else:
                    # Data-driven Quantile: ambang per-aspek dari quantile data pada *_Z
                    ql, qh = float(q_low), float(q_high)
                    qmap_low, qmap_high = {}, {}
                    for a in aspects:
                        s = _num(work[f"{a}_Z"]).dropna()
                        if len(s) == 0:
                            qmap_low[a], qmap_high[a] = -np.inf, np.inf
                        else:
                            qmap_low[a], qmap_high[a] = float(s.quantile(ql)), float(s.quantile(qh))
                    # label per-aspek
                    for a in aspects:
                        loA, hiA = qmap_low[a], qmap_high[a]
                        def _lab_q(v, loA=loA, hiA=hiA):
                            try: x = float(v)
                            except: return "Menengah"
                            if not np.isfinite(x): return "Menengah"
                            if x <= loA: return "Tidak Sejahtera"
                            if x >= hiA: return "Sejahtera"
                            return "Menengah"
                        work[f"Label_{a}"] = work[f"{a}_Z"].apply(_lab_q)
                    # WB Index + label berbasis quantile-nya sendiri
                    work["WB_Index_Z"] = Z.mean(axis=1)
                    sWB = _num(work["WB_Index_Z"]).dropna()
                    if len(sWB) == 0:
                        loWB, hiWB = -np.inf, np.inf
                    else:
                        loWB, hiWB = float(sWB.quantile(ql)), float(sWB.quantile(qh))
                    def _lab_q_wb(v, loWB=loWB, hiWB=hiWB):
                        try: x = float(v)
                        except: return "Menengah"
                        if not np.isfinite(x): return "Menengah"
                        if x <= loWB: return "Tidak Sejahtera"
                        if x >= hiWB: return "Sejahtera"
                        return "Menengah"
                    work["WB_Label"] = work["WB_Index_Z"].apply(_lab_q_wb)

                    dlog = f"[Fallback] Discretization Quantile selesai (q_low={ql}, q_high={qh})."

                df_new = work

            st.session_state.df_work = df_new
            log(dlog)
            st.success("Discretization selesai. Label per-aspek & WB_Label ditambahkan.")
        except Exception as e:
            st.error(f"Gagal membuat label: {e}")

    # Ringkasan label
    if st.session_state.df_work is not None:
        with st.expander("📊 Ringkasan Label", expanded=True):
            for c in ["WB_Label",
                      "Label_Education","Label_Financial","Label_Physical","Label_Psychological","Label_Relational"]:
                if c in st.session_state.df_work.columns:
                    st.write(f"**{c}**")
                    st.write(st.session_state.df_work[c].value_counts(dropna=False))

    preview_df(st.session_state.df_work, "Sesudah Discretization")





# 7) Log & Unduh
elif page == "7) Log & Unduh":
    st.subheader("🧾 Ringkasan Proses")

    # ---- LOG: tampilkan sebagai tabel sederhana + aksi opsional ----
    logs = st.session_state.get("logs", [])
    if not logs:
        st.info("Belum ada log proses.")
    else:
        # jadikan tabel agar mudah dibaca
        df_log = pd.DataFrame({"#": list(range(1, len(logs)+1)), "Log": logs})
        st.table(df_log)

        # aksi ringan (UI-only)
        col_log1, col_log2 = st.columns(2)
        with col_log1:
            if st.button("📋 Salin Semua Log", key="btn_copy_logs"):
                st.code("\n".join(logs), language="text")
        with col_log2:
            if st.button("🧹 Hapus Semua Log (UI)", key="btn_clear_logs"):
                st.session_state.logs = []
                st.success("Log dibersihkan (hanya UI, data tetap).")

    line()

    # ---- UNDuh DATASET ----
    st.subheader("📥 Unduh Dataset Hasil")

    df = st.session_state.get("df_work")
    if df is None or len(df) == 0:
        st.info("Tidak ada data untuk diunduh.")
    else:
        # ringkasannya dulu
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Baris", f"{df.shape[0]:,}")
        with col_b:
            st.metric("Kolom", f"{df.shape[1]:,}")
        with col_c:
            st.metric("Ukuran Memori", f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB")

        # pratinjau singkat
        with st.expander("👀 Pratinjau Data", expanded=False):
            n_preview = st.slider("Tampilkan n baris awal", 5, min(1000, max(5, len(df))), 20, 5, key="dl_preview_n")
            st.dataframe(df.head(n_preview))

        # opsi unduh
        st.markdown("**Format file:**")
        col_d1, col_d2 = st.columns(2)

        # CSV
        with col_d1:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Unduh CSV",
                data=csv_bytes,
                file_name="preprocessed.csv",
                mime="text/csv",
                key="dl_csv"
            )

        # Excel (.xlsx)
        with col_d2:
            try:
                xlsx_buf = io.BytesIO()
                # biarkan pandas pilih engine yang tersedia (openpyxl/xlsxwriter)
                with pd.ExcelWriter(xlsx_buf) as writer:
                    df.to_excel(writer, index=False, sheet_name="data")
                xlsx_buf.seek(0)
                st.download_button(
                    label="⬇️ Unduh Excel",
                    data=xlsx_buf,
                    file_name="preprocessed.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_xlsx"
                )
            except Exception as e:
                st.warning(f"Excel gagal dibuat ({e}). Pastikan paket openpyxl/xlsxwriter terpasang.")

        st.caption("Tips: CSV untuk interoperabilitas cepat; Excel enak untuk share & formatting.")


# 8) Visualisasi Clustering
elif page == "8) Visualisasi Clustering":
    st.subheader("Visualisasi Clustering (K-Means / DBSCAN / Hierarchical)")

    df = st.session_state.df_work
    preview_df(df, "Dataset Saat Ini")

    if VIS is None:
        st.error("Modul Visualisasi belum tersedia/terimport. Pastikan folder 'Visualisasi' berisi file Visualisasi.py dan modul algoritmanya.")
    else:
        try:
            df_new, vis_log = VIS.render(df)
            # Jika df_new berbeda (mis. user menempel label), simpan kembali
            if df is not None and df_new is not None and not df_new.equals(df):
                st.session_state.df_work = df_new
                st.success("Perubahan (label) telah diterapkan ke df_work.")
            # Catat log
            if vis_log:
                log(vis_log)
        except Exception as e:
            st.error(f"Gagal menjalankan Visualisasi: {e}")

    preview_df(st.session_state.df_work, "Setelah Visualisasi (jika label ditempel)")

# 9) Insight Cluster — baca rata-rata aspek per cluster dan beri ringkasan
elif page == "9) Insight Cluster":
    st.subheader("Insight dari Hasil Cluster")

    df = st.session_state.df_work
    preview_df(df, "Dataset Aktif (df_work)")

    if df is None or len(df) == 0:
        st.info("Belum ada data. Upload/siapkan data terlebih dahulu.")
    else:
        # --- Cari kolom label cluster ---
        label_candidates = [c for c in df.columns if c.lower().endswith("_label") or c.lower().startswith("cluster")]
        for c in ["KMeans_Label","DBSCAN_Label","Hierarchical_Label"]:
            if c in df.columns and c not in label_candidates:
                label_candidates.append(c)
        if not label_candidates:
            st.warning("Belum ada kolom label cluster. Jalankan K-Means/DBSCAN/Hierarchical dulu di halaman Visualisasi.")
        else:
            colA, colB = st.columns([1,1])
            with colA:
                label_col = st.selectbox("Pilih kolom label cluster", label_candidates, index=0, key="ins_label_col")
            with colB:
                basis = st.radio("Basis interpretasi", ["Z-score (direkomendasikan)", "Mean (1–5)"], index=0, key="ins_basis")

            base_aspects = ["Education","Financial","Physical","Psychological","Relational"]
            use_z = basis.startswith("Z-score")

            if use_z and all((a + "_Z") in df.columns for a in base_aspects):
                aspect_cols = [a + "_Z" for a in base_aspects]
                pretty = {"Education_Z":"Akademik","Financial_Z":"Finansial","Physical_Z":"Fisik",
                          "Psychological_Z":"Psikologis","Relational_Z":"Relasional"}
                lo_default, hi_default = -0.5, 0.5
            else:
                # fallback pakai mean 1–5 bila kolom *_Z belum ada
                aspect_cols = [a for a in base_aspects if a in df.columns]
                pretty = {"Education":"Akademik","Financial":"Finansial","Physical":"Fisik",
                          "Psychological":"Psikologis","Relational":"Relasional"}
                lo_default, hi_default = 2.5, 3.5

            if len(aspect_cols) < 3:
                st.warning(f"Kolom aspek kurang. Ditemukan: {aspect_cols}. "
                           f"Pastikan sudah menjalankan Data Transformation atau menyediakan kolom *_Z.")
            else:
                c1, c2, c3 = st.columns([1,1,1])
                with c1:
                    low_thr = st.number_input("Ambang rendah", value=float(lo_default), step=0.1, key="ins_thr_low")
                with c2:
                    high_thr = st.number_input("Ambang tinggi", value=float(hi_default), step=0.1, key="ins_thr_high")
                with c3:
                    show_var = st.checkbox("Tampilkan varians dalam-cluster", value=False, key="ins_show_var")

                # --- Coba pakai modul helper bila tersedia ---
                try:
                    # prefer modul di Visualisasi/ClusterInsight.py
                    try:
                        from Visualisasi.ClusterInsight import cluster_profiles, label_clusters, cluster_variability
                        use_helper = True
                    except Exception:
                        use_helper = False

                    if use_helper:
                        prof, mu, sd = cluster_profiles(df, label_col, aspect_cols, use_zscore=use_z)
                        labels_tbl = label_clusters(prof, low_thr=low_thr, high_thr=high_thr, pretty_names=pretty)

                        from Visualisasi.ClusterInsight import cluster_counts, make_text_summaries

                        # Hitung jumlah anggota per cluster (urut sesuai index prof/labels_tbl)
                        order = labels_tbl.index.astype(str).tolist()
                        counts = cluster_counts(df, label_col, order=order)

                        # Nama basis untuk ditampilkan
                        basis_name = "Z-score" if use_z else "Mean (1–5)"

                        # Mapping pretty yang sama dengan di atas
                        pretty_map = {
                            "Education_Z":"Akademik","Financial_Z":"Finansial","Physical_Z":"Fisik",
                            "Psychological_Z":"Psikologis","Relational_Z":"Relasional",
                            "Education":"Akademik","Financial":"Finansial","Physical":"Fisik",
                            "Psychological":"Psikologis","Relational":"Relasional",
                        }

                        summaries = make_text_summaries(
                            prof, labels_tbl, counts,
                            pretty_names=pretty_map,
                            basis_name=basis_name
                        )

                        st.markdown("### 📝 Ringkasan Naratif per Cluster")
                        for item in summaries:
                            with st.expander(item["title"], expanded=False):
                                st.markdown(item["body"])

                    else:
                        # ---- Fallback internal singkat tanpa modul ----
                        def _num(dfin: pd.DataFrame) -> pd.DataFrame:
                            out = pd.DataFrame(index=dfin.index)
                            for c in dfin.columns:
                                out[c] = pd.to_numeric(dfin[c], errors="coerce")
                            return out

                        X = _num(df[aspect_cols])
                        labs = df[label_col].astype(str)
                        data = X.join(labs.rename("_lab_")).dropna(axis=0, how="any")
                        # jika basis Z: asumsikan sudah Z; kalau mean, tidak diubah
                        prof = data.groupby("_lab_")[aspect_cols].mean().sort_index()

                        def _bucket(v, lo, hi):
                            try:
                                x = float(v)
                            except Exception:
                                return "Menengah"
                            if x >= hi: return "Sejahtera"
                            if x <= lo: return "Tidak Sejahtera"
                            return "Menengah"

                        cat = prof.applymap(lambda v: _bucket(v, low_thr, high_thr))
                        def _sumrow(row):
                            pos = [pretty.get(k, k) for k, v in row.items() if v == "Sejahtera"]
                            neg = [pretty.get(k, k) for k, v in row.items() if v == "Tidak Sejahtera"]
                            mid = [pretty.get(k, k) for k, v in row.items() if v == "Menengah"]
                            parts = []
                            if pos: parts.append(" & ".join(pos) + " Sejahtera")
                            if neg: parts.append(" & ".join(neg) + " Rendah")
                            if not parts and mid: parts.append("Mayoritas Menengah")
                            return " | ".join(parts) if parts else "Campuran"
                        labels_tbl = prof.copy()
                        labels_tbl["__Ringkasan__"] = cat.apply(_sumrow, axis=1)
                        for a in aspect_cols:
                            labels_tbl[f"Label_{a}"] = cat[a]

                    st.markdown("### Profil & Ringkasan per Cluster")
                    st.dataframe(labels_tbl.round(3))

                    # Unduh CSV ringkasan
                    st.download_button(
                        "Unduh Insight Cluster",
                        data=labels_tbl.to_csv(index=True).encode("utf-8"),
                        file_name="cluster_insight.csv",
                        mime="text/csv",
                    )

                    # Varians dalam-cluster (opsional)
                    if show_var:
                        if use_helper:
                            var_tbl = cluster_variability(df, label_col, aspect_cols)
                        else:
                            # fallback varians
                            def _num(dfin): return dfin.apply(pd.to_numeric, errors="coerce")
                            X = _num(df[aspect_cols])
                            labs = df[label_col].astype(str)
                            data = X.join(labs.rename("_lab_")).dropna(axis=0, how="any")
                            var_tbl = data.groupby("_lab_")[aspect_cols].var(ddof=0).join(
                                data.groupby("_lab_").size().rename("n")
                            )
                        st.markdown("### Varians Dalam-Cluster (lebih kecil = lebih kompak)")
                        st.dataframe(var_tbl.round(3))

                    # Opsional: injeksi label manusiawi ke df_work
                    if st.checkbox("Tambahkan nama cluster (ringkasan) ke dataset", value=False, key="ins_add_human"):
                        df_new = df.copy()
                        name_map = labels_tbl["__Ringkasan__"].to_dict()
                        df_new["Cluster_Label_Human"] = df_new[label_col].astype(str).map(name_map)
                        st.session_state.df_work = df_new
                        st.success("Kolom Cluster_Label_Human ditambahkan ke df_work.")

                    st.caption("Catatan: basis Z-score membuat antar-aspek komparabel. Ubah ambang untuk sensitivitas label. "
                               "Untuk DBSCAN, label -1 (noise) akan dihitung sebagai satu kelompok bila ada.")

                except Exception as e:
                    st.error(f"Gagal membangun insight cluster: {e}")

# 10) Action Plan (Wisdom)
elif page == "10) Action Plan (Wisdom)":
    st.subheader("🧭 Rencana Aksi (Wisdom) — ringkasan per cluster (dengan label)")

    try:
        from Visualisasi.ClusterInsight import cluster_profiles, cluster_counts
        from Visualisasi.ClusterWisdom import derive_wisdom_from_profiles, make_cluster_labels
    except Exception as e:
        st.error(f"Gagal import modul: {e}")
        st.stop()

    df = st.session_state.df_work
    preview_df(df, "Dataset Aktif (df_work)")
    if df is None or len(df) == 0:
        st.info("Belum ada data.")
        st.stop()

    # pilih kolom label (tidak menulis balik)
    label_candidates = [c for c in df.columns if c.lower().endswith("_label") or c.lower().startswith("cluster")]
    for c in ["KMeans_Label","DBSCAN_Label","Hierarchical_Label","HC_Label"]:
        if c in df.columns and c not in label_candidates:
            label_candidates.append(c)
    if not label_candidates:
        st.warning("Belum ada kolom label cluster.")
        st.stop()
    label_col = st.selectbox("Kolom label cluster", label_candidates, index=0)

    # basis & aspek otomatis
    base = ["Education","Financial","Physical","Psychological","Relational"]
    if all((a+"_Z") in df.columns for a in base):
        use_z = True;  aspect_cols = [a+"_Z" for a in base]; basis_name = "Z-score"
    else:
        use_z = False; aspect_cols = [a for a in base if a in df.columns]; basis_name = "Mean (1–5)"
    if len(aspect_cols) < 3:
        st.warning(f"Kolom aspek kurang. Ditemukan: {aspect_cols}")
        st.stop()

    # profil & jumlah (Insight)
    prof, mu, sd = cluster_profiles(df, label_col, aspect_cols, use_zscore=use_z)
    order = prof.index.astype(str).tolist()
    counts = cluster_counts(df, label_col, order=order)

    # (opsional) override label_map kalau kamu punya label manual; default: auto-label
    label_map = make_cluster_labels(prof, basis_name, aspect_cols)

    # derive wisdom + markdown
    from Visualisasi.ClusterWisdom import derive_wisdom_from_profiles as _derive
    wisdom, md_text = _derive(prof, counts, basis_name, aspect_cols, label_map=label_map)

    # render kartu rapi
    render_wisdom_cards(wisdom, prof, basis_name)

    # unduh markdown naratif
    st.download_button("📝 Unduh Ringkasan (Markdown)",
                       data=md_text.encode("utf-8"),
                       file_name="ringkasan_per_cluster.md",
                       mime="text/markdown")

    st.caption("Sistem memilih 1 penyebab & 1 aksi utama berbasis skor (gap × dampak ÷ biaya × n^0.5). Chip: hijau=tinggi, merah=rendah, abu=netral. Tidak mengubah df_work.")

