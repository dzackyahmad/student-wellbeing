# Visualisasi/Visualisasi.py
# Controller Streamlit untuk visualisasi clustering:
# - KMeans_Visual (manual, tanpa sklearn)
# - DBSCAN_Visual (manual, tanpa sklearn)
# - Hierarchical_Visual (scipy)
# Kompatibel Streamlit 1.11.0

from __future__ import annotations
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Import modul algoritma (dengan fallback) ---
try:
    from .KMeans_Visual import run_kmeans
    from .DBSCAN_Visual import run_dbscan
    from .Hierarchical_Visual import run_hierarchical
except Exception:
    from KMeans_Visual import run_kmeans
    from DBSCAN_Visual import run_dbscan
    from Hierarchical_Visual import run_hierarchical

# ---------------- Utils kolom ----------------
def numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def _default_axes(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = df.columns.tolist()
    for a, b in [("PC1","PC2"), ("UMAP1","UMAP2")]:
        if a in cols and b in cols:
            return a, b
    nums = numeric_columns(df)
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None, None

# ---------------- Metrik klaster ----------------
def _safe_numeric_array(X: pd.DataFrame) -> np.ndarray:
    df = pd.DataFrame(X)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df[num_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any").values

def compute_cluster_metrics(
    X: pd.DataFrame,
    labels: np.ndarray,
    algo: str,
    sample_size: int = 2000,
    random_state: int = 42
) -> Dict[str, Optional[float]]:
    """
    Hitung Silhouette, Calinski-Harabasz, Davies-Bouldin secara aman.
    - Untuk DBSCAN: buang label noise (-1) sebelum hitung metrik.
    - Butuh >= 2 cluster unik dan cukup sampel.
    - Silhouette di-subsample jika data besar supaya cepat.
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    X = _safe_numeric_array(pd.DataFrame(X))
    labels = np.asarray(labels)

    mask = np.ones(len(labels), dtype=bool)
    if algo.lower() == "dbscan":
        mask = labels != -1

    X_use = X[mask]
    y_use = labels[mask]
    uniq = np.unique(y_use)

    out = {"n_clusters": int(len(uniq)), "silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    if len(uniq) < 2 or X_use.shape[0] < 2:
        return out

    # Silhouette (sampling bila besar)
    try:
        if X_use.shape[0] > sample_size:
            rs = np.random.RandomState(random_state)
            idx = rs.choice(X_use.shape[0], size=sample_size, replace=False)
            sil = silhouette_score(X_use[idx], y_use[idx], metric="euclidean")
        else:
            sil = silhouette_score(X_use, y_use, metric="euclidean")
        out["silhouette"] = float(sil)
    except Exception:
        pass

    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(X_use, y_use))
    except Exception:
        pass
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(X_use, y_use))
    except Exception:
        pass

    return out

# ---------------- Halaman utama (dipanggil dari app.py) ----------------
def render(df_work: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], str]:
    import streamlit as st

    # CSS ringan: card & perbesar input
    st.markdown("""
        <style>
        .card {background:#111418;border:1px solid #272b33;border-radius:14px;padding:16px 18px;margin:.4rem 0 1rem 0;}
        .card-title {font-weight:600;font-size:1.05rem;margin-bottom:.6rem;}
        div[data-testid="stNumberInput"] input, div[data-baseweb="select"] > div {font-size:1.05rem;}
        div.stButton > button {font-size:1.05rem;padding:.45rem .9rem;}
        </style>
    """, unsafe_allow_html=True)

    st.title("📈 Visualisasi Clustering")
    if df_work is None or len(df_work) == 0:
        st.info("Belum ada data. Siapkan data di menu sebelumnya.")
        return df_work, "Visualisasi dilewati: df_work kosong."

    st.markdown("Pilih dua kolom numerik (mis. hasil PCA/UMAP) untuk scatter 2D.")
    cols_num = numeric_columns(df_work)
    if len(cols_num) < 2:
        st.warning("Kolom numerik < 2. Tambahkan PCA/UMAP atau pilih kolom numerik lain.")
        return df_work, "Visualisasi gagal: kolom numerik < 2."

    default_x, default_y = _default_axes(df_work)
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        x_col = st.selectbox("Sumbu X", cols_num, index=(cols_num.index(default_x) if default_x in cols_num else 0), key="vis_x")
    with c2:
        y_col = st.selectbox("Sumbu Y", cols_num, index=(cols_num.index(default_y) if default_y in cols_num else (1 if len(cols_num)>1 else 0)), key="vis_y")
    with c3:
        method = st.selectbox("Metode", ["K-Means","DBSCAN","Hierarchical"], index=0, key="vis_method")

    st.markdown("---")

    log_msg = ""
    df_out = df_work.copy()

    # ===================== PARAMETER PER METODE ======================
    if method == "K-Means":
        cA, cB, cC, cD = st.columns([1,1,1,1])
        with cA:
            k = st.number_input("k (cluster)", 1, 50, 3, 1, key="km_k")
        with cB:
            init = st.selectbox("init", ["kmeans++","random"], index=0, key="km_init")
        with cC:
            max_iter = st.number_input("max_iter", 10, 2000, 200, 10, key="km_maxit")
        with cD:
            tol = st.number_input("tol", 1e-8, 1.0, 1e-4, format="%.6f", key="km_tol")
        cE, cF = st.columns([1,1])
        with cE:
            seed = st.number_input("random_state", 0, 10000, 42, 1, key="km_seed")
        with cF:
            do_metrics = st.checkbox("Hitung metrik (Sil/CH/DBI)", value=True, key="km_mets")
        run_k = st.button("▶️ Jalankan K-Means", key="btn_run_km")
        st.markdown('</div>', unsafe_allow_html=True)

        if run_k:
            try:
                res = run_kmeans(
                    df_out, x_col=x_col, y_col=y_col,
                    k=int(k), init=init, max_iter=int(max_iter),
                    tol=float(tol), random_state=int(seed),
                    compute_silhouette=False  # kita hitung metrik sendiri
                )
                st.pyplot(res["figure"])

                # Metrik
                if do_metrics:
                    st.markdown('<div class="card"><div class="card-title">⚙️ Parameter K-Means</div>', unsafe_allow_html=True)
                    X = df_out[[x_col, y_col]]
                    labels = res["labels"].values if hasattr(res["labels"], "values") else res["labels"]
                    mets = compute_cluster_metrics(X, labels, algo="kmeans", random_state=int(seed))
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Silhouette ↑", "n/a" if mets["silhouette"] is None else f"{mets['silhouette']:.4f}")
                    m2.metric("Calinski–Harabasz ↑", "n/a" if mets["calinski_harabasz"] is None else f"{mets['calinski_harabasz']:.2f}")
                    m3.metric("Davies–Bouldin ↓", "n/a" if mets["davies_bouldin"] is None else f"{mets['davies_bouldin']:.4f}")
                st.caption(f"Inertia (SSE): {res['inertia']:.4f}")

                with st.expander("📌 Centroids"):
                    st.write(res["centroids"])
                with st.expander("🔎 Labels (head)"):
                    st.write(pd.Series(res["labels"]).head())

                if st.checkbox("Tempel label ke df (kolom: KMeans_Label)", value=True, key="km_apply"):
                    df_out.loc[pd.Index(df_out.index), "KMeans_Label"] = np.asarray(res["labels"]).astype(int)
                    st.success("Label KMeans_Label ditambahkan ke df_work.")
                log_msg = f"[Visual] K-Means: k={k}, init={init}, max_iter={max_iter}, tol={tol}, x={x_col}, y={y_col}"
            except Exception as e:
                st.error(f"Gagal K-Means: {e}")
                log_msg = f"[Visual][ERR] K-Means: {e}"

    elif method == "DBSCAN":
        cA, cB, cC = st.columns([1,1,1])
        with cA:
            eps = st.number_input("eps", 0.01, 100.0, 0.6, 0.05, key="db_eps")
        with cB:
            min_samples = st.number_input("minPts (min_samples)", 1, 200, 5, 1, key="db_min")
        with cC:
            metric = st.selectbox("metric", ["euclidean","manhattan","cosine","hamming"], index=0, key="db_metric")
        do_metrics = st.checkbox("Hitung metrik (abaikan noise)", value=True, key="db_mets")
        run_db = st.button("▶️ Jalankan DBSCAN", key="btn_run_db")
        st.markdown('</div>', unsafe_allow_html=True)

        if run_db:
            try:
                res = run_dbscan(
                    df_out, x_col=x_col, y_col=y_col,
                    eps=float(eps), min_samples=int(min_samples),
                    metric=str(metric), compute_silhouette=False
                )
                st.pyplot(res["figure"])

                if do_metrics:
                    st.markdown('<div class="card"><div class="card-title">⚙️ Parameter DBSCAN</div>', unsafe_allow_html=True)
                    X = df_out[[x_col, y_col]]
                    labels = res["labels"].values if hasattr(res["labels"], "values") else res["labels"]
                    mets = compute_cluster_metrics(X, labels, algo="dbscan")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Silhouette ↑", "n/a" if mets["silhouette"] is None else f"{mets['silhouette']:.4f}")
                    m2.metric("Calinski–Harabasz ↑", "n/a" if mets["calinski_harabasz"] is None else f"{mets['calinski_harabasz']:.2f}")
                    m3.metric("Davies–Bouldin ↓", "n/a" if mets["davies_bouldin"] is None else f"{mets['davies_bouldin']:.4f}")
                st.caption(f"Clusters terdeteksi (tanpa noise): {res.get('n_clusters','?')}")

                with st.expander("🔎 Labels (head)"):
                    st.write(pd.Series(res["labels"]).head())
                with st.expander("Core Mask (head)"):
                    cm = res.get("core_mask")
                    if cm is not None:
                        st.write(pd.Series(cm).head())

                if st.checkbox("Tempel label ke df (kolom: DBSCAN_Label, DBSCAN_is_core)", value=True, key="db_apply"):
                    df_out.loc[pd.Index(df_out.index), "DBSCAN_Label"] = np.asarray(res["labels"]).astype(int)
                    if res.get("core_mask") is not None:
                        df_out.loc[pd.Index(df_out.index), "DBSCAN_is_core"] = np.asarray(res["core_mask"]).astype(bool)
                    st.success("Kolom DBSCAN_Label & DBSCAN_is_core ditambahkan.")
                log_msg = f"[Visual] DBSCAN: eps={eps}, minPts={min_samples}, metric={metric}, x={x_col}, y={y_col}"
            except Exception as e:
                st.error(f"Gagal DBSCAN: {e}")
                log_msg = f"[Visual][ERR] DBSCAN: {e}"

    else:  # Hierarchical
        cA, cB, cC = st.columns([1,1,1])
        with cA:
            method_h = st.selectbox("method", ["ward","single","complete","average","weighted","centroid","median"], index=0, key="hc_method")
        with cB:
            metric_h = st.selectbox("metric", ["euclidean","cityblock","cosine","hamming","chebyshev"], index=0, key="hc_metric")
        with cC:
            cut_mode = st.radio("Pemotongan", ["maxclust (k)", "distance"], index=0, key="hc_cut")
        cD, cE, cF = st.columns([1,1,1])
        with cD:
            if cut_mode.startswith("maxclust"):
                k = st.number_input("k (maxclust)", 1, 50, 3, 1, key="hc_k")
                dist_thr = None
            else:
                dist_thr = st.number_input("distance threshold", 0.01, 10_000.0, 5.0, 0.1, key="hc_thr")
                k = None
        with cE:
            trunc_p = st.number_input("truncate dendrogram: last p (0=off)", 0, 500, 30, 1, key="hc_trunc")
        with cF:
            do_metrics = st.checkbox("Hitung metrik", value=True, key="hc_mets")
        run_h = st.button("▶️ Jalankan Hierarchical", key="btn_run_hc")
        st.markdown('</div>', unsafe_allow_html=True)

        if run_h:
            try:
                res = run_hierarchical(
                    df_out, x_col=x_col, y_col=y_col,
                    method=method_h, metric=metric_h,
                    n_clusters=int(k) if k is not None else None,
                    distance_threshold=float(dist_thr) if dist_thr is not None else None,
                    compute_silhouette=False,
                    truncate_dendro_lastp=int(trunc_p) if trunc_p > 0 else None,
                )
                st.pyplot(res["figure_scatter"])
                st.pyplot(res["figure_dendrogram"])

                if do_metrics:
                    st.markdown('<div class="card"><div class="card-title">⚙️ Parameter Hierarchical</div>', unsafe_allow_html=True)
                    X = df_out[[x_col, y_col]]
                    labels = res["labels"].values if hasattr(res["labels"], "values") else res["labels"]
                    mets = compute_cluster_metrics(X, labels, algo="hierarchical")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Silhouette ↑", "n/a" if mets["silhouette"] is None else f"{mets['silhouette']:.4f}")
                    m2.metric("Calinski–Harabasz ↑", "n/a" if mets["calinski_harabasz"] is None else f"{mets['calinski_harabasz']:.2f}")
                    m3.metric("Davies–Bouldin ↓", "n/a" if mets["davies_bouldin"] is None else f"{mets['davies_bouldin']:.4f}")
                st.caption(f"Clusters: {res['n_clusters']}")

                with st.expander("🔎 Labels (head)"):
                    st.write(pd.Series(res["labels"]).head())

                if st.checkbox("Tempel label ke df (kolom: HC_Label)", value=True, key="hc_apply"):
                    df_out.loc[pd.Index(df_out.index), "HC_Label"] = np.asarray(res["labels"]).astype(int)
                    st.success("Kolom HC_Label ditambahkan.")
                log_msg = f"[Visual] HC: method={method_h}, metric={metric_h}, cut={'k='+str(k) if k else 'dist='+str(dist_thr)}, x={x_col}, y={y_col}"
            except Exception as e:
                st.error(f"Gagal Hierarchical: {e}")
                log_msg = f"[Visual][ERR] HC: {e}"

    return df_out, log_msg

# ----------------------- Demo mandiri (opsional) --------------------
def _demo_streamlit():
    import streamlit as st
    st.set_page_config(page_title="Visualisasi Clustering", layout="wide")

    st.sidebar.header("Demo Data")
    n = st.sidebar.number_input("n samples", 50, 2000, 600, 50)
    seed = st.sidebar.number_input("random_state", 0, 10000, 42, 1)

    rng = np.random.default_rng(int(seed))
    A = rng.normal(loc=[0, 0], scale=[1.0, 1.0], size=(n//3, 2))
    B = rng.normal(loc=[5, 5], scale=[1.0, 1.0], size=(n//3, 2))
    C = rng.normal(loc=[0, 6], scale=[1.1, 1.1], size=(n - 2*(n//3), 2))
    X = np.vstack([A, B, C])
    df = pd.DataFrame(X, columns=["PC1", "PC2"])  # pakai nama PC1/PC2 agar auto-terpilih

    df_out, log_msg = render(df)
    st.sidebar.markdown("---")
    st.sidebar.write("Log:", log_msg)
    with st.expander("Data (head)"):
        st.write(df_out.head())

if __name__ == "__main__":
    try:
        _demo_streamlit()
    except Exception as e:
        print("Untuk demo: streamlit run Visualisasi/Visualisasi.py")
        print("Error:", e)
