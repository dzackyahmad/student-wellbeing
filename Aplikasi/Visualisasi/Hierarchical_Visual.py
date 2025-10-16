# Hierarchical_Visual.py
# Visualisasi Hierarchical Clustering (SciPy linkage, fcluster, dendrogram) + plotting
# Kompatibel Streamlit 1.11.0, tidak memakai sklearn.

from __future__ import annotations
from typing import Tuple, Optional, Dict, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform

MethodT = Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"]

# -------------------------------------------------------------------
# Util: ekstraksi 2 kolom numerik
# -------------------------------------------------------------------

def _ensure_numeric_2cols(
    df: pd.DataFrame, x_col: str, y_col: str
) -> Tuple[np.ndarray, pd.Index, pd.DataFrame]:
    """
    Ambil dua kolom numerik dari df, drop NaN pada pasangan (x,y).
    Return:
      - X: ndarray (n_samples, 2)
      - used_idx: index baris yang dipakai
      - df_xy: df dua kolom yang sudah bersih (align index)
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Kolom {x_col} / {y_col} tidak ditemukan.")
    df_xy = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if df_xy.shape[0] == 0:
        raise ValueError("Tidak ada baris valid untuk pasangan (x,y).")
    return df_xy.values.astype(float), df_xy.index, df_xy


# -------------------------------------------------------------------
# Silhouette sederhana (tanpa sklearn)
# -------------------------------------------------------------------

def silhouette_score_simple(
    X: np.ndarray, labels: np.ndarray, sample: int = 1000, random_state: Optional[int] = 42
) -> Optional[float]:
    """
    Silhouette rata-rata (euclidean). Jika cluster unik < 2 → None.
    Sampling untuk efisiensi pada data besar.
    """
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return None
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    if n > sample:
        idx = rng.choice(n, size=sample, replace=False)
        Xs = X[idx]
        ls = labels[idx]
    else:
        Xs = X
        ls = labels

    m = Xs.shape[0]
    D = np.zeros((m, m), dtype=float)
    for i in range(m):
        dif = Xs - Xs[i]
        D[i] = np.sqrt(np.sum(dif * dif, axis=1))

    s_vals = []
    for i in range(m):
        same = (ls == ls[i])
        other = ~same
        if same.sum() > 1:
            a = D[i, same].sum() / (same.sum() - 1)
        else:
            a = 0.0
        b = np.inf
        for cl in np.unique(ls[other]):
            mask = (ls == cl)
            b = min(b, D[i, mask].mean())
        s = 0.0 if (a == 0.0 and b == 0.0) else (b - a) / max(a, b)
        s_vals.append(s)

    return float(np.mean(s_vals)) if s_vals else None


# -------------------------------------------------------------------
# API tingkat DataFrame untuk controller
# -------------------------------------------------------------------

def run_hierarchical(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    method: MethodT = "ward",
    metric: str = "euclidean",
    n_clusters: Optional[int] = 3,
    distance_threshold: Optional[float] = None,
    compute_silhouette: bool = True,
    truncate_dendro_lastp: Optional[int] = None,   # contoh: 30 → ringkas dendrogram
) -> Dict[str, object]:
    """
    Jalankan hierarchical clustering:
    - Buat linkage dengan SciPy
    - Potong cluster via n_clusters (maxclust) atau distance_threshold
    - Kembalikan labels (0-based), linkage matrix, scatter figure, dendrogram figure

    Catatan:
    - 'ward' memaksa metrik euclidean pada jarak antar-merge (memang definisinya).
    - Jika n_clusters diberikan → diutamakan; jika None → gunakan distance_threshold.
    - Jika keduanya None → default n_clusters=3.
    """
    X, used_idx, df_xy = _ensure_numeric_2cols(df, x_col, y_col)

    # linkage: bisa langsung pakai matrix observasi (SciPy akan hitung pdist)
    # method 'ward' otomatis gunakan euclidean; untuk method lain, argumen metric dipakai.
    if method == "ward":
        Z = linkage(X, method="ward")
    else:
        Z = linkage(X, method=method, metric=metric)

    # Cut tree → labels (fcluster memberi label 1..K). Ubah ke 0..K-1 konsisten dengan modul lain.
    if n_clusters is not None:
        lab = fcluster(Z, t=int(n_clusters), criterion="maxclust")
    elif distance_threshold is not None:
        lab = fcluster(Z, t=float(distance_threshold), criterion="distance")
    else:
        lab = fcluster(Z, t=3, criterion="maxclust")

    # re-map ke 0..K-1 secara stabil (urut berdasarkan label yang muncul)
    uniq = np.unique(lab)
    remap = {u: i for i, u in enumerate(uniq)}
    labels0 = np.vectorize(remap.get)(lab)

    # Metrik evaluasi (opsional)
    sil = silhouette_score_simple(X, labels0) if compute_silhouette else None

    # Gambar scatter
    fig_sc = make_scatter_figure_hc(
        df_xy, x_col, y_col, labels0, title=f"Hierarchical ({method}, metric={metric})"
    )

    # Gambar dendrogram
    fig_den = make_dendrogram_figure(
        Z, title=f"Dendrogram ({method})", truncate_lastp=truncate_dendro_lastp
    )

    s_labels = pd.Series(labels0.astype(int), index=used_idx, name="HC_Label")

    return {
        "labels": s_labels,
        "linkage": Z,
        "n_clusters": int(len(np.unique(labels0))),
        "silhouette": sil,
        "figure_scatter": fig_sc,
        "figure_dendrogram": fig_den,
        "used_index": used_idx,
    }


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

def make_scatter_figure_hc(
    df_xy: pd.DataFrame,
    x_col: str,
    y_col: str,
    labels: np.ndarray,
    title: str = "Hierarchical Scatter",
) -> plt.Figure:
    """
    Scatter 2D diwarnai berdasarkan label cluster (0..K-1).
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(df_xy[x_col].values, df_xy[y_col].values, c=labels, s=20, alpha=0.9)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    return fig


def make_dendrogram_figure(
    Z: np.ndarray,
    title: str = "Dendrogram",
    truncate_lastp: Optional[int] = None,
) -> plt.Figure:
    """
    Buat dendrogram dari linkage matrix.
    - truncate_lastp: jika diberikan, gunakan 'lastp' untuk ringkas cabang (lebih cepat & bersih).
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    kwargs = {}
    if truncate_lastp is not None and truncate_lastp > 0:
        kwargs.update(dict(truncate_mode="lastp", p=int(truncate_lastp)))
    dendrogram(Z, ax=ax, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Observations (or merged leaves)")
    ax.set_ylabel("Distance")
    ax.grid(True, linestyle="--", alpha=0.15)
    return fig


# -------------------------------------------------------------------
# Demo mandiri (opsional)
# -------------------------------------------------------------------

def _demo_streamlit():
    """
    Jalankan cepat:
      streamlit run Hierarchical_Visual.py
    """
    import streamlit as st

    st.set_page_config(page_title="Hierarchical Demo", layout="wide")
    st.title("Hierarchical Clustering — Demo")

    # Data dummy 3 cluster
    rng = np.random.default_rng(9)
    A = rng.normal(loc=[0, 0], scale=[0.9, 0.9], size=(120, 2))
    B = rng.normal(loc=[4.0, 4.5], scale=[0.8, 0.8], size=(120, 2))
    C = rng.normal(loc=[-3.5, 5.0], scale=[0.8, 0.8], size=(120, 2))
    X = np.vstack([A, B, C])
    df = pd.DataFrame(X, columns=["X1", "X2"])

    colL, colR = st.columns([1, 1])
    with colL:
        x_col = st.selectbox("X axis", df.columns.tolist(), index=0)
        y_col = st.selectbox("Y axis", df.columns.tolist(), index=1)
        method = st.selectbox("method", ["ward","single","complete","average","weighted","centroid","median"], index=0)
        metric = st.selectbox("metric", ["euclidean","cityblock","cosine","hamming","chebyshev"], index=0)
        cut_mode = st.radio("Pemotongan", ["maxclust (k)", "distance"], index=0)
    with colR:
        truncate_p = st.number_input("truncate dendrogram last p (0=off)", 0, 200, 30, 1)
        comp_sil = st.checkbox("Hitung silhouette", value=True)

        if cut_mode.startswith("maxclust"):
            k = st.number_input("k (cluster)", 1, 20, 3, 1)
            tdist = None
        else:
            tdist = st.number_input("distance threshold", 0.01, 10_000.0, 5.0, 0.1)
            k = None

    if st.button("Run Hierarchical"):
        res = run_hierarchical(
            df, x_col=x_col, y_col=y_col,
            method=method, metric=metric,
            n_clusters=int(k) if k is not None else None,
            distance_threshold=float(tdist) if tdist is not None else None,
            compute_silhouette=bool(comp_sil),
            truncate_dendro_lastp=int(truncate_p) if truncate_p > 0 else None,
        )
        st.pyplot(res["figure_scatter"])
        st.pyplot(res["figure_dendrogram"])
        st.write(f"Clusters: {res['n_clusters']}")
        if res["silhouette"] is not None:
            st.write(f"Silhouette (sampled): {res['silhouette']:.4f}")
        with st.expander("Labels (head)"):
            st.write(res["labels"].head())


if __name__ == "__main__":
    try:
        _demo_streamlit()
    except Exception as e:
        print("Untuk demo: streamlit run Hierarchical_Visual.py")
        print("Error:", e)
