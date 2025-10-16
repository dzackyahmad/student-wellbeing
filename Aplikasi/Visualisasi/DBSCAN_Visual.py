# DBSCAN_Visual.py
# Implementasi DBSCAN manual (tanpa sklearn) + plotting Matplotlib
# Kompatibel Streamlit 1.11.0 dan dapat dipanggil langsung untuk demo.

from __future__ import annotations
from typing import Tuple, Optional, Dict, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
      - df_xy: df dua kolom yang sudah bersih
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Kolom {x_col} / {y_col} tidak ditemukan.")
    df_xy = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if df_xy.shape[0] == 0:
        raise ValueError("Tidak ada baris valid untuk pasangan (x,y).")
    X = df_xy.values.astype(float)
    return X, df_xy.index, df_xy


# -------------------------------------------------------------------
# Jarak & tetangga
# -------------------------------------------------------------------

def _pairwise_dists(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Matriks jarak penuh O(n^2) untuk n <= ~20k (praktis untuk visualisasi).
    metric: 'euclidean' | 'manhattan' | 'cosine' | 'hamming'
    """
    metric = metric.lower()
    n, d = X.shape
    if metric == "euclidean":
        # ||x-y|| = sqrt(||x||^2 + ||y||^2 - 2 x.y)
        x2 = np.sum(X * X, axis=1, keepdims=True)       # (n,1)
        d2 = np.maximum(x2 + x2.T - 2.0 * (X @ X.T), 0) # (n,n) no negatives
        return np.sqrt(d2, out=d2)                      # re-use memory

    elif metric == "manhattan":
        # |x-y|_1 = sum(|x_i - y_i|)
        # Vectorized via broadcasting; O(n^2*d) — cukup untuk visualisasi
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            dif = np.abs(X - X[i])
            D[i] = np.sum(dif, axis=1)
        return D

    elif metric == "cosine":
        # cosine distance = 1 - (x·y)/(||x|| ||y||)
        # normalisasi baris
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        S = Xn @ Xn.T
        S = np.clip(S, -1.0, 1.0)
        return 1.0 - S

    elif metric == "hamming":
        # proporsi elemen yang berbeda (untuk 2D numeric → efektif: (x!=y).mean())
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            neq = (X != X[i]).astype(float)
            D[i] = np.mean(neq, axis=1)
        return D

    else:
        raise ValueError(f"Metric tidak didukung: {metric}")


def _region_query_from_D(D: np.ndarray, i: int, eps: float) -> np.ndarray:
    """
    Beri indeks tetangga (termasuk i sendiri) dengan jarak <= eps.
    D: matriks jarak (n,n)
    """
    return np.where(D[i] <= eps)[0]


# -------------------------------------------------------------------
# DBSCAN core
# -------------------------------------------------------------------

def dbscan_fit(
    X: np.ndarray,
    eps: float,
    min_samples: int,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    DBSCAN tanpa sklearn.

    Parameters
    ----------
    X : ndarray (n, d)
    eps : float                radius neighborhood
    min_samples : int          minimum tetangga (termasuk titiknya sendiri)
    metric : str               'euclidean' | 'manhattan' | 'cosine' | 'hamming'

    Returns
    -------
    labels : ndarray (n,)      label cluster (0..C-1), noise = -1
    core_mask : ndarray (n,)   True untuk core points
    n_clusters : int           jumlah cluster ditemukan
    """
    n = X.shape[0]
    if eps <= 0.0:
        raise ValueError("eps harus > 0.")
    if min_samples < 1:
        raise ValueError("min_samples harus >= 1.")

    D = _pairwise_dists(X, metric=metric)

    labels = np.full(n, -1, dtype=int)   # -1 = noise (default)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    # Precompute neighbors per titik untuk efisiensi
    neighbors = [ _region_query_from_D(D, i, eps) for i in range(n) ]
    core_mask = np.array([len(neighbors[i]) >= min_samples for i in range(n)], dtype=bool)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        N_i = neighbors[i]
        if len(N_i) < min_samples:
            labels[i] = -1  # noise (sementara; bisa jadi border nanti)
            continue

        # Buat cluster baru
        labels[i] = cluster_id
        # seed set awal = semua tetangga
        seeds = list(N_i.tolist())
        # Proses BFS memperluas cluster
        k = 0
        while k < len(seeds):
            j = seeds[k]
            if not visited[j]:
                visited[j] = True
                N_j = neighbors[j]
                if len(N_j) >= min_samples:  # core
                    # gabungkan tetangga baru ke seeds
                    for t in N_j:
                        if t not in seeds:    # hindari duplikasi
                            seeds.append(int(t))
            # Assign label bila belum diberi label
            if labels[j] == -1:
                labels[j] = cluster_id
            k += 1

        cluster_id += 1

    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    return labels, core_mask, n_clusters


# -------------------------------------------------------------------
# Silhouette sederhana (abaikan noise)
# -------------------------------------------------------------------

def silhouette_score_simple_dbscan(
    X: np.ndarray, labels: np.ndarray, sample: int = 1000, random_state: Optional[int] = 42
) -> Optional[float]:
    """
    Silhouette rata-rata untuk DBSCAN:
    - Abaikan titik noise (label = -1)
    - Jika cluster unik < 2 setelah buang noise → return None
    """
    valid = labels != -1
    if valid.sum() <= 1:
        return None
    L = labels[valid]
    Xv = X[valid]

    # Jika sisa cluster < 2 → tidak terdefinisi
    if len(np.unique(L)) < 2:
        return None

    rng = np.random.default_rng(random_state)
    n = Xv.shape[0]
    if n > sample:
        idx = rng.choice(n, size=sample, replace=False)
        Xv = Xv[idx]
        L = L[idx]

    m = Xv.shape[0]
    # matriks jarak euclidean untuk silhouette (praktik umum)
    D = np.zeros((m, m), dtype=float)
    for i in range(m):
        dif = Xv - Xv[i]
        D[i] = np.sqrt(np.sum(dif * dif, axis=1))

    s_vals = []
    for i in range(m):
        same = (L == L[i])
        other = ~same
        if same.sum() > 1:
            a = D[i, same].sum() / (same.sum() - 1)
        else:
            a = 0.0
        b = np.inf
        for cl in np.unique(L[other]):
            mask = (L == cl)
            b = min(b, D[i, mask].mean())
        s = 0.0 if (a == 0.0 and b == 0.0) else (b - a) / max(a, b)
        s_vals.append(s)

    return float(np.mean(s_vals)) if s_vals else None


# -------------------------------------------------------------------
# API tingkat DataFrame untuk controller
# -------------------------------------------------------------------

def run_dbscan(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    compute_silhouette: bool = True,
) -> Dict[str, object]:
    """
    Endpoint utama untuk Visualisasi controller:
    - Validasi dua kolom numerik
    - Jalankan DBSCAN
    - Kembalikan label (Series align dengan index asli), core_mask (Series),
      n_clusters, silhouette (opsional), dan Figure scatter.

    Return dict:
      {
        "labels": pd.Series[name="DBSCAN_Label"],
        "core_mask": pd.Series[name="DBSCAN_is_core"],
        "n_clusters": int,
        "silhouette": Optional[float],
        "figure": matplotlib.figure.Figure,
        "used_index": pd.Index
      }
    """
    X, used_idx, df_xy = _ensure_numeric_2cols(df, x_col, y_col)

    labels, core_mask, n_clusters = dbscan_fit(
        X, eps=float(eps), min_samples=int(min_samples), metric=str(metric).lower()
    )
    sil = silhouette_score_simple_dbscan(X, labels) if compute_silhouette else None

    s_labels = pd.Series(labels, index=used_idx, name="DBSCAN_Label")
    s_core = pd.Series(core_mask.astype(bool), index=used_idx, name="DBSCAN_is_core")

    fig = make_scatter_figure_dbscan(
        df_xy, x_col, y_col, labels=labels, core_mask=core_mask,
        title=f"DBSCAN (eps={eps}, min_samples={min_samples}, metric={metric})"
    )
    return {
        "labels": s_labels,
        "core_mask": s_core,
        "n_clusters": int(n_clusters),
        "silhouette": sil,
        "figure": fig,
        "used_index": used_idx,
    }


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

def make_scatter_figure_dbscan(
    df_xy: pd.DataFrame,
    x_col: str,
    y_col: str,
    labels: np.ndarray,
    core_mask: Optional[np.ndarray] = None,
    title: str = "DBSCAN Scatter",
) -> plt.Figure:
    """
    Scatter 2D:
    - Titik diwarnai berdasar label cluster (noise = -1)
    - Core vs border/noise diberi marker berbeda untuk membantu diagnosis
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    x = df_xy[x_col].values
    y = df_xy[y_col].values

    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    # Plot cluster per label agar noise bisa dibedakan markernya
    for lab in unique_labels:
        mask = labels == lab
        if lab == -1:
            # Noise → tanda berbeda (marker '.')
            ax.scatter(x[mask], y[mask], s=18, marker='.', alpha=0.7, label="noise")
        else:
            # Cluster
            if core_mask is not None:
                cm = np.asarray(core_mask)[mask]
                # core
                if cm.any():
                    ax.scatter(x[mask][cm], y[mask][cm], s=36, alpha=0.95, label=f"cluster {lab} (core)")
                # border
                if (~cm).any():
                    ax.scatter(x[mask][~cm], y[mask][~cm], s=18, alpha=0.85, marker='o', label=f"cluster {lab} (border)")
            else:
                ax.scatter(x[mask], y[mask], s=24, alpha=0.9, label=f"cluster {lab}")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    # Legend ringkas jika label banyak
    try:
        ax.legend(loc="best", fontsize=8, ncol=2)
    except Exception:
        pass
    return fig


# -------------------------------------------------------------------
# Demo mandiri (opsional)
# -------------------------------------------------------------------

def _demo_streamlit():
    """
    Jalankan:
      streamlit run DBSCAN_Visual.py
    untuk demo cepat tanpa controller.
    """
    import streamlit as st

    st.set_page_config(page_title="DBSCAN Demo", layout="wide")
    st.title("DBSCAN Manual — Demo")

    # Data dummy dengan noise
    rng = np.random.default_rng(7)
    A = rng.normal(loc=[0, 0], scale=[0.6, 0.6], size=(200, 2))
    B = rng.normal(loc=[4.5, 4.5], scale=[0.6, 0.6], size=(200, 2))
    N = rng.uniform(low=-2.5, high=6.5, size=(50, 2))  # noise
    X = np.vstack([A, B, N])
    df = pd.DataFrame(X, columns=["X1", "X2"])

    colL, colR = st.columns([1, 1])
    with colL:
        x_col = st.selectbox("X axis", df.columns.tolist(), index=0, key="db_x")
        y_col = st.selectbox("Y axis", df.columns.tolist(), index=1, key="db_y")
        eps = st.number_input("eps", min_value=0.01, max_value=10.0, value=0.6, step=0.05, format="%.2f")
        min_samples = st.number_input("min_samples", min_value=1, max_value=100, value=5, step=1)
    with colR:
        metric = st.selectbox("metric", ["euclidean", "manhattan", "cosine", "hamming"], index=0)
        comp_sil = st.checkbox("Hitung silhouette (abaikan noise)", value=True)

    if st.button("Run DBSCAN"):
        res = run_dbscan(
            df, x_col=x_col, y_col=y_col,
            eps=float(eps), min_samples=int(min_samples),
            metric=str(metric), compute_silhouette=bool(comp_sil)
        )
        st.pyplot(res["figure"])
        st.write(f"Clusters: {res['n_clusters']}")
        if res["silhouette"] is not None:
            st.write(f"Silhouette (sampled, non-noise): {res['silhouette']:.4f}")
        with st.expander("Labels (head)"):
            st.write(res["labels"].head())
        with st.expander("Core mask (head)"):
            st.write(res["core_mask"].head())


if __name__ == "__main__":
    try:
        _demo_streamlit()
    except Exception as e:
        print("Untuk demo: streamlit run DBSCAN_Visual.py")
        print("Error:", e)
