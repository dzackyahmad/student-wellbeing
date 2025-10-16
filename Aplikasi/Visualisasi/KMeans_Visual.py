# KMeans_Visual.py
# Manual K-Means (tanpa sklearn) + plotting Matplotlib
# Kompatibel Streamlit 1.11.0, bisa dipanggil langsung untuk testing/demo.

from __future__ import annotations
from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- Util Numerik ---------------------------

def _ensure_numeric_2cols(
    df: pd.DataFrame, x_col: str, y_col: str
) -> Tuple[np.ndarray, pd.Index, pd.DataFrame]:
    """
    Ambil dua kolom numerik dari df, drop NaN pada pasangan (x,y),
    return:
      - X: ndarray shape (n_samples, 2)
      - idx: index baris yang dipakai
      - df_xy: df[x_col, y_col] yang sudah dibersihkan (untuk join label)
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Kolom {x_col} / {y_col} tidak ditemukan di DataFrame.")

    df_xy = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce")
    df_xy = df_xy.dropna(axis=0, how="any")
    if df_xy.shape[0] == 0:
        raise ValueError("Tidak ada baris valid (semua NaN/invalid) pada pasangan (x,y).")

    X = df_xy.values.astype(float)
    return X, df_xy.index, df_xy


def _squared_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Jarak kuadrat Euclidean antara dua vektor 1D."""
    d = a - b
    return float(np.dot(d, d))


def _pairwise_sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Hitung matriks jarak kuadrat dari setiap titik X ke setiap centroid C.
    X: (n, d), C: (k, d) -> out: (n, k)
    """
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c
    x2 = np.sum(X * X, axis=1, keepdims=True)          # (n,1)
    c2 = np.sum(C * C, axis=1, keepdims=True).T        # (1,k)
    xc = X @ C.T                                       # (n,k)
    return x2 + c2 - 2.0 * xc


# ------------------------- Inisialisasi -----------------------------

def _init_random(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pilih k titik acak dari X sebagai centroid awal."""
    n = X.shape[0]
    if k > n:
        raise ValueError("k lebih besar dari jumlah sampel.")
    idx = rng.choice(n, size=k, replace=False)
    return X[idx, :].copy()


def _init_kmeans_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n, d = X.shape
    # validasi jumlah titik unik
    uniq_rows = np.unique(X, axis=0)
    if k > len(uniq_rows):
        raise ValueError("k lebih besar dari jumlah titik unik pada data.")

    centroids = np.empty((k, d), dtype=float)
    chosen = np.zeros(n, dtype=bool)

    # 1) pilih centroid pertama uniform
    i0 = int(rng.integers(0, n))
    centroids[0] = X[i0]
    chosen[i0] = True

    # 2) probabilitas ∝ D(x)^2 ke centroid terdekat
    closest_sq = _pairwise_sq_dists(X, centroids[0:1]).reshape(-1)
    closest_sq = np.clip(closest_sq, 0.0, None)

    for j in range(1, k):
        # nolkan peluang titik yang sudah terpilih
        w = closest_sq.copy()
        w[chosen] = 0.0
        w = np.clip(w, 0.0, None)
        tot = w.sum()

        if not np.isfinite(tot) or tot <= 0.0:
            # fallback: pilih uniform dari yang belum terpilih
            candidates = np.where(~chosen)[0]
            next_i = int(rng.choice(candidates))
        else:
            probs = w / tot
            # proteksi numerik: pastikan non-negative dan sum≈1
            probs = np.clip(probs, 0.0, None)
            probs = probs / probs.sum()
            next_i = int(rng.choice(n, p=probs))

        centroids[j] = X[next_i]
        chosen[next_i] = True

        # update D(x)^2 terdekat
        dist_new = _pairwise_sq_dists(X, centroids[j:j+1]).reshape(-1)
        dist_new = np.clip(dist_new, 0.0, None)
        closest_sq = np.minimum(closest_sq, dist_new)

    return centroids



# --------------------------- K-Means Core ---------------------------

def kmeans_fit(
    X: np.ndarray,
    k: int,
    init: str = "kmeans++",          # "kmeans++" | "random"
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Jalankan algoritma K-Means (Lloyd) manual.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)  -- di modul ini difokuskan 2D untuk plotting
    k : int                               jumlah cluster
    init : str                            "kmeans++" (default) atau "random"
    max_iter : int                        batas iterasi
    tol : float                           toleransi pergerakan centroid (L2)
    random_state : Optional[int]          seed RNG

    Returns
    -------
    labels : ndarray (n_samples,)         label cluster [0..k-1]
    centroids : ndarray (k, n_features)   posisi centroid akhir
    inertia : float                       jumlah SSE (total within-cluster sum of squares)
    n_iter : int                          iterasi yang dipakai
    """
    if k < 1:
        raise ValueError("k harus >= 1.")
    if X.ndim != 2:
        raise ValueError("X harus 2D (n_samples, n_features).")

    rng = np.random.default_rng(random_state)
    if init == "random":
        C = _init_random(X, k, rng)
    else:
        C = _init_kmeans_plus(X, k, rng)

    prev_C = C.copy()
    for it in range(1, max_iter + 1):
        # Assignment step
        d2 = _pairwise_sq_dists(X, C)           # (n,k)
        labels = np.argmin(d2, axis=1)

        # Update step
        new_C = C.copy()
        moved = 0.0
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_C[j] = X[mask].mean(axis=0)
            # Jika cluster kosong, re-seed ke titik terjauh dari centroid terdekat
            else:
                # cari titik dengan jarak terdekat terbesar
                closest = np.min(_pairwise_sq_dists(X, C), axis=1)
                far_i = int(np.argmax(closest))
                new_C[j] = X[far_i]

            moved += np.linalg.norm(new_C[j] - C[j])

        C = new_C
        if moved <= tol:
            return labels, C, float(_inertia(X, C, labels)), it

        prev_C = C

    return labels, C, float(_inertia(X, C, labels)), max_iter


def _inertia(X: np.ndarray, C: np.ndarray, labels: np.ndarray) -> float:
    """Total SSE (jumlah jarak kuadrat dari tiap titik ke centroid klasternya)."""
    d2 = _pairwise_sq_dists(X, C)
    return float(np.sum(d2[np.arange(X.shape[0]), labels]))


def silhouette_score_simple(
    X: np.ndarray, labels: np.ndarray, sample: int = 1000, random_state: Optional[int] = 42
) -> float:
    """
    Silhouette score sederhana (tanpa sklearn).
    Untuk efisiensi, sample titik secara acak jika n > sample.
    NOTE: Kompleksitas O(n^2) pada sample — cukup untuk diagnosis cepat.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    if n == 0:
        return float("nan")
    if n > sample:
        idx = rng.choice(n, size=sample, replace=False)
        Xs = X[idx]
        ls = labels[idx]
    else:
        Xs = X
        ls = labels

    # precompute jarak matriks
    # gunakan jarak Euclidean (bukan kuadrat) untuk silhouette
    # (hemat memori: hitung incremental)
    m = Xs.shape[0]
    D = np.zeros((m, m), dtype=float)
    for i in range(m):
        dif = Xs - Xs[i]
        D[i] = np.sqrt(np.sum(dif * dif, axis=1))

    s_vals = []
    for i in range(m):
        same = (ls == ls[i])
        other = ~same
        # a(i): mean intra-cluster distance (kecuali diri sendiri)
        if same.sum() > 1:
            a = D[i, same].sum() / (same.sum() - 1)
        else:
            a = 0.0
        # b(i): minimum mean distance ke cluster lain
        b = np.inf
        for cl in np.unique(ls[other]):
            mask = ls == cl
            b = min(b, D[i, mask].mean())
        s = 0.0 if (a == 0.0 and b == 0.0) else (b - a) / max(a, b)
        s_vals.append(s)

    return float(np.mean(s_vals)) if s_vals else float("nan")


# --------------------------- API Tingkat DF -------------------------

def run_kmeans(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    k: int = 3,
    init: str = "kmeans++",
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = 42,
    compute_silhouette: bool = True,
) -> Dict[str, object]:
    """
    Endpoint utama untuk controller:
    - Validasi & ambil dua kolom numerik
    - Jalankan K-Means
    - Kembalikan label (Series ter-align index asli), centroid (DataFrame), metrik, dan Figure scatter.

    Return dict keys:
      {
        "labels": pd.Series[name="KMeans_Label"],
        "centroids": pd.DataFrame(columns=[x_col, y_col]),
        "inertia": float,
        "silhouette": Optional[float],
        "figure": matplotlib.figure.Figure,
        "used_index": pd.Index
      }
    """
    X, used_idx, df_xy = _ensure_numeric_2cols(df, x_col, y_col)

    labels, C, inertia, n_iter = kmeans_fit(
        X, k=k, init=init, max_iter=max_iter, tol=tol, random_state=random_state
    )

    sil = silhouette_score_simple(X, labels) if compute_silhouette and k > 1 else None

    # bungkus hasil
    s_labels = pd.Series(labels, index=used_idx, name="KMeans_Label")
    cent = pd.DataFrame(C, columns=[x_col, y_col])

    fig = make_scatter_figure(
        df_xy, x_col, y_col, s_labels.loc[df_xy.index].values, centroids=cent, title=f"K-Means (k={k})"
    )

    return {
        "labels": s_labels,
        "centroids": cent,
        "inertia": inertia,
        "silhouette": sil,
        "figure": fig,
        "used_index": used_idx,
    }


# --------------------------- Plotting -------------------------------

def make_scatter_figure(
    df_xy: pd.DataFrame,
    x_col: str,
    y_col: str,
    labels: np.ndarray,
    centroids: Optional[pd.DataFrame] = None,
    title: str = "K-Means Scatter",
) -> plt.Figure:
    """
    Buat matplotlib Figure scatter 2D:
    - Titik diwarnai berdasarkan label cluster
    - Tampilkan centroid sebagai 'X' besar
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # untuk kestabilan warna, gunakan labels (int)
    sc = ax.scatter(df_xy[x_col].values, df_xy[y_col].values, c=labels, s=18, alpha=0.9)
    if centroids is not None and len(centroids) > 0:
        ax.scatter(
            centroids[x_col].values, centroids[y_col].values,
            marker="X", s=160, edgecolor="black"
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    return fig


# ----------------------- Demo mandiri (opsional) --------------------

def _demo_streamlit():
    """
    Jalankan:
      streamlit run KMeans_Visual.py
    untuk mencoba cepat (tanpa controller Visualisasi.py).
    """
    import streamlit as st  # aman, hanya dipakai kalau demo dipanggil
    st.set_page_config(page_title="K-Means Demo", layout="wide")
    st.title("K-Means Manual — Demo")

    # Data dummy
    rng = np.random.default_rng(7)
    A = rng.normal(loc=[0, 0], scale=[1.0, 1.0], size=(150, 2))
    B = rng.normal(loc=[5, 5], scale=[1.0, 1.0], size=(150, 2))
    C = rng.normal(loc=[0, 6], scale=[1.2, 1.2], size=(150, 2))
    X = np.vstack([A, B, C])
    df = pd.DataFrame(X, columns=["X1", "X2"])

    st.write("Gunakan dua kolom numerik (atau komponen PCA/UMAP dari app utama).")
    x_col = st.selectbox("X axis", df.columns.tolist(), index=0)
    y_col = st.selectbox("Y axis", df.columns.tolist(), index=1)
    k = st.number_input("k (cluster)", 1, 10, 3, 1)
    init = st.selectbox("init", ["kmeans++", "random"], index=0)
    max_iter = st.number_input("max_iter", 10, 1000, 200, 10)
    tol = st.number_input("tol", 1e-8, 1.0, 1e-4, format="%.6f")
    seed = st.number_input("random_state", 0, 10_000, 42, 1)

    if st.button("Run K-Means"):
        res = run_kmeans(
            df, x_col=x_col, y_col=y_col, k=int(k),
            init=init, max_iter=int(max_iter), tol=float(tol),
            random_state=int(seed)
        )
        st.pyplot(res["figure"])
        st.write(f"Inertia (SSE): {res['inertia']:.4f}")
        if res["silhouette"] is not None:
            st.write(f"Silhouette (sampled): {res['silhouette']:.4f}")
        with st.expander("Centroids"):
            st.write(res["centroids"])
        with st.expander("Labels (head)"):
            st.write(res["labels"].head())


if __name__ == "__main__":
    # Hanya jika file ini dijalankan langsung:
    try:
        _demo_streamlit()
    except Exception as e:
        print("Untuk demo Streamlit jalankan: streamlit run KMeans_Visual.py")
        print("Error:", e)
