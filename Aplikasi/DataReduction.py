# DataReduction.py
# Reduksi dimensi: PCA manual (SVD) + UMAP (dengan fallback)
# Kompatibel dengan Streamlit 1.11.0 dan app.py sebelumnya

from typing import Tuple, Optional, List
import numpy as np
import pandas as pd

# ---------------- Utils ----------------
def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ambil kolom numerik & coerce; drop non-numerik."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df[numeric_cols].apply(pd.to_numeric, errors="coerce")

def _zscale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd

# ---------------- PCA Manual ----------------
def pca_reduce(
    df: pd.DataFrame,
    n_components: int = 2,
    scale_zscore: bool = True
) -> Tuple[pd.DataFrame, dict, str, np.ndarray]:
    """
    PCA manual (SVD). Return: (df_pca, model, log, explained_variance_ratio)
    """
    if df is None or len(df) == 0:
        return df, {}, "Data kosong – tidak ada reduksi.", np.array([])

    X = _ensure_numeric(df).values
    X = np.nan_to_num(X)

    if scale_zscore:
        X, mu, sd = _zscale(X)
    else:
        mu = np.zeros(X.shape[1]); sd = np.ones(X.shape[1])

    try:
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        components = VT[:n_components]
        scores = np.dot(X, components.T)
        ev_all = (S**2) / np.sum(S**2)
        ev_ratio = ev_all[:n_components]

        cols_pc = [f"PC{i+1}" for i in range(n_components)]
        df_pca = pd.DataFrame(scores[:, :n_components], columns=cols_pc, index=df.index)

        model = {
            "components": components,
            "mean": mu,
            "std": sd,
            "explained_variance_ratio": ev_ratio,
        }
        log = (
            f"[PCA] OK: {n_components} komponen. "
            f"Explained Variance: {np.round(ev_ratio*100, 2)}% | "
            f"Total: {round(float(np.sum(ev_ratio))*100, 2)}%"
        )
        return df_pca, model, log, ev_ratio
    except Exception as e:
        return df, {}, f"Gagal PCA: {e}", np.array([])

# ---------------- UMAP ----------------
def _import_umap_class():
    """Coba import UMAP dari berbagai namespace."""
    try:
        from umap import UMAP
        return UMAP
    except Exception:
        try:
            from umap.umap_ import UMAP
            return UMAP
        except Exception as e:
            raise ImportError("Paket 'umap-learn' tidak ditemukan.") from e

def umap_reduce(
    df: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    scale_zscore: bool = True,
    random_state: int = 42
) -> Tuple[pd.DataFrame, dict, str, np.ndarray]:
    """
    UMAP reduction. Return: (df_umap, model, log, empty evr)
    Jika 'umap-learn' tidak ada → fallback ke PCA (kolom dinamai UMAP1..k).
    """
    if df is None or len(df) == 0:
        return df, {}, "Data kosong – tidak ada reduksi.", np.array([])

    X = _ensure_numeric(df).values
    X = np.nan_to_num(X)
    if scale_zscore:
        X, mu, sd = _zscale(X)
    else:
        mu = np.zeros(X.shape[1]); sd = np.ones(X.shape[1])

    try:
        UMAP = _import_umap_class()
        um = UMAP(
            n_components=n_components,
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            metric=metric,
            random_state=random_state,
        )
        emb = um.fit_transform(X)  # (n_samples, n_components)
        cols = [f"UMAP{i+1}" for i in range(n_components)]
        df_umap = pd.DataFrame(emb, columns=cols, index=df.index)
        model = {
            "params": {
                "n_components": n_components,
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "metric": metric,
                "scale_zscore": scale_zscore,
                "random_state": random_state,
            },
            "mean": mu,
            "std": sd,
        }
        log = (
            f"[UMAP] OK: comp={n_components}, neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}. "
            f"Embedding: {emb.shape}"
        )
        return df_umap, model, log, np.array([])
    except ImportError as e:
        # Fallback ke PCA (tetap beri nama kolom UMAP*)
        df_pca, model, log_pca, ev = pca_reduce(pd.DataFrame(X, index=df.index), n_components=n_components, scale_zscore=False)
        df_pca.columns = [f"UMAP{i+1}" for i in range(n_components)]
        log = f"[UMAP] Paket 'umap-learn' tidak ditemukan → fallback PCA.\n{log_pca}"
        return df_pca, model, log, ev
    except Exception as e:
        return df, {}, f"Gagal UMAP: {e}", np.array([])

# ---------------- Scree helper untuk PCA ----------------
def scree_data(df: pd.DataFrame, scale_zscore: bool = True) -> Tuple[List[int], List[float]]:
    """Data untuk scree plot PCA (EVR per komponen)."""
    if df is None or len(df) == 0:
        return [], []
    X = _ensure_numeric(df).values
    X = np.nan_to_num(X)
    if scale_zscore:
        X, _, _ = _zscale(X)
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    evr = (S**2) / np.sum(S**2)
    return list(range(1, len(evr) + 1)), evr.tolist()
