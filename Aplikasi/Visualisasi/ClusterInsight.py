# Visualisasi/ClusterInsight.py
# Logika insight cluster tanpa sklearn — kompatibel dengan Streamlit 1.11.0 (dipanggil dari Visualisasi.py).
# Fokus pada kolom aspek (disarankan *_Z): Education[_Z], Financial[_Z], Physical[_Z], Psychological[_Z], Relational[_Z]

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


# --------------------------- Utilities dasar ---------------------------

def _to_numeric_series(s: pd.Series) -> pd.Series:
    """
    Koersi numerik yang toleran:
    - Mengganti koma desimal menjadi titik,
    - errors='coerce' agar noise → NaN, aman untuk dropna downstream.
    """
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _num(df: pd.DataFrame) -> pd.DataFrame:
    """Koersi seluruh kolom DataFrame menjadi numerik (pakai _to_numeric_series)."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        out[c] = _to_numeric_series(df[c])
    return out


def _smart_sort_index(idx: pd.Index) -> List[str]:
    """
    Sortir label cluster 'secara manusiawi':
    - Coba urutkan numerik jika semua bisa dikonversi (mis. '0','1','-1'),
    - Jika campur (angka & string), urutkan string biasa.
    """
    vals = idx.astype(str).tolist()
    try:
        as_int = [int(x) for x in vals]
        order = np.argsort(as_int)
        return [vals[i] for i in order]
    except Exception:
        return sorted(vals)


# --------------------------- Inti insight ---------------------------

def cluster_profiles(
    df: pd.DataFrame,
    label_col: str,
    aspect_cols: List[str],
    use_zscore: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Hitung profil (rata-rata) tiap aspek per-cluster.
    - df: DataFrame sumber (harus memuat kolom label dan aspek).
    - label_col: nama kolom label cluster (mis. 'KMeans_Label', 'DBSCAN_Label', ...).
    - aspect_cols: daftar kolom aspek (disarankan *_Z untuk komparabilitas).
    - use_zscore: jika True → normalisasi global (mean 0, std 1) sebelum agregasi.

    Return:
      df_prof  : DataFrame (index=label cluster str, kolom=aspect_cols) berisi rata-rata per aspek.
      mu, sd   : statistik global (berdasarkan df[aspect_cols]) untuk referensi/kebutuhan lain.
    """
    if label_col not in df.columns:
        raise ValueError(f"Kolom label '{label_col}' tidak ditemukan di df.")

    missing = [c for c in aspect_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom aspek berikut tidak ditemukan: {missing}")

    # Koersi numerik & gabungkan label
    X = _num(df[aspect_cols])
    lab = df[label_col].astype(str)
    data = X.join(lab.rename("_lab_")).dropna(axis=0, how="any").copy()

    # Z-score global opsional (lebih fair antar-aspek)
    if use_zscore:
        mu = X.mean(axis=0)
        sd = X.std(axis=0).replace(0, 1.0)
        Xz = (X - mu) / sd
        data = Xz.join(lab.rename("_lab_")).dropna(axis=0, how="any")
    else:
        mu = X.mean(axis=0)
        sd = X.std(axis=0).replace(0, 1.0)

    # Agregasi mean per cluster
    prof = data.groupby("_lab_")[aspect_cols].mean()

    # Urutkan label secara "smart"
    order = _smart_sort_index(prof.index)
    prof = prof.loc[order]

    return prof, mu.to_dict(), sd.to_dict()


def _bucket(v: float, lo: float, hi: float) -> str:
    """
    Kategori tiga tingkat:
    - v >= hi → 'Sejahtera'
    - v <= lo → 'Tidak Sejahtera'
    - else    → 'Menengah'
    """
    if not np.isfinite(v):
        return "Menengah"
    if v >= hi:
        return "Sejahtera"
    if v <= lo:
        return "Tidak Sejahtera"
    return "Menengah"


def label_clusters(
    df_prof: pd.DataFrame,
    low_thr: float = -0.5,
    high_thr: float = 0.5,
    pretty_names: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Beri label kategori per-aspek untuk setiap cluster + kolom ringkasan '__Ringkasan__'.
    - df_prof: output dari cluster_profiles (mean per aspek per cluster).
    - low_thr/high_thr: ambang batas pada skala df_prof (Jika df_prof berbasis Z, pakai ±0.5 default).
    - pretty_names: mapping nama kolom → label ramah (mis. {'Education_Z':'Akademik', ...}).

    Return: DataFrame = df_prof + '__Ringkasan__' + kolom 'Label_<aspek>'
    """
    aspects = df_prof.columns.tolist()
    pretty = pretty_names or {a: a for a in aspects}

    # Kategorikan per-aspek
    cat = df_prof.applymap(lambda x: _bucket(float(x), low_thr, high_thr))

    # Ringkas jadi kalimat manusiawi
    def _summarize_row(row: pd.Series) -> str:
        pos = [pretty.get(k, k) for k, v in row.items() if v == "Sejahtera"]
        neg = [pretty.get(k, k) for k, v in row.items() if v == "Tidak Sejahtera"]
        mid = [pretty.get(k, k) for k, v in row.items() if v == "Menengah"]
        parts = []
        if pos:
            parts.append(" & ".join(pos) + " Sejahtera")
        if neg:
            parts.append(" & ".join(neg) + " Rendah")
        if not parts and mid:
            parts.append("Mayoritas Menengah")
        return " | ".join(parts) if parts else "Campuran"

    summary = cat.apply(_summarize_row, axis=1)

    out = df_prof.copy()
    out["__Ringkasan__"] = summary
    for a in aspects:
        out[f"Label_{a}"] = cat[a]
    return out


def cluster_variability(
    df: pd.DataFrame,
    label_col: str,
    aspect_cols: List[str]
) -> pd.DataFrame:
    """
    Ukur kompaksi cluster: varians dalam-cluster per-aspek (lebih kecil = lebih kompak).
    Return: DataFrame var per aspek + ukuran cluster (n).
    """
    X = _num(df[aspect_cols])
    labs = df[label_col].astype(str)
    data = X.join(labs.rename("_lab_")).dropna(axis=0, how="any").copy()
    var_in = data.groupby("_lab_")[aspect_cols].var(ddof=0)
    size = data.groupby("_lab_").size().rename("n")

    # Urutkan label smart
    order = _smart_sort_index(var_in.index)
    var_in = var_in.loc[order]
    size = size.loc[order]

    return var_in.join(size)


# --------------------------- Util pendukung opsional ---------------------------

def make_human_name_map(labels_table: pd.DataFrame) -> Dict[str, str]:
    """
    Bangun mapping label cluster → ringkasan manusiawi dari tabel hasil label_clusters().
    - labels_table harus memiliki kolom '__Ringkasan__'
    """
    if "__Ringkasan__" not in labels_table.columns:
        raise ValueError("Kolom '__Ringkasan__' tidak ditemukan. Panggil label_clusters() terlebih dahulu.")
    return labels_table["__Ringkasan__"].to_dict()


def attach_human_names(
    df: pd.DataFrame,
    label_col: str,
    name_map: Dict[str, str],
    new_col: str = "Cluster_Label_Human"
) -> pd.DataFrame:
    """
    Tambahkan nama cluster 'manusiawi' ke df berdasarkan mapping ringkasan.
    - label_col akan di-cast ke str agar cocok dengan key name_map.
    """
    out = df.copy()
    out[new_col] = out[label_col].astype(str).map(name_map)
    return out


# --------------------------- Contoh pemakaian mandiri ---------------------------
if __name__ == "__main__":
    # Demo kecil kalau file ini dijalankan langsung (opsional).
    # Buat data sintetis 100 baris, 5 aspek (sudah Z), dan label cluster acak.
    rng = np.random.default_rng(42)
    n = 100
    df_demo = pd.DataFrame({
        "Education_Z": rng.normal(0, 1, n),
        "Financial_Z": rng.normal(0, 1, n),
        "Physical_Z": rng.normal(0, 1, n),
        "Psychological_Z": rng.normal(0, 1, n),
        "Relational_Z": rng.normal(0, 1, n),
        "KMeans_Label": rng.integers(0, 3, n)
    })

    aspects_z = ["Education_Z","Financial_Z","Physical_Z","Psychological_Z","Relational_Z"]
    prof, mu, sd = cluster_profiles(df_demo, "KMeans_Label", aspects_z, use_zscore=True)
    labels_tbl = label_clusters(
        prof, low_thr=-0.5, high_thr=0.5,
        pretty_names={
            "Education_Z":"Akademik", "Financial_Z":"Finansial", "Physical_Z":"Fisik",
            "Psychological_Z":"Psikologis", "Relational_Z":"Relasional"
        }
    )
    name_map = make_human_name_map(labels_tbl)
    df_demo2 = attach_human_names(df_demo, "KMeans_Label", name_map)

    print("== Profil per-cluster ==")
    print(labels_tbl.head())
    print("\n== Sampel df_demo dengan nama manusiawi ==")
    print(df_demo2.head())

# ---------- Narasi teks interaktif ----------

def cluster_counts(df: pd.DataFrame, label_col: str, order: Optional[List[str]] = None) -> pd.Series:
    """Hitung jumlah anggota per cluster (return Series index=label(str), value=int)."""
    vc = df[label_col].astype(str).value_counts()
    if order is not None:
        vc = vc.reindex(order).fillna(0).astype(int)
    return vc

def make_text_summaries(
    df_prof: pd.DataFrame,
    labels_table: pd.DataFrame,
    counts: pd.Series,
    pretty_names: Optional[Dict[str,str]] = None,
    basis_name: str = "Z-score"
) -> List[Dict[str, str]]:
    """
    Buat ringkasan naratif per cluster.
    Input:
      - df_prof: hasil cluster_profiles (mean per-aspek per-cluster)
      - labels_table: hasil label_clusters (punya kolom '__Ringkasan__' + Label_*)
      - counts: jumlah anggota per cluster (Series, index=label str)
      - pretty_names: mapping nama kolom → label ramah
      - basis_name: 'Z-score' atau 'Mean (1–5)' untuk ditampilkan
    Output: list of dict {cluster, title, body}
    """
    aspects = [c for c in df_prof.columns]
    pretty = pretty_names or {a: a for a in aspects}

    results = []
    for cid in df_prof.index.astype(str).tolist():
        row = df_prof.loc[cid]
        ringkas = labels_table.loc[cid, "__Ringkasan__"] if "__Ringkasan__" in labels_table.columns else ""
        n = int(counts.get(cid, 0))

        # Cari aspek tertinggi/terendah (berdasar nilai mean pada skala df_prof)
        try:
            top_aspect = max(aspects, key=lambda a: float(row[a]))
            low_aspect = min(aspects, key=lambda a: float(row[a]))
        except Exception:
            top_aspect, low_aspect = None, None

        # Judul + isi
        title = f"Cluster {cid} — {ringkas}" if ringkas else f"Cluster {cid}"
        lines = []
        lines.append(f"**Basis**: {basis_name} &nbsp;&nbsp;|&nbsp;&nbsp; **Anggota**: **{n}**")
        lines.append("")
        lines.append("**Rata-rata Aspek:**")
        for a in aspects:
            nm = pretty.get(a, a)
            try:
                val = float(row[a])
            except Exception:
                val = np.nan
            # format 2 desimal (Z) atau 2 desimal (mean 1–5 juga ok)
            lines.append(f"- {nm}: **{val:.2f}**")
        if top_aspect and low_aspect:
            lines.append("")
            lines.append(
                f"**Highlight**: Tertinggi pada **{pretty.get(top_aspect, top_aspect)}**, "
                f"terendah pada **{pretty.get(low_aspect, low_aspect)}**."
            )
        if ringkas:
            lines.append("")
            lines.append(f"**Kesimpulan singkat**: {ringkas}")

        results.append({
            "cluster": cid,
            "title": title,
            "body": "\n".join(lines)
        })
    return results

# ====== BEGIN: Knowledge Layer (rekomendasi kebijakan) ======
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

# Template tindakan default (singkat, bisa kamu kustom)
_DEFAULT_TEMPLATES = {
    "Education": {
        "pos": [
            "Skalakan mentoring/peer-tutoring yang efektif.",
            "Standardisasi materi & rubrik penilaian antar mata kuliah."
        ],
        "neg": [
            "Klinik belajar mingguan untuk mata kuliah inti.",
            "Audit beban tugas & penyelarasan LO antar-dosen."
        ]
    },
    "Financial": {
        "pos": [
            "Program literasi finansial lanjutan (budgeting proaktif).",
            "Optimasi akses beasiswa & kerja paruh waktu on-campus."
        ],
        "neg": [
            "Dana darurat & opsi cicilan UKT fleksibel.",
            "Kemitraan kerja paruh waktu dengan jadwal ramah akademik."
        ]
    },
    "Physical": {
        "pos": [
            "Kampanye sleep hygiene & akses fasilitas olahraga.",
            "Paket nutrisi sehat untuk mahasiswa aktif organisasi."
        ],
        "neg": [
            "Program habit-building aktivitas fisik ringan terstruktur.",
            "Skrining kesehatan berkala & rujukan cepat."
        ]
    },
    "Psychological": {
        "pos": [
            "Workshop coping-stress & mindfulness tingkat lanjut.",
            "Pelatihan kader sebaya (gatekeeper)."
        ],
        "neg": [
            "Perluas kapasitas konseling; SLA & alur rujukan jelas.",
            "Kelas manajemen stres berbasis CBT singkat (4–6 sesi)."
        ]
    },
    "Relational": {
        "pos": [
            "Fasilitasi komunitas minat & proyek kolaboratif lintas jurusan.",
            "Program buddy untuk mahasiswa baru/transfer."
        ],
        "neg": [
            "Kelompok dukungan sosial terjadwal (10–12 orang).",
            "Pelatihan komunikasi asertif & resolusi konflik."
        ]
    }
}

def _aspect_base_name(col: str) -> str:
    return str(col).replace("_Z", "")

def _priority_score(value: float, n_members: int, basis: str = "Z") -> float:
    """Prioritas = deviasi * sqrt(n). Basis Z: |z|; basis mean (1–5): |value-3|."""
    dev = abs(float(value)) if basis.upper().startswith("Z") else abs(float(value) - 3.0)
    return dev * np.sqrt(max(1, int(n_members)))

def _pick_actions(aspect: str, is_positive: bool, templates: dict) -> list:
    base = templates.get(aspect, {})
    return base.get("pos" if is_positive else "neg", [])

def make_policy_recommendations(
    df_prof: pd.DataFrame,
    counts: pd.Series,
    basis_name: str = "Z-score",
    templates: Optional[dict] = None,
    high_thr: float = 0.5,
    low_thr: float = -0.5,
    top_k: int = 3
) -> List[Dict[str, object]]:
    """
    Bentuk rekomendasi kebijakan per-cluster dari profil rata-rata per-aspek (df_prof).
    Hanya aspek yang melewati ambang yang diikutkan.
    """
    tpls = templates or _DEFAULT_TEMPLATES
    is_z = basis_name.upper().startswith("Z")
    results = []

    for cid in df_prof.index.astype(str).tolist():
        row = df_prof.loc[cid]
        n = int(counts.get(cid, 0))
        pos, neg = [], []

        for col in df_prof.columns:
            aspect = _aspect_base_name(col)
            val = float(row[col])

            if is_z:
                is_pos = val >= float(high_thr)
                is_neg = val <= float(low_thr)
            else:
                is_pos = val >= float(high_thr)  # ex: 3.5
                is_neg = val <= float(low_thr)   # ex: 2.5

            if not (is_pos or is_neg):
                continue

            prio = _priority_score(val, n, basis="Z" if is_z else "MEAN")
            actions = _pick_actions(aspect, is_positive=is_pos, templates=tpls)
            item = {"aspect": aspect, "score": val, "priority": prio, "actions": actions}
            (pos if is_pos else neg).append(item)

        pos = sorted(pos, key=lambda d: (-d["priority"], -d["score"]))[:int(top_k)]
        neg = sorted(neg, key=lambda d: (-d["priority"], d["score"]))[:int(top_k)]

        results.append({"cluster": cid, "n": n, "positives": pos, "concerns": neg})

    return results

def recommendations_to_text(
    recs: List[Dict[str, object]],
    pretty_names: Optional[Dict[str, str]] = None,
    basis_name: str = "Z-score"
) -> str:
    """Konversi struktur rekomendasi → Markdown siap diunduh."""
    pretty = pretty_names or {}
    lines = [f"# Rekomendasi Kebijakan (Basis: {basis_name})", ""]
    for r in recs:
        lines.append(f"## Cluster {r['cluster']} — Anggota: {r['n']}")
        lines.append("")
        if r["positives"]:
            lines.append("**Kekuatan (dipertahankan & diskalakan):**")
            for it in r["positives"]:
                nm = pretty.get(it["aspect"], it["aspect"])
                lines.append(f"- {nm} (skor: {it['score']:.2f}) — prioritas {it['priority']:.2f}")
                for a in it["actions"]:
                    lines.append(f"  - {a}")
            lines.append("")
        if r["concerns"]:
            lines.append("**Isu (prioritas perbaikan):**")
            for it in r["concerns"]:
                nm = pretty.get(it["aspect"], it["aspect"])
                lines.append(f"- {nm} (skor: {it['score']:.2f}) — prioritas {it['priority']:.2f}")
                for a in it["actions"]:
                    lines.append(f"  - {a}")
            lines.append("")
        lines.append("---"); lines.append("")
    return "\n".join(lines)

# (opsional) expose simbol agar jelas tersedia
__all__ = [
    # yang lama
    # 'cluster_profiles', 'label_clusters', 'cluster_variability',
    # 'cluster_counts', 'make_text_summaries', 'make_human_name_map', 'attach_human_names',
    # yang baru
    'make_policy_recommendations', 'recommendations_to_text'
]
# ====== END: Knowledge Layer ======
