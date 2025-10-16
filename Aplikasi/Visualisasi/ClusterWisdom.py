# Visualisasi/ClusterWisdom.py
# LOGIC ONLY — Ringkasan per-cluster (Masalah → Solusi) + Prioritisasi Top-1 + Label/Golongan
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# ============================================================
# 0) TEMPLATE AKSI & ATURAN LINTAS-DOMAIN
# ============================================================

ACTIONS_DOMAIN = {
    "Education": {
        "boost": [
            "Skalakan peer-tutoring terjadwal untuk MK sulit",
            "Standardisasi rubrik & bank soal antarkelas",
            "Mini-SPOC/kelas akselerasi untuk talenta tinggi",
            "Clinic mingguan ‘Ask-TA’ (hybrid: luring+LMS)",
            "Early-feedback dosen (rubrik + contoh jawaban)",
        ],
        "fix": [
            "Remedial terstruktur + kontrak belajar personal",
            "Audit beban tugas & sinkronisasi LO antar-dosen",
            "Sesi belajar terarah berbasis data topik tersulit",
            "Early-warning LMS (absensi/nilai) + outreach proaktif",
            "Pelatihan strategi belajar (retrieval, spaced practice)",
        ],
    },
    "Financial": {
        "boost": [
            "Matching beasiswa berbasis profil (otomasi formulir)",
            "Workshop budgeting & simulasi cash-flow",
            "Portal kerja paruh waktu on-campus (jadwal ramah kuliah)",
            "Kurikulum literasi finansial berjenjang (dasar→lanjut)",
        ],
        "fix": [
            "Dana darurat mikro + skema cicilan UKT fleksibel",
            "Konseling finansial 1-on-1 (30 menit, jalur cepat)",
            "Kampanye hak bantuan biaya & kanal pengajuan cepat",
            "Kemitraan part-time dengan slot waktu terjamin",
        ],
    },
    "Physical": {
        "boost": [
            "Membership olahraga lintas fasilitas (subsidi)",
            "Program ‘Move-15’: 15 menit gerak harian (akuntabilitas)",
            "Kampanye sleep-hygiene & notifikasi rutinitas tidur",
            "Paket nutrisi sehat di kantin (bundle hemat)",
        ],
        "fix": [
            "Screening kesehatan berkala + rujukan cepat",
            "Kelas low-impact (yoga, stretching, pilates) mingguan",
            "Voucher makan sehat untuk cluster target",
            "Edukasi manajemen energi (kafein, cahaya, jadwal tidur)",
        ],
    },
    "Psychological": {
        "boost": [
            "Mindfulness mingguan (hybrid) & modul coping adaptif",
            "Pelatihan kader sebaya (gatekeeper) per angkatan",
            "Peer-support group tematik (akademik/keluarga/finansial)",
        ],
        "fix": [
            "Tambah kuota konselor + SLA respons 3×24 jam",
            "CBT singkat (4–6 sesi) fokus manajemen stres/kecemasan",
            "Protokol krisis & jalur rujukan eksternal yang jelas",
            "Edukasi manajemen waktu & ekspektasi (self-compassion)",
        ],
    },
    "Relational": {
        "boost": [
            "Program buddy/mentee untuk maba/transfer",
            "Proyek lintas jurusan (minat/servis masyarakat)",
            "Event komunitas kecil, rutin, low-barrier (boardgame/club)",
        ],
        "fix": [
            "Kelompok dukungan sosial 10–12 orang (fasilitator terlatih)",
            "Pelatihan komunikasi asertif & resolusi konflik",
            "Koordinasi jadwal organisasi ↔ akademik (hindari bentrok)",
            "Pendampingan adaptasi sosial bagi mahasiswa non-domisili",
        ],
    },
}

CROSS_RULES: List[Dict] = [
    {
        "if_low": {"Financial", "Psychological"},
        "because": "Tekanan finansial berkontribusi pada stres psikologis, menurunkan fokus & ketahanan.",
        "do": [
            "Padukan konseling finansial + CBT singkat manajemen stres",
            "Prioritaskan dana darurat & cicilan UKT, check-in 2 mingguan",
            "Support group ‘financial-stress’ dengan fasilitator",
        ],
    },
    {
        "if_low": {"Education", "Psychological"},
        "because": "Kesulitan akademik menurunkan self-efficacy dan meningkatkan kecemasan (lingkaran nilai rendah ↔ cemas).",
        "do": [
            "Remedial terstruktur + clinic ‘Ask-TA’, target realistis",
            "CBT singkat fokus test-anxiety & self-regulation",
            "Kontrak belajar mingguan & monitoring progres",
        ],
    },
    {
        "if_low": {"Physical", "Psychological"},
        "because": "Kurang tidur/aktivitas fisik memperburuk mood & stres; kesejahteraan mental turun.",
        "do": [
            "Sleep-hygiene + ‘Move-15’ harian (akuntabilitas kelompok)",
            "Mindfulness ringan 10 menit/hari + tracking kebiasaan",
            "Rujukan cepat ke klinik untuk evaluasi pola tidur",
        ],
    },
    {
        "if_low": {"Relational", "Psychological"},
        "because": "Dukungan sosial rendah mengurangi buffer stres & meningkatkan risiko menarik diri.",
        "do": [
            "Buddy system + peer-support group kecil",
            "Aktivitas komunitas low-barrier untuk membangun belonging",
            "Pelatihan komunikasi asertif dasar",
        ],
    },
    {
        "if_low": {"Financial", "Education"},
        "because": "Masalah finansial membatasi waktu/sumber belajar (harus kerja), menekan performa akademik.",
        "do": [
            "Beasiswa darurat/pekerjaan kampus dengan slot aman akademik",
            "Workshop budgeting + time-blocking untuk pekerja sambil kuliah",
            "Remedial fleksibel pada MK inti",
        ],
    },
]

PRETTY = {
    "Education": "Akademik",
    "Financial": "Finansial",
    "Physical": "Fisik",
    "Psychological": "Psikologis",
    "Relational": "Relasional",
}

# ============================================================
# 0.1) PRIORITISASI (penyebab & aksi utama)
# ============================================================

DOMAIN_IMPACT = {   # dampak relatif ke performa/retensi
    "Education": 1.00,
    "Psychological": 0.95,
    "Financial": 0.90,
    "Relational": 0.75,
    "Physical": 0.70,
}
DOMAIN_COST_FIX = {  # biaya/kerumitan intervensi saat domain rendah
    "Education": 1.15, "Psychological": 1.05, "Financial": 1.20, "Relational": 1.00, "Physical": 1.00,
}
DOMAIN_COST_BOOST = {  # biaya scaling domain kuat
    "Education": 1.05, "Psychological": 1.02, "Financial": 1.10, "Relational": 1.00, "Physical": 1.00,
}
CROSS_RULE_WEIGHT = {  # kekuatan hubungan lintas-domain
    frozenset({"Financial","Psychological"}): 1.10,
    frozenset({"Education","Psychological"}): 1.10,
    frozenset({"Physical","Psychological"}): 1.00,
    frozenset({"Relational","Psychological"}): 0.95,
    frozenset({"Financial","Education"}): 1.00,
}
ALPHA = 0.5  # bobot ukuran cluster n^alpha

def _gap_value(basis_name: str, score: float) -> float:
    return abs(score) if str(basis_name).lower().startswith("z") else abs(score - 3.0)

# ============================================================
# 1) AMBANG OTOMATIS & DETEKSI STATUS
# ============================================================

def _auto_thresholds(basis_name: str) -> Tuple[float, float]:
    if str(basis_name).lower().startswith("z"):
        return -0.4, 0.4
    return 3.0, 3.5  # skala 1–5

def _strip_z(col: str) -> str:
    return col.replace("_Z", "")

def detect_status_per_cluster(
    prof: pd.DataFrame, basis_name: str, aspect_cols: List[str]
) -> List[Dict]:
    lo, hi = _auto_thresholds(basis_name)
    out = []
    for cid, row in prof[aspect_cols].iterrows():
        high, low, neutral = [], [], []
        for c in aspect_cols:
            v = float(row[c])
            d = _strip_z(c)
            if v >= hi:   high.append((d, v))
            elif v <= lo: low.append((d, v))
            else:         neutral.append((d, v))
        out.append({"cluster": str(cid), "high": high, "low": low, "neutral": neutral})
    return out

# ============================================================
# 2) KETERKAITAN ANTAR-DOMAIN (heuristik rule-based)
# ============================================================

def linkages_and_causes(low_domains: set, high_domains: set) -> Tuple[List[str], List[str], List[str]]:
    linkage, causes, cross_actions = [], [], []
    for rule in CROSS_RULES:
        need = rule.get("if_low") or set()
        if need.issubset(low_domains):
            linkage.append(f"Keterkaitan: {', '.join(sorted(need))} saling memperburuk.")
            causes.append(rule["because"])
            cross_actions.extend(rule["do"])
    if high_domains:
        linkage.append(f"Kekuatan yang bisa diskalakan: {', '.join(sorted(high_domains))}.")
        causes.append("Praktik baik pada domain kuat dapat dijadikan role model/peer learning.")
    return linkage, list(dict.fromkeys(causes)), list(dict.fromkeys(cross_actions))

# ============================================================
# 3) AUTO-LABEL (nama golongan) dari pola ↑/↓
# ============================================================

def _dev_from_center(basis_name: str, v: float) -> float:
    return abs(v) if str(basis_name).lower().startswith("z") else abs(v - 3.0)

def make_cluster_labels(
    prof: pd.DataFrame,
    basis_name: str,
    aspect_cols: Optional[List[str]] = None,
    top_high: int = 2,
    top_low: int = 2,
) -> Dict[str, str]:
    if aspect_cols is None:
        aspect_cols = [c for c in prof.columns if pd.api.types.is_numeric_dtype(prof[c])]

    lo, hi = _auto_thresholds(basis_name)
    labels: Dict[str, str] = {}
    for cid, row in prof[aspect_cols].iterrows():
        highs, lows = [], []
        for c in aspect_cols:
            v = float(row[c]); d = _strip_z(c)
            if v >= hi: highs.append((PRETTY.get(d, d), v))
            elif v <= lo: lows.append((PRETTY.get(d, d), v))
        highs.sort(key=lambda x: _dev_from_center(basis_name, x[1]), reverse=True)
        lows.sort(key=lambda x: _dev_from_center(basis_name, x[1]), reverse=True)
        hi_part = " & ".join(f"{nm}↑" for nm, _ in highs[:top_high])
        lo_part = " & ".join(f"{nm}↓" for nm, _ in lows[:top_low])
        name = "Seimbang"
        if hi_part and lo_part: name = f"{hi_part} | {lo_part}"
        elif hi_part:           name = hi_part
        elif lo_part:           name = lo_part
        labels[str(cid)] = name
    return labels

# ============================================================
# 4) SKOR PENYEBAB & AKSI UTAMA (Top-1)
# ============================================================

def score_causes(
    low_pairs: List[Tuple[str, float]], basis_name: str, n_members: int, high_domains: set
) -> Tuple[str, float, str]:
    best = ("", -1.0, "")
    for d, v in low_pairs:
        gap = _gap_value(basis_name, v)
        score = gap * DOMAIN_IMPACT.get(d, 0.8) * (max(1, n_members) ** ALPHA) / DOMAIN_COST_FIX.get(d, 1.0)
        why = f"Gap {d} {v:+.2f} (basis {basis_name}), dampak {DOMAIN_IMPACT.get(d,1.0):.2f}, biaya {DOMAIN_COST_FIX.get(d,1.0):.2f}."
        if score > best[1]:
            best = (f"Defisit {PRETTY.get(d,d)}", score, why)

    low_set = {d for d, _ in low_pairs}
    for rule in CROSS_RULES:
        need = rule["if_low"]
        if need.issubset(low_set):
            parts = []
            for d in need:
                sc = next((vv for dd, vv in low_pairs if dd == d), 0.0)
                parts.append(_gap_value(basis_name, sc) * DOMAIN_IMPACT.get(d, 0.8) / DOMAIN_COST_FIX.get(d, 1.0))
            score = sum(parts) * (max(1, n_members) ** ALPHA) * CROSS_RULE_WEIGHT.get(frozenset(need), 1.0)
            why = f"Kombinasi {', '.join(sorted(need))} (bobot rule {CROSS_RULE_WEIGHT.get(frozenset(need),1.0):.2f})."
            if score > best[1]:
                best = (f"Interaksi {', '.join(sorted(PRETTY.get(x,x) for x in need))}", score, why)

    if best[1] < 0 and high_domains:
        d = sorted(list(high_domains), key=lambda x: DOMAIN_IMPACT.get(x,0.8), reverse=True)[0]
        best = (f"Skalakan {PRETTY.get(d,d)}", 0.0, f"Tidak ada defisit; kekuatan {d} berdampak {DOMAIN_IMPACT.get(d,1.0):.2f}.")
    return best

def _actions_for_domain(domain: str, is_low: bool, k: int = 5) -> List[str]:
    pack = ACTIONS_DOMAIN.get(domain, {})
    return (pack.get("fix", []) if is_low else pack.get("boost", []))[:k]

def score_actions(
    low_pairs: List[Tuple[str,float]], high_pairs: List[Tuple[str,float]], basis_name: str, n_members: int
) -> Tuple[str, float, str]:
    candidates: List[Tuple[str,float,str]] = []

    for d, v in low_pairs:
        gap = _gap_value(basis_name, v)
        impact = DOMAIN_IMPACT.get(d, 0.8)
        cost = DOMAIN_COST_FIX.get(d, 1.0)
        base = gap * impact * (max(1, n_members) ** ALPHA) / cost
        for a in _actions_for_domain(d, is_low=True, k=5):
            candidates.append((f"{PRETTY.get(d,d)}: {a}", base, f"Gap {d} {v:+.2f}, impact {impact:.2f}, cost {cost:.2f}"))

    for d, v in high_pairs:
        gap = max(0.2, _gap_value(basis_name, v))
        impact = DOMAIN_IMPACT.get(d, 0.8)
        cost = DOMAIN_COST_BOOST.get(d, 1.0)
        base = gap * impact * (max(1, n_members) ** ALPHA) / cost
        for a in _actions_for_domain(d, is_low=False, k=5):
            candidates.append((f"Skalakan {PRETTY.get(d,d)}: {a}", base, f"Kekuatan {d} {v:+.2f}, impact {impact:.2f}, cost {cost:.2f}"))

    low_set = {d for d, _ in low_pairs}
    for rule in CROSS_RULES:
        need = rule["if_low"]
        if need.issubset(low_set):
            parts = []
            for d in need:
                sc = next((vv for dd, vv in low_pairs if dd == d), 0.0)
                parts.append(_gap_value(basis_name, sc) * DOMAIN_IMPACT.get(d, 0.8) / DOMAIN_COST_FIX.get(d, 1.0))
            base = (sum(parts)/len(parts)) * (max(1, n_members) ** ALPHA) * CROSS_RULE_WEIGHT.get(frozenset(need), 1.0)
            for a in rule["do"]:
                candidates.append((f"Intervensi lintas-domain: {a}", base, f"Rule {', '.join(sorted(need))}"))

    if not candidates:
        return ("(Tidak ada aksi spesifik — semua domain netral/tinggi)", 0.0, "N/A")

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]

# ============================================================
# 5) NARASI PER-CLUSTER
# ============================================================

def build_cluster_wisdom(
    prof: pd.DataFrame,
    counts: pd.Series,
    basis_name: str,
    aspect_cols: List[str],
    max_actions_per_domain: int = 5,
    label_map: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    status = detect_status_per_cluster(prof, basis_name, aspect_cols)
    out = []
    for s in status:
        cid = s["cluster"]
        try:
            n = int(counts.loc[int(cid)])
        except Exception:
            try: n = int(counts.loc[cid])
            except Exception: n = 0

        low_domains = {d for d, _ in s["low"]}
        high_domains = {d for d, _ in s["high"]}

        problems  = [f"{PRETTY.get(d,d)} rendah (skor {v:.2f})"  for d, v in s["low"]]
        strengths = [f"{PRETTY.get(d,d)} tinggi (skor {v:.2f})" for d, v in s["high"]]

        linkages, causes, cross_actions = linkages_and_causes(low_domains, high_domains)

        recs = []
        for d, _ in s["low"]:
            acts = _actions_for_domain(d, is_low=True, k=max_actions_per_domain)
            if acts: recs.append(f"{PRETTY.get(d,d)}: " + "; ".join(acts))
        for d, _ in s["high"]:
            acts = _actions_for_domain(d, is_low=False, k=max_actions_per_domain)
            if acts: recs.append(f"Skalakan {PRETTY.get(d,d)}: " + "; ".join(acts))
        if cross_actions:
            recs.append("Intervensi lintas-domain: " + "; ".join(cross_actions))

        primary_cause, _, cause_why = score_causes(s["low"], basis_name, n, high_domains)
        primary_action, _, act_why  = score_actions(s["low"], s["high"], basis_name, n)

        out.append({
            "cluster": cid,
            "label": (label_map or {}).get(cid, ""),
            "n": n,
            "problems": problems,
            "strengths": strengths,
            "linkages": linkages,
            "causes": causes,
            "recommendations": recs,
            "basis": basis_name,
            "primary_cause": primary_cause,
            "primary_action": primary_action,
            "why": f"{cause_why} | {act_why}",
        })
    return out

# ============================================================
# 6) MARKDOWN
# ============================================================

def wisdom_to_markdown(wisdom: List[Dict]) -> str:
    if not wisdom:
        return "# Ringkasan Per-Cluster\n\n(Tidak ada data.)\n"
    lines = ["# Ringkasan Per-Cluster (Masalah → Solusi)", ""]
    for w in wisdom:
        title = f"Cluster {w['cluster']}"
        if w.get("label"): title += f" — {w['label']}"
        lines.append(f"## {title} — Anggota: {int(w['n'])} · Basis: {w['basis']}")
        if w.get("primary_action"): lines.append(f"**📌 Rekomendasi Utama:** {w['primary_action']}")
        if w.get("primary_cause"):  lines.append(f"**🧠 Penyebab Utama (hipotesis):** {w['primary_cause']}")
        if w.get("why"):            lines.append(f"_Alasan singkat:_ {w['why']}")
        if w["problems"]:
            lines.append("**Fokus Masalah:**")
            for p in w["problems"]: lines.append(f"- {p}")
        if w["linkages"]:
            lines.append("**Keterkaitan Antar-Domain:**")
            for t in w["linkages"]: lines.append(f"- {t}")
        if w["causes"]:
            lines.append("**Asumsi Penyebab (tambahan):**")
            for c in w["causes"]: lines.append(f"- {c}")
        if w["recommendations"]:
            lines.append("**Saran Solusi (lengkap):**")
            for r in w["recommendations"]: lines.append(f"- {r}")
        if w["strengths"]:
            lines.append("_Modal yang bisa diskalakan:_ " + "; ".join(w["strengths"]))
        lines.append("")
    return "\n".join(lines)

# ============================================================
# 7) ENTRYPOINT UNTUK APP
# ============================================================

def derive_wisdom_from_profiles(
    prof: pd.DataFrame,
    counts: pd.Series,
    basis_name: str,
    aspect_cols: Optional[List[str]] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict], str]:
    if aspect_cols is None:
        aspect_cols = [c for c in prof.columns if pd.api.types.is_numeric_dtype(prof[c])]
    if label_map is None:
        label_map = make_cluster_labels(prof, basis_name, aspect_cols)
    wisdom = build_cluster_wisdom(prof, counts, basis_name, aspect_cols, label_map=label_map)
    return wisdom, wisdom_to_markdown(wisdom)
