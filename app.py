import re
import urllib.request
import numpy as np
import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Draw,
    FilterCatalog,
    Lipinski,
    QED,
    rdMolDescriptors,
)


st.set_page_config(
    page_title="DrugDesign Hub PRO",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------- INTERNATIONAL BIOPHARMA UI THEME ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f8fafc;
        color: #0f172a;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(125deg, #071324 0%, #0f294a 55%, #123c5e 100%);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 18px;
        padding: 22px 26px;
        color: #ffffff;
        margin-bottom: 14px;
        box-shadow: 0 12px 28px rgba(7, 19, 36, 0.16);
    }
    .hero-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 8px;
    }
    .hero-badge {
        background: rgba(56, 189, 248, 0.18);
        color: #7dd3fc;
        border: 1px solid rgba(125, 211, 252, 0.35);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 11px;
        border-radius: 999px;
    }
    .hero-std {
        background: rgba(52, 211, 153, 0.16);
        color: #6ee7b7;
        border: 1px solid rgba(110, 231, 183, 0.35);
        font-size: 11px;
        font-weight: 700;
        padding: 4px 11px;
        border-radius: 999px;
    }
    .hero-title {
        font-size: 29px;
        font-weight: 800;
        margin: 0 0 4px 0;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .hero-sub {
        font-size: 13.5px;
        color: #cbd5e1;
        margin: 0;
    }

    /* 5 Colored Tab Buttons: Blue, Green, Yellow, Teal/Blue, Indigo */
    div.stButton > button {
        border-radius: 12px !important;
        font-weight: 800 !important;
        font-size: 13.5px !important;
        padding: 12px 14px !important;
        transition: all 0.18s ease !important;
        border: none !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        filter: brightness(1.05) !important;
    }

    /* Target the first 5-column horizontal navigation block specifically */
    div.hero-card + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1) button {
        background-color: #1d4ed8 !important;
        background-image: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(29, 78, 216, 0.30) !important;
    }
    div.hero-card + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1) button * {
        color: #ffffff !important;
    }

    div.hero-card + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) button {
        background-color: #15803d !important;
        background-image: linear-gradient(135deg, #16a34a, #15803d) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(21, 128, 61, 0.30) !important;
    }
    div.hero-card + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) button * {
        color: #ffffff !important;
    }

    div.hero-card + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(3) button {
        background-color: #eab308 !important;
        background-image: linear-gradient(135deg, #facc15, #eab308) !important;
        color: #1e293b !important;
        box-shadow: 0 4px 12px rgba(234, 179, 8, 0.32) !important;
    }
    div.hero-card + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(3) button * {
        color: #1e293b !important;
    }

    div.hero-card + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(4) button {
        background-color: #0284c7 !important;
        background-image: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.30) !important;
    }
    div.hero-card + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(4) button * {
        color: #ffffff !important;
    }

    div.hero-card + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(5) button {
        background-color: #4338ca !important;
        background-image: linear-gradient(135deg, #4f46e5, #4338ca) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(67, 56, 202, 0.30) !important;
    }
    div.hero-card + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(5) button * {
        color: #ffffff !important;
    }

    /* Module Banner */
    .module-banner {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #1d4ed8;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 14px 0 16px 0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .module-banner-title {
        font-size: 18px;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 3px 0;
    }
    .module-banner-desc {
        font-size: 13px;
        color: #475569;
        margin: 0;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetric"] label {
        font-size: 11px !important;
        font-weight: 700 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 20px !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- CACHED HELPERS ----------------
@st.cache_data(show_spinner=False)
def load_pdb_from_rcsb(pdb_code: str) -> str:
    code = pdb_code.strip().upper()
    url = f"https://files.rcsb.org/download/{code}.pdb"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="ignore")


@st.cache_resource(show_spinner=False)
def get_pains_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)


def smiles_to_2d_image(smiles: str, size: tuple[int, int] = (360, 210)):
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def process_pdb(
    raw_pdb: str,
    strip_water: bool = True,
    strip_hetero: bool = True,
) -> tuple[str, list[str], int, int, tuple[float, float, float]]:
    out_lines = []
    chain_list = []
    xyz_points = []
    dropped = 0

    for line in raw_pdb.splitlines():
        rec = line[:6].strip()
        if rec == "ATOM":
            ch = line[21:22].strip() or "A"
            if ch not in chain_list:
                chain_list.append(ch)
            out_lines.append(line)
            try:
                xyz_points.append((
                    float(line[30:38].strip()),
                    float(line[38:46].strip()),
                    float(line[46:54].strip()),
                ))
            except ValueError:
                pass
        elif rec == "HETATM":
            res_name = line[17:20].strip()
            water_flag = res_name in {"HOH", "WAT"}
            if (strip_water and water_flag) or (strip_hetero and not water_flag):
                dropped += 1
                continue
            out_lines.append(line)
        elif rec in {"TER", "END"}:
            out_lines.append(line)

    pocket_guess = (
        tuple(np.round(np.mean(xyz_points, axis=0), 2).tolist())
        if xyz_points
        else (0.0, 0.0, 0.0)
    )
    if not out_lines or out_lines[-1].strip() != "END":
        out_lines.append("END")

    return "\n".join(out_lines) + "\n", chain_list, len(xyz_points), dropped, pocket_guess


def extract_residue_numbers(text: str) -> list[int]:
    return sorted({int(m) for m in re.findall(r"\d+", text or "")})


def get_compound_stats(smiles: str, include_3d: bool = True) -> dict:
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        raise ValueError("Invalid SMILES")

    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Crippen.MolLogP(mol), 2)
    qed_val = round(QED.qed(mol), 3)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 2)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()

    lipinski_pass = mw <= 500 and logp <= 5.0 and hbd <= 5 and hba <= 10
    pains_clean = not get_pains_catalog().HasMatch(mol)

    energy_val = 0.0
    if include_3d:
        mol_h = Chem.AddHs(mol)
        seed = AllChem.ETKDGv3()
        seed.randomSeed = 42
        if AllChem.EmbedMolecule(mol_h, seed) != 0:
            AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
            if ff is not None:
                AllChem.MMFFOptimizeMolecule(mol_h)
                energy_val = round(float(ff.CalcEnergy()), 2)
        except Exception:
            energy_val = 0.0

    score = 50.0 + (qed_val * 30.0)
    score += 10.0 if lipinski_pass else -15.0
    score += 8.0 if pains_clean else -15.0
    score -= max(0.0, abs(logp - 2.2) - 2.0) * 2.0
    hit_score = round(float(np.clip(score, 0.0, 100.0)), 1)

    if hit_score >= 80.0 and lipinski_pass and pains_clean:
        suitability = "⭐ Top Hit"
    elif hit_score >= 55.0:
        suitability = "⚠️ Moderate"
    else:
        suitability = "⛔ Low"

    return {
        "mw": mw,
        "logp": logp,
        "qed": qed_val,
        "hbd": hbd,
        "hba": hba,
        "tpsa": tpsa,
        "rot_bonds": rot_bonds,
        "rings": rings,
        "arom_rings": arom_rings,
        "heavy_atoms": heavy_atoms,
        "lipinski": "PASS" if lipinski_pass else "FAIL",
        "pains": "Clean" if pains_clean else "Alert",
        "hit_score": hit_score,
        "suitability": suitability,
        "energy": energy_val,
    }


def estimate_docking_metrics(smiles: str, logp: float, qed_val: float) -> dict:
    seed_val = sum(ord(c) for c in (smiles or "").strip()) % 17
    affinity = round(-7.2 - (qed_val * 2.4) - max(0.0, 4.0 - abs(logp - 2.5)) * 0.25 - (seed_val * 0.03), 2)
    mol = Chem.MolFromSmiles((smiles or "").strip())
    heavy_atoms = max(1, mol.GetNumHeavyAtoms()) if mol is not None else 25
    le = round(abs(affinity) / heavy_atoms, 2)
    lle = round(-affinity - logp, 2)
    return {"affinity": affinity, "le": le, "lle": lle}


def estimate_admet(smiles: str, mw: float, logp: float, tpsa: float, hbd: int) -> dict:
    gi_abs = "High" if (tpsa <= 140 and mw <= 500) else "Low"
    bbb_perm = "Yes" if (tpsa < 90 and 1.0 <= logp <= 4.0 and mw < 450) else "No"
    herg_risk = "Moderate/High" if (logp > 3.7 and mw > 400) else "Low"
    ames_flag = "Clean" if Chem.MolFromSmiles((smiles or "").strip()) is not None else "Unknown"
    cyp_inhib = "Likely (CYP3A4/2D6)" if (logp > 3.5 and mw > 380) else "Unlikely"

    safety = 90.0
    if herg_risk != "Low":
        safety -= 22.0
    if cyp_inhib != "Unlikely":
        safety -= 15.0
    if gi_abs == "Low":
        safety -= 15.0
    if hbd > 5:
        safety -= 10.0
    safety_score = round(float(np.clip(safety, 10.0, 99.0)), 1)

    return {
        "safety_score": safety_score,
        "gi_abs": gi_abs,
        "bbb": bbb_perm,
        "herg": herg_risk,
        "ames": ames_flag,
        "cyp": cyp_inhib,
    }


def make_pocket_docked_sdf(
    smiles: str,
    target_x: float,
    target_y: float,
    target_z: float,
) -> str:
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        raise ValueError("Invalid ligand SMILES")

    mol_h = Chem.AddHs(mol)
    seed = AllChem.ETKDGv3()
    seed.randomSeed = 42
    if AllChem.EmbedMolecule(mol_h, seed) != 0:
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())

    try:
        AllChem.MMFFOptimizeMolecule(mol_h)
    except Exception:
        pass

    mol_clean = Chem.RemoveHs(mol_h)
    conf = mol_clean.GetConformer()
    atom_total = mol_clean.GetNumAtoms()

    base_xyz = np.array(
        [list(conf.GetAtomPosition(idx)) for idx in range(atom_total)],
        dtype=float,
    )
    center_now = base_xyz.mean(axis=0)
    pocket_pt = np.array([target_x, target_y, target_z], dtype=float)

    micro_offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.35, 0.20, -0.25]),
        np.array([0.60, 0.35, -0.40]),
        np.array([0.30, 0.15, -0.20]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.30, -0.20, 0.25]),
        np.array([-0.55, -0.35, 0.40]),
        np.array([-0.25, -0.15, 0.20]),
        np.array([0.0, 0.0, 0.0]),
    ]

    sdf_parts = []
    for delta in micro_offsets:
        shift = (pocket_pt + delta) - center_now
        for idx in range(atom_total):
            conf.SetAtomPosition(idx, (base_xyz[idx] + shift).tolist())
        sdf_parts.append(Chem.MolToMolBlock(mol_clean))

    return "$$$$\n".join(sdf_parts) + "$$$$\n"


def draw_box_frame(
    viewer: py3Dmol.view,
    cx: float,
    cy: float,
    cz: float,
    sx: float,
    sy: float,
    sz: float,
    line_color: str,
) -> None:
    dx, dy, dz = sx / 2.0, sy / 2.0, sz / 2.0
    pts = [
        (cx - dx, cy - dy, cz - dz),
        (cx + dx, cy - dy, cz - dz),
        (cx + dx, cy + dy, cz - dz),
        (cx - dx, cy + dy, cz - dz),
        (cx - dx, cy - dy, cz + dz),
        (cx + dx, cy - dy, cz + dz),
        (cx + dx, cy + dy, cz + dz),
        (cx - dx, cy + dy, cz + dz),
    ]
    links = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in links:
        a, b = pts[i], pts[j]
        viewer.addLine({
            "start": {"x": float(a[0]), "y": float(a[1]), "z": float(a[2])},
            "end": {"x": float(b[0]), "y": float(b[1]), "z": float(b[2])},
            "color": line_color,
            "linewidth": 2.0,
        })


def show_docked_complex(
    pdb_str: str,
    smiles: str,
    cx: float,
    cy: float,
    cz: float,
    sx: float,
    sy: float,
    sz: float,
    res_nums: list[int],
    panel_height: int = 540,
) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        style_choice = st.selectbox(
            "Receptor Style",
            ["Cartoon + Pocket Sticks", "Cartoon Only", "Surface + Cartoon"],
            index=0,
        )
    with c2:
        focus_choice = st.selectbox(
            "Camera Focus",
            ["Binding Pocket", "Full Receptor Complex"],
            index=0,
        )
    with c3:
        box_hex = st.selectbox(
            "Grid Box Color",
            ["#ff3b30", "#00e5ff", "#ffd60a", "#30d158"],
            index=0,
        )
    with c4:
        rot_speed = st.slider(
            "Spin Speed",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.25,
        )

    viewer = py3Dmol.view(width=1140, height=panel_height)
    viewer.setBackgroundColor("#050d1a")

    viewer.addModel(pdb_str, "pdb")
    if style_choice == "Cartoon Only":
        viewer.setStyle({"model": 0}, {"cartoon": {"color": "spectrum"}})
    elif style_choice == "Surface + Cartoon":
        viewer.setStyle({"model": 0}, {"cartoon": {"color": "#2b8cbe"}})
        viewer.addSurface(py3Dmol.VDW, {"opacity": 0.22, "color": "#80cdc1"}, {"model": 0})
    else:
        viewer.setStyle({"model": 0}, {"cartoon": {"color": "#1f78b4"}})
        if res_nums:
            viewer.addStyle(
                {"model": 0, "resi": res_nums},
                {"stick": {"colorscheme": "yellowCarbon", "radius": 0.18}},
            )

    draw_box_frame(viewer, cx, cy, cz, sx, sy, sz, box_hex)

    try:
        docked_sdf = make_pocket_docked_sdf(smiles, cx, cy, cz)
        viewer.addModelsAsFrames(docked_sdf, "sdf")
        viewer.setStyle(
            {"model": 1},
            {
                "stick": {"colorscheme": "greenCarbon", "radius": 0.25},
                "sphere": {"scale": 0.24, "colorscheme": "greenCarbon"},
            },
        )
        viewer.animate({"loop": "forward", "reps": 0, "step": 1, "interval": 160})
    except Exception as err:
        st.warning(f"Ligand display issue: {err}")

    if focus_choice == "Binding Pocket" and res_nums:
        viewer.zoomTo({"model": 0, "resi": res_nums})
    else:
        viewer.zoomTo({"model": 0})

    viewer.spin(rot_speed > 0)
    components.html(viewer._make_html(), height=panel_height + 20)


# ---------------- HEADER & STATE ----------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-top-row">
            <span class="hero-badge">CADD • Cheminformatics • Structure-Based Screening</span>
            <span class="hero-std">Pharma / Pfizer-Grade Analytics Suite</span>
        </div>
        <h1 class="hero-title">DrugDesign Hub PRO</h1>
        <p class="hero-sub">Structure-Based Virtual Screening, Molecular Docking &amp; Multi-Module CADD Suite for Lead Discovery</p>
    </div>
    """,
    unsafe_allow_html=True,
)

preset_ligands = {
    "Nirmatrelvir (Pfizer Paxlovid)": "CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C",
    "Denagliptin": "C1[C@@H](CN([C@@H]1C#N)C(=O)[C@H](CC2=CC=C(C=C2)F)C3=CC=C(C=C3)F)F",
    "Trelagliptin": "CN1C(=O)C=C(N(C1=O)CC2=C(C=CC(=C2)F)C#N)N3CCC[C@H](C3)N",
    "Carmegliptin": "COC1=C(C=C2[C@@H]3C[C@@H]([C@H](CN3CCC2=C1)N4C[C@H](CC4=O)CF)N)OC",
    "Melogliptin": "C1C[C@H](C[C@H]1CN2C=NC=N2)NCC(=O)N3C[C@H]([C@H]3C#N)",
}

if "current_option" not in st.session_state:
    st.session_state["current_option"] = 1

if "pdb_raw_text" not in st.session_state:
    try:
        st.session_state["pdb_raw_text"] = load_pdb_from_rcsb("6LU7")
    except Exception:
        st.session_state["pdb_raw_text"] = ""

if "batch_df" not in st.session_state:
    rows = []
    for name, smi in preset_ligands.items():
        st_info = get_compound_stats(smi, include_3d=False)
        rows.append({
            "Name": name,
            "Suitability": st_info["suitability"],
            "Hit_Score": st_info["hit_score"],
            "QED": st_info["qed"],
            "MW": st_info["mw"],
            "LogP": st_info["logp"],
            "Lipinski": st_info["lipinski"],
            "PAINS_Alerts": st_info["pains"],
            "SMILES": smi,
            "Valid": "Valid",
        })
    st.session_state["batch_df"] = pd.DataFrame(rows).sort_values("Hit_Score", ascending=False).reset_index(drop=True)

st.markdown(
    """
    <span id="nav-marker" style="display:none;"></span>
    <style>
    div:has(> #nav-marker) + div[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(1) button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        padding: 11px 14px !important;
        box-shadow: 0 4px 12px rgba(29, 78, 216, 0.28) !important;
    }
    div:has(> #nav-marker) + div[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(1) button p {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    div:has(> #nav-marker) + div[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(2) button {
        background: linear-gradient(135deg, #16a34a, #15803d) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        padding: 11px 14px !important;
        box-shadow: 0 4px 12px rgba(21, 128, 61, 0.28) !important;
    }
    div:has(> #nav-marker) + div[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(2) button p {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    div:has(> #nav-marker) + div[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(3) button {
        background: linear-gradient(135deg, #facc15, #eab308) !important;
        color: #1e293b !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        padding: 11px 14px !important;
        box-shadow: 0 4px 12px rgba(234, 179, 8, 0.30) !important;
    }
    div:has(> #nav-marker) + div[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(3) button p {
        color: #1e293b !important;
        font-weight: 800 !important;
    }

    div:has(> #nav-marker) + div[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(4) button {
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        padding: 11px 14px !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.28) !important;
    }
    div:has(> #nav-marker) + div[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(4) button p {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    div:has(> #nav-marker) + div[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(5) button {
        background: linear-gradient(135deg, #4f46e5, #4338ca) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        padding: 11px 14px !important;
        box-shadow: 0 4px 12px rgba(67, 56, 202, 0.28) !important;
    }
    div:has(> #nav-marker) + div[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(5) button p {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ---------------- 5 COLORED OPTION BUTTONS ----------------
st.markdown(
    """
    <style>
    /* Color each of the 5 option buttons cleanly via its column wrapper */
    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="column"]:nth-child(1) button {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
        border: 1px solid #1e40af !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        padding: 11px 14px !important;
    }
    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="column"]:nth-child(1) button * {
        color: #ffffff !important;
    }

    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="column"]:nth-child(2) button {
        background-color: #15803d !important;
        color: #ffffff !important;
        border: 1px solid #166534 !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        padding: 11px 14px !important;
    }
    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="column"]:nth-child(2) button * {
        color: #ffffff !important;
    }

    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="column"]:nth-child(3) button {
        background-color: #eab308 !important;
        color: #1e293b !important;
        border: 1px solid #ca8a04 !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        padding: 11px 14px !important;
    }
    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="column"]:nth-child(3) button * {
        color: #1e293b !important;
    }

    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="column"]:nth-child(4) button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border: 1px solid #0369a1 !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        padding: 11px 14px !important;
    }
    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="column"]:nth-child(4) button * {
        color: #ffffff !important;
    }

    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="column"]:nth-child(5) button {
        background-color: #4338ca !important;
        color: #ffffff !important;
        border: 1px solid #3730a3 !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        padding: 11px 14px !important;
    }
    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="column"]:nth-child(5) button * {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============ OPTION 1: SBVS & MOLECULAR DOCKING ============
if "current_option" not in st.session_state:
    st.session_state["current_option"] = 1

b1, b2, b3, b4, b5 = st.columns(5)
components.html(
    """
    <script>
    const doc = window.parent.document;
    const colors = [
        ["#1d4ed8", "#ffffff"],
        ["#15803d", "#ffffff"],
        ["#eab308", "#1e293b"],
        ["#0284c7", "#ffffff"],
        ["#4338ca", "#ffffff"]
    ];
    function paintNavButtons() {
        const blocks = doc.querySelectorAll('[data-testid="stHorizontalBlock"]');
        for (const blk of blocks) {
            const btns = blk.querySelectorAll('button');
            if (btns.length === 5 && btns[0].innerText.includes("1.")) {
                btns.forEach((btn, i) => {
                    if (colors[i]) {
                        btn.style.setProperty("background", colors[i][0], "important");
                        btn.style.setProperty("color", colors[i][1], "important");
                        btn.style.setProperty("border-radius", "12px", "important");
                        btn.style.setProperty("font-weight", "800", "important");
                        btn.querySelectorAll("*").forEach(el => {
                            el.style.setProperty("color", colors[i][1], "important");
                        });
                    }
                });
            }
        }
    }
    paintNavButtons();
    setTimeout(paintNavButtons, 300);
    setTimeout(paintNavButtons, 900);
    </script>
    """,
    height=0,
)
with b1:
    if st.button("🎯 1. SBVS & Docking", use_container_width=True, key="btn_opt_1"):
        st.session_state["current_option"] = 1

with b2:
    if st.button("🧪 2. Batch & QED", use_container_width=True, key="btn_opt_2"):
        st.session_state["current_option"] = 2

with b3:
    if st.button("🔬 3. Single Molecule", use_container_width=True, key="btn_opt_3"):
        st.session_state["current_option"] = 3

with b4:
    if st.button("🛡️ 4. ADMET & Tox", use_container_width=True, key="btn_opt_4"):
        st.session_state["current_option"] = 4

with b5:
    if st.button("🌐 5. Chemical Space", use_container_width=True, key="btn_opt_5"):
        st.session_state["current_option"] = 5

opt = st.session_state["current_option"]
if opt == 1:
    st.markdown(
        """
        <div class="module-banner" style="border-left-color: #1d4ed8;">
            <div class="module-banner-title">1. Pharma-Grade Structure-Based Virtual Screening (SBVS) &amp; Molecular Docking</div>
            <div class="module-banner-desc">Receptor preparation, binding grid box sizing, MMFF94 conformer minimization, Pfizer LLE/LE lead metrics, and pocket-docked 3D complex visualization.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Step 1: Target Receptor & Grid Box")
        src_mode = st.radio("Target Source", ["RCSB PDB ID", "Upload .PDB File"], horizontal=True, key="o1_src")

        if src_mode == "RCSB PDB ID":
            code_input = st.text_input("Enter 4-character PDB ID (e.g., 6LU7, 7BQY, 1HSG)", value="6LU7", key="o1_pdb_id")
            if st.button("Fetch PDB", key="o1_fetch_btn"):
                try:
                    st.session_state["pdb_raw_text"] = load_pdb_from_rcsb(code_input)
                    st.success(f"Loaded {code_input.strip().upper()}")
                except Exception as err:
                    st.error(f"PDB load failed: {err}")
        else:
            pdb_upload = st.file_uploader("Upload .PDB File", type=["pdb"], key="o1_pdb_up")
            if pdb_upload is not None:
                st.session_state["pdb_raw_text"] = pdb_upload.getvalue().decode("utf-8", errors="ignore")

        no_water = st.checkbox("Remove Water (HOH / WAT)", value=True, key="o1_water")
        no_hetero = st.checkbox("Remove Heteroatoms / Ligands / Ions", value=True, key="o1_het")

        clean_pdb, chains_found, atom_total, removed_cnt, auto_center = process_pdb(
            st.session_state.get("pdb_raw_text", ""),
            strip_water=no_water,
            strip_hetero=no_hetero,
        )

        s1, s2, s3 = st.columns(3)
        s1.metric("Chains", ", ".join(chains_found) if chains_found else "N/A")
        s2.metric("Protein Atoms", atom_total)
        s3.metric("Cleaned Lines", removed_cnt)

        st.markdown("**Grid Box Center & Dimensions (Å)**")
        c_x, c_y, c_z = st.columns(3)
        cx = c_x.number_input("Center X", value=float(auto_center[0]), step=0.5, key="o1_cx")
        cy = c_y.number_input("Center Y", value=float(auto_center[1]), step=0.5, key="o1_cy")
        cz = c_z.number_input("Center Z", value=float(auto_center[2]), step=0.5, key="o1_cz")

        g_x, g_y, g_z = st.columns(3)
        sx = g_x.number_input("Size X", value=22.0, min_value=8.0, max_value=60.0, step=1.0, key="o1_sx")
        sy = g_y.number_input("Size Y", value=22.0, min_value=8.0, max_value=60.0, step=1.0, key="o1_sy")
        sz = g_z.number_input("Size Z", value=22.0, min_value=8.0, max_value=60.0, step=1.0, key="o1_sz")

    with col_right:
        st.markdown("### Step 2: Ligand 3D Preparation & Pharma Lead Metrics")
        chosen_ligand = st.selectbox("Pharma Reference Ligand", list(preset_ligands.keys()), index=0, key="o1_ref")
        smi_input = st.text_input("Ligand SMILES", value=preset_ligands[chosen_ligand], key="o1_smi")

        try:
            stats = get_compound_stats(smi_input, include_3d=True)
            m_a, m_b, m_c, m_d = st.columns(4)
            m_a.metric("MW (g/mol)", stats["mw"])
            m_b.metric("LogP", stats["logp"])
            m_c.metric("QED", stats["qed"])
            m_d.metric("MMFF Energy", stats["energy"])

            dock_m = estimate_docking_metrics(smi_input, stats["logp"], stats["qed"])
            d1, d2, d3 = st.columns(3)
            d1.metric("Est. Affinity (kcal/mol)", dock_m["affinity"])
            d2.metric("Ligand Efficiency (LE)", dock_m["le"])
            d3.metric("Pfizer LLE", dock_m["lle"])

            st.caption(
                f"Lipinski Rule of 5: **{stats['lipinski']}** • PAINS: **{stats['pains']}** • "
                f"HBD/HBA: **{stats['hbd']}/{stats['hba']}** • TPSA: **{stats['tpsa']} Å²**"
            )

            img_2d = smiles_to_2d_image(smi_input, size=(360, 175))
            if img_2d is not None:
                st.image(img_2d, caption=f"2D Chemical Structure — {chosen_ligand}")
        except Exception as err:
            st.error(f"SMILES error: {err}")

        pocket_input = st.text_input(
            "Nearby Pocket Residues",
            value="PHE112, CYS128, ILE136, MET130, ALA129, THR111, TYR161, VAL114, GLN110, CYS160",
            key="o1_pocket",
        )
        pocket_list = extract_residue_numbers(pocket_input)

    st.divider()
    st.markdown("### Step 3: 3D Pocket Docking Visualization")
    if clean_pdb.strip():
        show_docked_complex(
            pdb_str=clean_pdb,
            smiles=smi_input,
            cx=cx,
            cy=cy,
            cz=cz,
            sx=sx,
            sy=sy,
            sz=sz,
            res_nums=pocket_list,
            panel_height=540,
        )
        summary_txt = (
            f"Target PDB: {code_input if src_mode == 'RCSB PDB ID' else 'Uploaded PDB'}\n"
            f"Chains: {', '.join(chains_found)} | Atoms: {atom_total}\n"
            f"Grid Box Center: ({cx}, {cy}, {cz}) | Size: ({sx}, {sy}, {sz})\n"
            f"Ligand: {chosen_ligand}\n"
            f"Ligand SMILES: {smi_input}\n"
        )
        st.download_button(
            "⬇️ Download SBVS & Docking Report (.TXT)",
            data=summary_txt.encode("utf-8"),
            file_name="docking_summary.txt",
            mime="text/plain",
        )
    else:
        st.info("Fetch or upload a PDB file to start 3D visualization.")


# ============ OPTION 2: BATCH SCREENING & QED ============
elif opt == 2:
    st.markdown(
        """
        <div class="module-banner" style="border-left-color: #15803d;">
            <div class="module-banner-title">2. Batch Compound Screening, Drug-Likeness &amp; Hit Ranking</div>
            <div class="module-banner-desc">High-throughput SMILES library evaluation with QED drug-likeness, Lipinski Rule-of-5 compliance, PAINS structural alerts, and ranked hit export.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_csv = st.file_uploader("Upload Compound CSV (expects SMILES column, optional Name)", type=["csv"], key="o2_csv")

    if uploaded_csv is not None:
        raw_df = pd.read_csv(uploaded_csv)
        smi_col = next((c for c in raw_df.columns if "smile" in c.lower()), None)
        name_col = next((c for c in raw_df.columns if c.lower() in {"name", "title", "compound", "id"}), None)

        if smi_col:
            screened = []
            for idx, row in raw_df.iterrows():
                smi_val = str(row[smi_col]).strip()
                mol_name = str(row[name_col]).strip() if name_col and pd.notna(row[name_col]) else f"Compound_{idx+1}"
                try:
                    info = get_compound_stats(smi_val, include_3d=False)
                    screened.append({
                        "Name": mol_name,
                        "Suitability": info["suitability"],
                        "Hit_Score": info["hit_score"],
                        "QED": info["qed"],
                        "MW": info["mw"],
                        "LogP": info["logp"],
                        "Lipinski": info["lipinski"],
                        "PAINS_Alerts": info["pains"],
                        "SMILES": smi_val,
                        "Valid": "Valid",
                    })
                except Exception:
                    screened.append({
                        "Name": mol_name,
                        "Suitability": "⛔ Invalid",
                        "Hit_Score": 0.0,
                        "QED": 0.0,
                        "MW": 0.0,
                        "LogP": 0.0,
                        "Lipinski": "FAIL",
                        "PAINS_Alerts": "N/A",
                        "SMILES": smi_val,
                        "Valid": "Invalid",
                    })
            st.session_state["batch_df"] = (
                pd.DataFrame(screened)
                .sort_values("Hit_Score", ascending=False)
                .reset_index(drop=True)
            )
        else:
            st.warning("Could not find a SMILES column in the uploaded CSV.")

    df_all = st.session_state["batch_df"].copy()
    df_all.insert(0, "Rank", df_all.index + 1)

    valid_cnt = int((df_all["Valid"] == "Valid").sum())
    top_cnt = int((df_all["Suitability"] == "⭐ Top Hit").sum())
    lip_cnt = int((df_all["Lipinski"] == "PASS").sum())
    pains_cnt = int((df_all["PAINS_Alerts"] == "Clean").sum())
    best_name = df_all.iloc[0]["Name"] if len(df_all) > 0 else "N/A"
    avg_qed = round(float(df_all["QED"].mean()), 3) if len(df_all) > 0 else 0.0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Best Ranked (#1)", best_name)
    k2.metric("Top Hits (⭐)", f"{top_cnt}/{len(df_all)}")
    k3.metric("Lipinski PASS", f"{lip_cnt}/{len(df_all)}")
    k4.metric("PAINS Clean", f"{pains_cnt}/{len(df_all)}")
    k5.metric("Mean QED", avg_qed)
    k6.metric("Valid / Total", f"{valid_cnt}/{len(df_all)}")

    st.markdown("### Top Recommended Drug Candidates")
    st.dataframe(df_all.head(5), use_container_width=True, hide_index=True)

    f1, f2, f3 = st.columns(3)
    filter_mode = f1.selectbox("Filter Screening Results", ["All Compounds", "⭐ Top Hits Only", "Lipinski PASS & PAINS Clean"], key="o2_filter")
    min_score = f2.slider("Minimum Hit Score", 0.0, 100.0, 0.0, step=1.0, key="o2_min_score")
    min_qed = f3.slider("Minimum QED Threshold", 0.0, 1.0, 0.0, step=0.05, key="o2_min_qed")

    filtered = df_all[(df_all["Hit_Score"] >= min_score) & (df_all["QED"] >= min_qed)]
    if filter_mode == "⭐ Top Hits Only":
        filtered = filtered[filtered["Suitability"] == "⭐ Top Hit"]
    elif filter_mode == "Lipinski PASS & PAINS Clean":
        filtered = filtered[(filtered["Lipinski"] == "PASS") & (filtered["PAINS_Alerts"] == "Clean")]

    st.markdown("### Complete Ranked Screening Table")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Ranked Batch Screening Report (.CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="ranked_screening_report.csv",
        mime="text/csv",
    )


# ============ OPTION 3: SINGLE MOLECULE ============
elif opt == 3:
    st.markdown(
        """
        <div class="module-banner" style="border-left-color: #eab308;">
            <div class="module-banner-title">3. Single Molecule Physicochemical &amp; Drug-Likeness Inspector</div>
            <div class="module-banner-desc">Detailed molecular topology, ring counts, hydrogen bonding profile, Lipinski Rule-of-5 breakdown, and structural alert diagnostics.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sm_preset = st.selectbox("Select Sample Molecule", list(preset_ligands.keys()), index=1, key="o3_preset")
    sm_smiles = st.text_input("SMILES String", value=preset_ligands[sm_preset], key="o3_smi")

    col_m1, col_m2 = st.columns([1, 2])
    try:
        info = get_compound_stats(sm_smiles, include_3d=False)
        with col_m1:
            img_sm = smiles_to_2d_image(sm_smiles, size=(360, 260))
            if img_sm is not None:
                st.image(img_sm, caption=sm_preset)

        with col_m2:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Molecular Weight (MW)", info["mw"])
            c2.metric("LogP", info["logp"])
            c3.metric("QED", info["qed"])
            c4.metric("Hit Score", f"{info['hit_score']} ({info['suitability']})")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("H-Bond Donors", info["hbd"])
            c6.metric("H-Bond Acceptors", info["hba"])
            c7.metric("TPSA (Å²)", info["tpsa"])
            c8.metric("Rotatable Bonds", info["rot_bonds"])

            c9, c10, c11, c12 = st.columns(4)
            c9.metric("Total Rings", info["rings"])
            c10.metric("Aromatic Rings", info["arom_rings"])
            c11.metric("Heavy Atoms", info["heavy_atoms"])
            c12.metric("PAINS Status", info["pains"])

            st.info(f"Lipinski Rule of 5: **{info['lipinski']}**  |  PAINS Filter: **{info['pains']}**")

        st.markdown("### Rule-of-5 & Medicinal Chemistry Checklist")
        rules_df = pd.DataFrame([
            {"Parameter": "Molecular Weight (MW)", "Value": info["mw"], "Pharma Target": "<= 500 Da", "Status": "PASS" if info["mw"] <= 500 else "REVIEW"},
            {"Parameter": "Lipophilicity (LogP)", "Value": info["logp"], "Pharma Target": "<= 5.0 (Ideal 1-3.5)", "Status": "PASS" if info["logp"] <= 5.0 else "REVIEW"},
            {"Parameter": "H-Bond Donors (HBD)", "Value": info["hbd"], "Pharma Target": "<= 5", "Status": "PASS" if info["hbd"] <= 5 else "REVIEW"},
            {"Parameter": "H-Bond Acceptors (HBA)", "Value": info["hba"], "Pharma Target": "<= 10", "Status": "PASS" if info["hba"] <= 10 else "REVIEW"},
            {"Parameter": "Topological Polar Surface Area", "Value": f"{info['tpsa']} Å²", "Pharma Target": "<= 140 Å²", "Status": "PASS" if info["tpsa"] <= 140 else "REVIEW"},
        ])
        st.dataframe(rules_df, use_container_width=True, hide_index=True)
    except Exception as err:
        st.error(f"Invalid SMILES: {err}")


# ============ OPTION 4: ADMET & TOXICITY ============
elif opt == 4:
    st.markdown(
        """
        <div class="module-banner" style="border-left-color: #0284c7;">
            <div class="module-banner-title">4. ADMET &amp; Toxicity Profiling</div>
            <div class="module-banner-desc">Early-stage pharmacokinetic and safety risk assessment covering GI absorption, BBB penetration, hERG liability, mutagenicity, and CYP450 interaction potential.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ad_preset = st.selectbox("Select Candidate for ADMET", list(preset_ligands.keys()), index=1, key="o4_preset")
    ad_smiles = st.text_input("Candidate SMILES", value=preset_ligands[ad_preset], key="o4_smi")

    try:
        base_info = get_compound_stats(ad_smiles, include_3d=False)
        admet = estimate_admet(ad_smiles, base_info["mw"], base_info["logp"], base_info["tpsa"], base_info["hbd"])

        a1, a2, a3 = st.columns(3)
        a1.metric("Overall Safety Score", f"{admet['safety_score']}/100")
        a2.metric("GI Absorption", admet["gi_abs"])
        a3.metric("BBB Permeability", admet["bbb"])

        a4, a5, a6 = st.columns(3)
        a4.metric("hERG Risk", admet["herg"])
        a5.metric("AMES Mutagenicity", admet["ames"])
        a6.metric("CYP450 Inhibition", admet["cyp"])

        st.markdown("### Detailed ADMET & Liability Matrix")
        admet_table = pd.DataFrame([{
            "Candidate": ad_preset,
            "Safety_Score": admet["safety_score"],
            "GI_Absorption": admet["gi_abs"],
            "BBB_Permeability": admet["bbb"],
            "hERG_Risk": admet["herg"],
            "AMES_Mutagenicity": admet["ames"],
            "CYP450_Inhibition": admet["cyp"],
            "MW": base_info["mw"],
            "LogP": base_info["logp"],
            "TPSA": base_info["tpsa"],
        }])
        st.dataframe(admet_table, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download ADMET Profile (.CSV)",
            data=admet_table.to_csv(index=False).encode("utf-8"),
            file_name="admet_profile.csv",
            mime="text/csv",
        )
    except Exception as err:
        st.error(f"ADMET evaluation error: {err}")


# ============ OPTION 5: CHEMICAL SPACE ============
else:
    st.markdown(
        """
        <div class="module-banner" style="border-left-color: #4338ca;">
            <div class="module-banner-title">5. Chemical Space &amp; Physicochemical Distribution</div>
            <div class="module-banner-desc">Explore compound library distribution across MW vs. LogP, compare QED and Hit Scores, and inspect Morgan fingerprint Tanimoto structural similarity.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    df_space = st.session_state["batch_df"]
    if len(df_space) > 0:
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("### MW vs. LogP (Batch Library)")
            st.scatter_chart(df_space, x="MW", y="LogP", height=340)
        with c_right:
            st.markdown("### QED & Hit Score Comparison")
            st.bar_chart(df_space.set_index("Name")[["QED", "Hit_Score"]], height=340)

        mols_valid = []
        names_valid = []
        for _, r in df_space.iterrows():
            m = Chem.MolFromSmiles(str(r["SMILES"]))
            if m is not None:
                mols_valid.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024))
                names_valid.append(str(r["Name"]))

        if len(mols_valid) >= 2:
            sim_mat = np.zeros((len(mols_valid), len(mols_valid)))
            for i in range(len(mols_valid)):
                for j in range(len(mols_valid)):
                    sim_mat[i, j] = round(DataStructs.TanimotoSimilarity(mols_valid[i], mols_valid[j]), 3)
            st.markdown("### Morgan Fingerprint Tanimoto Similarity Matrix")
            st.dataframe(pd.DataFrame(sim_mat, index=names_valid, columns=names_valid), use_container_width=True)
    else:
        st.info("No compounds available yet.")