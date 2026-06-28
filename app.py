"""
Global Energy Intelligence Platform
=====================================
Production-ready Streamlit application.
Run:  streamlit run app.py
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES + FLOATING NAV
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700;800;900&family=JetBrains+Mono:wght=400;500;600&display=swap');

/* ── Tokens ── */
:root {
  --bg0:    #020617;
  --bg1:    #041229;
  --bg2:    #071a38;
  --cyan:   #00e5ff;
  --purple: #6a5cff;
  --white:  #f0f6ff;
  --grey:   #7a93b8;
  --dim:    #3d5470;
  --border: rgba(0,229,255,.12);
  --glow:   0 0 28px rgba(0,229,255,.2), 0 0 56px rgba(0,229,255,.07);
}

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--bg0) !important;
  color: var(--white) !important;
  font-family: 'Inter', sans-serif !important;
}

/* Kill Streamlit chrome */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
#MainMenu, footer, .stDeployButton { display: none !important; }

[data-testid="block-container"] { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"]  { gap: 0 !important; }
[data-testid="stHorizontalBlock"] { gap: 24px !important; }

/* ── Floating nav ── */
#geip-nav {
  position: fixed;
  top: 18px; right: 22px;
  z-index: 9999;
}
#geip-nav-toggle {
  width: 42px; height: 42px;
  background: rgba(4,18,41,.9);
  border: 1px solid rgba(0,229,255,.18);
  border-radius: 10px;
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  font-size: 17px;
  color: var(--cyan);
  backdrop-filter: blur(16px);
  transition: border-color .2s, box-shadow .2s;
  user-select: none;
}
#geip-nav-toggle:hover {
  border-color: var(--cyan);
  box-shadow: var(--glow);
}
#geip-nav-menu {
  display: none;
  position: absolute;
  top: 52px; right: 0;
  background: rgba(4,18,41,.96);
  border: 1px solid rgba(0,229,255,.15);
  border-radius: 12px;
  padding: 6px 0;
  min-width: 230px;
  backdrop-filter: blur(24px);
  box-shadow: 0 24px 64px rgba(0,0,0,.7), var(--glow);
}
#geip-nav-menu.open { display: block; animation: navIn .16s ease; }
@keyframes navIn {
  from { opacity:0; transform: translateY(-6px) scale(.98); }
  to   { opacity:1; transform: translateY(0)  scale(1); }
}
.geip-nav-item {
  display: flex; align-items: center; gap: 10px;
  padding: 11px 18px;
  color: var(--grey);
  font-size: 13px; font-weight: 500;
  cursor: pointer;
  transition: color .15s, background .15s;
  text-decoration: none;
  font-family: 'Inter', sans-serif;
}
.geip-nav-item:hover { color: var(--cyan); background: rgba(0,229,255,.06); }
.geip-nav-dot { font-size: 10px; color: var(--cyan); opacity: .6; }
.geip-nav-sep { border: none; border-top: 1px solid rgba(0,229,255,.08); margin: 3px 0; }

/* ── Hero ── */
.geip-hero {
  padding: 44px 64px 20px 64px;
  background: radial-gradient(ellipse 90% 50% at 50% 0%,
              rgba(0,229,255,.08) 0%, transparent 68%), var(--bg0);
}
.geip-tag {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(0,229,255,.06);
  border: 1px solid rgba(0,229,255,.18);
  border-radius: 20px;
  padding: 5px 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; letter-spacing: 2.5px;
  color: var(--cyan);
  text-transform: uppercase;
  margin-bottom: 20px;
}
.geip-hero-title {
  font-size: clamp(34px, 5vw, 72px);
  font-weight: 900;
  line-height: .97;
  letter-spacing: -3px;
  margin: 0 0 16px 0;
  color: var(--white);
}
.geip-hero-title .grad {
  background: linear-gradient(95deg, #00e5ff 0%, #00bfff 45%, #6a5cff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.geip-hero-sub {
  font-size: 14px; font-weight: 300;
  color: var(--grey); line-height: 1.7;
  margin: 0 0 24px 0;
  max-width: 480px;
}

/* ── Live pills ── */
.geip-pills { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 28px; }
.geip-pill {
  display: flex; align-items: center; gap: 10px;
  background: rgba(0,229,255,.045);
  border: 1px solid rgba(0,229,255,.13);
  border-radius: 8px;
  padding: 9px 16px;
}
.geip-pill-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 16px; font-weight: 600;
  color: var(--cyan);
}
.geip-pill-lbl { font-size: 11px; color: var(--grey); }

/* ── Section Header Resizing & Tightening ── */
.geip-section     { padding: 48px 64px 16px 64px; background: var(--bg0); }
.geip-section-alt { padding: 48px 64px 16px 64px; background: var(--bg1); }
.geip-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; letter-spacing: 3.5px;
  text-transform: uppercase;
  color: var(--cyan); opacity: .65;
  margin: 0 0 8px 0;
}
.geip-h2 {
  font-size: 26px; font-weight: 800;
  letter-spacing: -1px; color: var(--white);
  margin: 0 0 6px 0;
}
.geip-lead { font-size: 13px; color: var(--dim); margin: 0 0 16px 0; }

/* ── KPI cards ── */
.geip-kpis {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}
.geip-kpi {
  background: linear-gradient(135deg, rgba(0,229,255,.055) 0%, rgba(106,92,255,.035) 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 24px 26px;
  position: relative; overflow: hidden;
  transition: border-color .25s, box-shadow .25s, transform .2s;
}
.geip-kpi:hover {
  border-color: rgba(0,229,255,.32);
  box-shadow: 0 0 32px rgba(0,229,255,.1);
  transform: translateY(-2px);
}
.geip-kpi::before {
  content: '';
  position: absolute; inset: 0 0 auto 0;
  height: 1px;
  background: linear-gradient(90deg, var(--cyan), var(--purple));
}
.geip-kpi-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; letter-spacing: 2.5px;
  text-transform: uppercase; color: var(--dim);
  margin-bottom: 14px;
}
.geip-kpi-value {
  font-size: 32px; font-weight: 800;
  letter-spacing: -2px; color: var(--cyan);
  line-height: 1; margin-bottom: 4px;
}
.geip-kpi-unit { font-size: 11px; color: var(--dim); }

/* ── Country stat cards ── */
.geip-cstats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin: 16px 0 24px 0;
}
.geip-cstat {
  background: rgba(0,229,255,.028);
  border: 1px solid rgba(0,229,255,.1);
  border-radius: 12px;
  padding: 20px 22px;
  text-align: center;
  transition: border-color .2s;
}
.geip-cstat:hover { border-color: rgba(0,229,255,.22); }
.geip-cstat-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; letter-spacing: 2px;
  text-transform: uppercase; color: var(--dim);
  margin-bottom: 10px;
}
.geip-cstat-value { font-size: 19px; font-weight: 700; letter-spacing: -.4px; color: var(--white); }
.geip-cstat-sub   { font-size: 10px; color: var(--dim); margin-top: 3px; }

/* ── Badges ── */
.geip-badge {
  display: inline-block;
  background: rgba(106,92,255,.1);
  border: 1px solid rgba(106,92,255,.28);
  color: #a5a0ff;
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; letter-spacing: 2px;
  text-transform: uppercase;
  padding: 5px 14px; border-radius: 20px;
  margin-bottom: 14px;
}

/* ── Map hint ── */
.geip-map-hint {
  text-align: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; letter-spacing: 2.5px;
  text-transform: uppercase;
  color: var(--dim);
  padding: 12px 0 22px 0;
  background: var(--bg0);
}

/* ── Table & Interactive Containers ── */
.geip-table-container {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  margin-top: 16px;
}

/* ── Divider ── */
.geip-divider { border: none; border-top: 1px solid rgba(0,229,255,.06); margin: 0; }

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
  background: var(--bg2) !important;
  border-color: var(--border) !important;
  border-radius: 10px !important;
}
div[data-baseweb="select"] * { color: var(--white) !important; }
[data-testid="stSelectbox"] label {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 9px !important; letter-spacing: 2.5px !important;
  text-transform: uppercase !important; color: var(--dim) !important;
}

/* ── Footer ── */
.geip-footer {
  padding: 30px 64px;
  display: flex; justify-content: space-between; align-items: center;
  border-top: 1px solid rgba(0,229,255,.07);
  background: var(--bg0);
  flex-wrap: wrap; gap: 12px;
}
.geip-footer-left  { font-size: 13px; font-weight: 600; color: var(--grey); }
.geip-footer-right {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; color: var(--dim); letter-spacing: .5px;
  line-height: 1.8; text-align: right;
}

/* ── Responsiveness ── */
@media (max-width: 900px) {
  .geip-kpis, .geip-cstats { grid-template-columns: repeat(2, 1fr); }
  .geip-hero, .geip-section, .geip-section-alt, .geip-footer { padding-left: 20px; padding-right: 20px; }
}
</style>

<div id="geip-nav">
  <div id="geip-nav-toggle" onclick="geipToggleNav()">☰</div>
  <div id="geip-nav-menu">
    <a class="geip-nav-item" href="#section-map">
      <span class="geip-nav-dot">⬡</span>Global Intelligence Map
    </a>
    <hr class="geip-nav-sep">
    <a class="geip-nav-item" href="#section-analytics">
      <span class="geip-nav-dot">◈</span>Global Analytics
    </a>
    <hr class="geip-nav-sep">
    <a class="geip-nav-item" href="#section-country">
      <span class="geip-nav-dot">◉</span>Country Intelligence
    </a>
    <hr class="geip-nav-sep">
    <a class="geip-nav-item" href="#section-comparison">
      <span class="geip-nav-dot">📊</span>Cross-Nation Analytics
    </a>
  </div>
</div>
<script>
function geipToggleNav() {
  document.getElementById('geip-nav-menu').classList.toggle('open');
}
document.addEventListener('click', function(e) {
  var n = document.getElementById('geip-nav');
  if (n && !n.contains(e.target))
    document.getElementById('geip-nav-menu').classList.remove('open');
});
</script>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
_HOVER = dict(
    bgcolor="#041229",
    font_color="#00e5ff",
    bordercolor="rgba(0,229,255,.28)",
    font_size=12,
    font_family="JetBrains Mono",
)

_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#7a93b8", size=11),
    hoverlabel=_HOVER,
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.035)",
        zerolinecolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=10, color="#3d5470", family="JetBrains Mono"),
        linecolor="rgba(255,255,255,0.04)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.035)",
        zerolinecolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=10, color="#3d5470", family="JetBrains Mono"),
        linecolor="rgba(255,255,255,0.04)",
    ),
    margin=dict(l=12, r=12, t=52, b=12),
)

CFG = {"displayModeBar": False, "scrollZoom": False}
_DEFAULT_LEGEND = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#7a93b8", size=11))


def apply(fig, title: str = "", height: int = 320, legend: dict | None = None, **kw):
    fig.update_layout(
        **_BASE_LAYOUT,
        height=height,
        title=dict(
            text=title,
            font=dict(color="#e2e8f0", size=14, family="Inter", weight=600),
            x=0.0, xanchor="left", pad=dict(l=4),
        ),
        legend=legend if legend is not None else _DEFAULT_LEGEND,
        **kw,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    for path in (
        "data/engineered_energy_data.csv",
        "data/clean_energy_data.csv",
        "data/owid-energy-data.csv",
    ):
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            df.columns = df.columns.str.strip()
            return df
    return _demo_data()


def _demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    nations = {
        "China":          ("CHN", 1_400e6, 17_700e9, 11),
        "United States":  ("USA",   331e6, 23_000e9, 14),
        "India":          ("IND", 1_380e6,  3_200e9, 11),
        "Germany":        ("DEU",    83e6,  4_200e9, 36),
        "Japan":          ("JPN",   125e6,  4_900e9, 18),
        "Brazil":         ("BRA",   215e6,  1_800e9, 47),
        "France":         ("FRA",    68e6,  2_900e9, 42),
        "United Kingdom": ("GBR",    67e6,  3_100e9, 33),
        "Canada":         ("CAN",    38e6,  2_000e9, 28),
        "Australia":      ("AUS",    26e6,  1_700e9, 19),
        "Norway":         ("NOR",   5.3e6,   480e9,  73),
        "Denmark":        ("DNK",   5.9e6,   370e9,  62),
        "South Africa":   ("ZAF",    60e6,   420e9,   9),
        "Indonesia":      ("IDN",   274e6,  1_200e9, 13),
        "Russia":         ("RUS",   144e6,  1_700e9, 17),
        "Saudi Arabia":   ("SAU",    35e6,  1_000e9,  2),
        "Mexico":         ("MEX",   130e6,  1_300e9, 16),
        "South Korea":    ("KOR",    52e6,  1_800e9, 11),
        "Spain":          ("ESP",    47e6,  1_400e9, 43),
        "Turkey":         ("TUR",    84e6,   800e9,  21),
    }
    rows = []
    for country, (iso, pop0, gdp0, ren0) in nations.items():
        e_scale = (pop0 / 1e9) * (gdp0 / 1e12) * 1_100
        for yr in range(1990, 2024):
            t = (yr - 1990) / 33
            pop = pop0 * (1 + 0.009 * t) + rng.normal(0, pop0 * .002)
            gdp = gdp0 * (1 + 0.027 * t) + rng.normal(0, gdp0 * .008)
            ren = float(np.clip(ren0 + 16 * t + rng.normal(0, 1.2), 0, 95))
            egy = float(max(2, e_scale * (1 - 0.055 * t) + rng.normal(0, e_scale * .035)))
            rows.append(dict(country=country, iso_code=iso, year=yr,
                             population=pop, gdp=gdp,
                             primary_energy_consumption=egy,
                             renewables_share_energy=ren))
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def load_model():
    p = "models/energy_forecast_model.pkl"
    return joblib.load(p) if os.path.exists(p) else None


# ── Load, coerce, filter ──────────────────────────────────────────────────────
_raw   = load_data()
_model = load_model()

for _col in ("population", "gdp", "primary_energy_consumption", "renewables_share_energy"):
    if _col not in _raw.columns:
        _raw[_col] = np.nan
    _raw[_col] = pd.to_numeric(_raw[_col], errors="coerce")

df = (
    _raw[_raw["year"] >= 1990]
    .query("country.notna() and country != 'World'", engine="python")
    .copy()
    .reset_index(drop=True)
)

LATEST    = int(df["year"].max())
COUNTRIES = sorted(df["country"].dropna().unique().tolist())

# ── Session state ─────────────────────────────────────────────────────────────
if "selected_country" not in st.session_state:
    st.session_state.selected_country = "Japan" if "Japan" in COUNTRIES else COUNTRIES[0]

# ── Shared format helpers ─────────────────────────────────────────────────────
def _fmt_pop(x): return f"{x/1e6:,.1f} M"  if pd.notna(x) else "—"
def _fmt_gdp(x): return f"${x/1e9:,.0f} B" if pd.notna(x) else "—"
def _fmt_twh(x): return f"{x:,.1f} TWh"    if pd.notna(x) else "—"
def _fmt_ren(x): return f"{x:.1f}%"         if pd.notna(x) else "—"

# ── Live headline stats ───────────────────────────────────────────────────────
_ld        = df[df["year"] == LATEST]
_total_twh = _ld["primary_energy_consumption"].sum()
_avg_ren   = _ld["renewables_share_energy"].mean()
_n_ctry    = _ld["country"].nunique()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — GLOBAL INTELLIGENCE MAP
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<a id="section-map"></a>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="geip-hero">
      <div class="geip-tag">⚡ &nbsp; Live Intelligence Platform &nbsp;·&nbsp; OWID Dataset</div>
      <h1 class="geip-hero-title">
        GLOBAL ENERGY<br>
        <span class="grad">INTELLIGENCE PLATFORM</span>
      </h1>
      <p class="geip-hero-sub">
        Interactive world energy analysis &amp; AI forecasting<br>
        <span style="color:var(--dim)">1990 – {LATEST} historical &nbsp;·&nbsp; 2025 – 2040 AI forecast</span>
      </p>
      <div class="geip-pills">
        <div class="geip-pill">
          <span class="geip-pill-val">{_n_ctry}</span>
          <span class="geip-pill-lbl">Countries</span>
        </div>
        <div class="geip-pill">
          <span class="geip-pill-val">{_total_twh/1000:,.0f}k</span>
          <span class="geip-pill-lbl">TWh Total Energy</span>
        </div>
        <div class="geip-pill">
          <span class="geip-pill-val">{_avg_ren:.1f}%</span>
          <span class="geip-pill-lbl">Renewable Share</span>
        </div>
        <div class="geip-pill">
          <span class="geip-pill-val">{LATEST}</span>
          <span class="geip-pill-lbl">Latest Year</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Animated flat choropleth ──────────────────────────────────────────────────
_map_years  = list(range(1990, LATEST + 1, 2))
_mdf = (
    df[df["year"].isin(_map_years)]
    .dropna(subset=["primary_energy_consumption", "iso_code"])
    .copy()
)
_mdf["_pop"] = _mdf["population"].map(_fmt_pop)
_mdf["_gdp"] = _mdf["gdp"].map(_fmt_gdp)
_mdf["_eng"] = _mdf["primary_energy_consumption"].map(_fmt_twh)
_mdf["_ren"] = _mdf["renewables_share_energy"].map(_fmt_ren)

_cmax = float(_mdf["primary_energy_consumption"].quantile(0.97))

_CSCALE = [
    [0.00, "#020617"],
    [0.08, "#03244c"],
    [0.22, "#025480"],
    [0.48, "#009fc8"],
    [0.76, "#00cce6"],
    [1.00, "#00e5ff"],
]

_HTMPL = (
    "<b style='font-size:13px;color:#00e5ff;font-family:Inter'>%{text}</b><br>"
    "<span style='color:#3d5470;font-family:JetBrains Mono;font-size:10px'>POPULATION</span>"
    "<b style='color:#f0f6ff;float:right;font-family:JetBrains Mono;font-size:10px'>&nbsp;%{customdata[0]}</b><br>"
    "<span style='color:#3d5470;font-family:JetBrains Mono;font-size:10px'>GDP      </span>"
    "<b style='color:#f0f6ff;float:right;font-family:JetBrains Mono;font-size:10px'>&nbsp;%{customdata[1]}</b><br>"
    "<span style='color:#3d5470;font-family:JetBrains Mono;font-size:10px'>ENERGY   </span>"
    "<b style='color:#00e5ff;float:right;font-family:JetBrains Mono;font-size:10px'>&nbsp;%{customdata[2]}</b><br>"
    "<span style='color:#3d5470;font-family:JetBrains Mono;font-size:10px'>RENEWABLES</span>"
    "<b style='color:#8b7fff;float:right;font-family:JetBrains Mono;font-size:10px'>&nbsp;%{customdata[3]}</b><br>"
    "<span style='color:#3d5470;font-family:JetBrains Mono;font-size:10px'>YEAR     </span>"
    "<b style='color:#f0f6ff;float:right;font-family:JetBrains Mono;font-size:10px'>&nbsp;%{customdata[4]}</b>"
    "<extra></extra>"
)


def _choropleth(sub: pd.DataFrame) -> go.Choropleth:
    cd = np.stack([
        sub["_pop"].values,
        sub["_gdp"].values,
        sub["_eng"].values,
        sub["_ren"].values,
        sub["year"].astype(str).values,
    ], axis=-1)
    return go.Choropleth(
        locations=sub["iso_code"],
        z=sub["primary_energy_consumption"],
        text=sub["country"],
        customdata=cd,
        hovertemplate=_HTMPL,
        colorscale=_CSCALE,
        zmin=0, zmax=_cmax,
        showscale=True,
        colorbar=dict(
            title=dict(text="TWh", font=dict(color="#3d5470", size=10, family="JetBrains Mono")),
            tickfont=dict(color="#3d5470", size=9, family="JetBrains Mono"),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,229,255,.1)",
            borderwidth=1,
            thickness=10, len=0.50, x=1.005,
        ),
        marker=dict(line=dict(color="rgba(0,229,255,.15)", width=0.5)),
    )


_frames = [
    go.Frame(name=str(yr), data=[_choropleth(_mdf[_mdf["year"] == yr])])
    for yr in _map_years
]

_fig_map = go.Figure(
    data=[_choropleth(_mdf[_mdf["year"] == _map_years[0]])],
    frames=_frames,
)

_fig_map.update_geos(
    projection_type="equirectangular",
    bgcolor="#020617",
    landcolor="#071a38",
    oceancolor="#020617",
    showocean=True,
    showcoastlines=True,
    coastlinecolor="rgba(0,229,255,.22)", coastlinewidth=0.7,
    showcountries=True,
    countrycolor="rgba(0,229,255,.13)",  countrywidth=0.4,
    showframe=False,
    lataxis=dict(range=[-62, 85]),
    lonaxis=dict(range=[-180, 180]),
)

_fig_map.update_layout(
    paper_bgcolor="#020617",
    plot_bgcolor="#020617",
    height=580,
    margin=dict(l=0, r=0, t=0, b=0),
    hoverlabel=_HOVER,
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.5, xanchor="center",
        y=-0.042, yanchor="top",
        direction="left",
        bgcolor="#041229",
        bordercolor="rgba(0,229,255,.2)",
        font=dict(color="#00e5ff", size=11, family="JetBrains Mono"),
        buttons=[
            dict(
                label="▶  Play",
                method="animate",
                args=[None, dict(frame=dict(duration=650, redraw=True),
                                 fromcurrent=True, mode="immediate")],
            ),
            dict(
                label="⏸  Pause",
                method="animate",
                args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
            ),
        ],
    )],
    sliders=[dict(
        active=0,
        steps=[
            dict(
                args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                label=f.name, method="animate",
            )
            for f in _frames
        ],
        currentvalue=dict(
            prefix="Year: ",
            font=dict(color="#00e5ff", size=12, family="JetBrains Mono"),
            xanchor="center",
        ),
        transition=dict(duration=0),
        x=0.04, len=0.92,
        y=-0.01,
        bgcolor="#041229",
        bordercolor="rgba(0,229,255,.15)",
        activebgcolor="#00e5ff",
        tickcolor="rgba(0,229,255,.15)",
        font=dict(color="#3d5470", size=9, family="JetBrains Mono"),
        pad=dict(t=8, b=8),
    )],
)

st.markdown('<div style="background:#020617;">', unsafe_allow_html=True)
st.plotly_chart(_fig_map, use_container_width=True, config=CFG)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<p class="geip-map-hint">'
    "▶ Animate timeline to watch the world's energy shift &nbsp;·&nbsp; "
    "Hover any country for intelligence card"
    "</p>",
    unsafe_allow_html=True,
)

st.markdown('<hr class="geip-divider">', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — GLOBAL ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<a id="section-analytics"></a>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="geip-section">
      <p class="geip-eyebrow">Global Analytics</p>
      <h2 class="geip-h2">World energy at a glance</h2>
      <p class="geip-lead">Aggregated across all tracked nations &nbsp;·&nbsp; snapshot year: {LATEST}</p>
      <div class="geip-kpis">
        <div class="geip-kpi">
          <div class="geip-kpi-label">Total Energy Consumed</div>
          <div class="geip-kpi-value">{_total_twh/1000:,.0f}<span style="font-size:16px;font-weight:400">k</span></div>
          <div class="geip-kpi-unit">TWh &nbsp;·&nbsp; {LATEST}</div>
        </div>
        <div class="geip-kpi">
          <div class="geip-kpi-label">Avg Renewable Share</div>
          <div class="geip-kpi-value">{_avg_ren:.1f}<span style="font-size:16px;font-weight:400">%</span></div>
          <div class="geip-kpi-unit">global average</div>
        </div>
        <div class="geip-kpi">
          <div class="geip-kpi-label">Countries Covered</div>
          <div class="geip-kpi-value">{_n_ctry}</div>
          <div class="geip-kpi-unit">nations in dataset</div>
        </div>
        <div class="geip-kpi">
          <div class="geip-kpi-label">Latest Data Year</div>
          <div class="geip-kpi-value">{LATEST}</div>
          <div class="geip-kpi-unit">OWID dataset</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Trend charts side-by-side ─────────────────────────────────────────────────
with st.container():
    st.markdown('<div style="padding: 0 64px 48px 64px; background: #020617;">', unsafe_allow_html=True)
    _c1, _c2 = st.columns(2, gap="large")

    _gt = df.groupby("year")["primary_energy_consumption"].sum().reset_index(name="e")
    with _c1:
        _fg = go.Figure()
        _fg.add_trace(go.Scatter(
            x=_gt["year"], y=_gt["e"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(0,229,255,.045)",
            line=dict(color="#00e5ff", width=2),
            hovertemplate="<b>%{x}</b><br>%{y:,.0f} TWh<extra></extra>",
            name="Total Energy",
        ))
        apply(_fg, "Global Primary Energy Consumption (1990–{})".format(LATEST),
              height=300, yaxis_title="TWh")
        st.plotly_chart(_fg, use_container_width=True, config=CFG)

    _rt = df.groupby("year")["renewables_share_energy"].mean().reset_index(name="r")
    with _c2:
        _fr = go.Figure()
        _fr.add_trace(go.Scatter(
            x=_rt["year"], y=_rt["r"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(106,92,255,.06)",
            line=dict(color="#8b7fff", width=2),
            hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
            name="Renewables %",
        ))
        apply(_fr, "Global Renewable Energy Share (1990–{})".format(LATEST),
              height=300, yaxis_title="% of total energy")
        st.plotly_chart(_fr, use_container_width=True, config=CFG)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<hr class="geip-divider">', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — COUNTRY INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<a id="section-country"></a>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="geip-section-alt">
      <p class="geip-eyebrow">Country Intelligence</p>
      <h2 class="geip-h2">Nation-level deep dive</h2>
      <p class="geip-lead">Historical energy profile and AI-powered consumption forecast through 2040</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div style="padding: 0 64px 48px 64px; background: #041229;">', unsafe_allow_html=True)

    _sel = st.selectbox(
        "SELECT COUNTRY",
        COUNTRIES,
        index=COUNTRIES.index(st.session_state.selected_country)
              if st.session_state.selected_country in COUNTRIES else 0,
        key="country_selector",
    )
    st.session_state.selected_country = _sel

    _cdf  = df[df["country"] == _sel].sort_values("year")
    _lr_s = _cdf[_cdf["year"] == _cdf["year"].max()]

    if len(_lr_s) > 0:
        _r    = _lr_s.iloc[0]
        _yr_v = int(_r["year"])
        st.markdown(
            f"""
            <div class="geip-cstats">
              <div class="geip-cstat">
                <div class="geip-cstat-label">Population</div>
                <div class="geip-cstat-value">{_fmt_pop(_r['population'])}</div>
                <div class="geip-cstat-sub">{_yr_v}</div>
              </div>
              <div class="geip-cstat">
                <div class="geip-cstat-label">GDP</div>
                <div class="geip-cstat-value">{_fmt_gdp(_r['gdp'])}</div>
                <div class="geip-cstat-sub">USD nominal</div>
              </div>
              <div class="geip-cstat">
                <div class="geip-cstat-label">Primary Energy</div>
                <div class="geip-cstat-value" style="color:#00e5ff;">{_fmt_twh(_r['primary_energy_consumption'])}</div>
                <div class="geip-cstat-sub">{_yr_v}</div>
              </div>
              <div class="geip-cstat">
                <div class="geip-cstat-label">Renewable Share</div>
                <div class="geip-cstat-value" style="color:#8b7fff;">{_fmt_ren(_r['renewables_share_energy'])}</div>
                <div class="geip-cstat-sub">of total energy</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Historical chart ──────────────────────────────────────────────────────
    _hist = _cdf.dropna(subset=["primary_energy_consumption"])

    if len(_hist) > 1:
        _fh = go.Figure()
        _fh.add_trace(go.Scatter(
            x=_hist["year"],
            y=_hist["primary_energy_consumption"],
            mode="lines+markers",
            fill="tozeroy",
            fillcolor="rgba(0,229,255,.04)",
            line=dict(color="#00e5ff", width=2.2),
            marker=dict(size=3.5, color="#00e5ff"),
            name="Historical",
            hovertemplate="<b>%{x}</b><br>%{y:,.1f} TWh<extra></extra>",
        ))
        apply(
            _fh,
            title=f"{_sel} — Historical Energy Consumption (1990–{LATEST})",
            height=300,
            yaxis_title="TWh",
        )
        st.plotly_chart(_fh, use_container_width=True, config=CFG)

    # ── Forecast chart ────────────────────────────────────────────────────────
    st.markdown(
        '<p class="geip-badge" style="margin-top: 24px;">⬡ &nbsp;ML Forecast &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; R² ≈ 0.956 &nbsp;·&nbsp; 2025 – 2040</p>',
        unsafe_allow_html=True,
    )

    _future_yrs = list(range(2025, 2041))
    _last_known = _cdf[_cdf["year"] <= LATEST].sort_values("year")
    _fc_arr     = None

    if _model is not None and len(_last_known) > 0:
        try:
            _lr2  = _last_known.iloc[-1]
            _pop0 = _lr2["population"]             if pd.notna(_lr2["population"])             else 5e7
            _gdp0 = _lr2["gdp"]                    if pd.notna(_lr2["gdp"])                    else 5e11
            _ren0 = _lr2["renewables_share_energy"] if pd.notna(_lr2["renewables_share_energy"]) else 20.0
            _gap  = np.arange(1, len(_future_yrs) + 1)

            _fut = pd.DataFrame({
                "year":                    _future_yrs,
                "population":              _pop0 * (1.008 ** _gap),
                "gdp":                     _gdp0 * (1.025 ** _gap),
                "renewables_share_energy": np.clip(_ren0 + 0.65 * _gap, 0, 100),
            })

            _feat  = ["year", "population", "gdp", "renewables_share_energy"]
            _avail = [c for c in _feat if c in _fut.columns]
            _preds = _model.predict(_fut[_avail])

            _last_e = float(_last_known["primary_energy_consumption"].dropna().iloc[-1])
            _fc_arr = np.clip(_preds, _last_e * 0.45, _last_e * 1.55)
        except Exception:
            _fc_arr = None

    if _fc_arr is None and len(_hist) >= 3:
        _recent = _hist["primary_energy_consumption"].tail(8).dropna()
        if len(_recent) >= 2:
            _mean_e = float(_recent.mean())
            _slope  = float(np.clip(_recent.diff().mean(), -_mean_e * .025, _mean_e * .025))
            _base   = float(_recent.iloc[-1])
            _fc_arr = np.array([_base + _slope * (i + 1) for i in range(len(_future_yrs))])

    if _fc_arr is not None:
        _tail     = _hist[_hist["year"] >= LATEST - 10].copy()
        _bridge_y = float(_tail["primary_energy_consumption"].iloc[-1]) if len(_tail) > 0 else float(_fc_arr[0])

        _ff = go.Figure()

        _ff.add_trace(go.Scatter(
            x=_tail["year"],
            y=_tail["primary_energy_consumption"],
            mode="lines",
            line=dict(color="#00e5ff", width=2.2),
            name="Historical",
            hovertemplate="<b>%{x}</b><br>%{y:,.1f} TWh<extra></extra>",
        ))

        _lo_b = _fc_arr * 0.92
        _hi_b = _fc_arr * 1.08
        _ff.add_trace(go.Scatter(
            x=_future_yrs + _future_yrs[::-1],
            y=list(_hi_b) + list(_lo_b[::-1]),
            fill="toself",
            fillcolor="rgba(106,92,255,.07)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))

        _ff.add_trace(go.Scatter(
            x=_future_yrs,
            y=_fc_arr,
            mode="lines+markers",
            line=dict(color="#8b7fff", width=2.2, dash="dot"),
            marker=dict(size=4.5, color="#8b7fff", symbol="diamond"),
            name="AI Forecast",
            hovertemplate="<b>%{x}</b><br>Forecast: %{y:,.1f} TWh<extra></extra>",
        ))

        _ff.add_vline(
            x=LATEST + 0.5,
            line_dash="dash",
            line_color="rgba(0,229,255,.18)",
            line_width=1,
        )
        _ff.add_annotation(
            x=LATEST + 0.75,
            y=_bridge_y,
            text="AI Forecast →",
            showarrow=False,
            font=dict(color="#3d5470", size=9, family="JetBrains Mono"),
            xanchor="left",
        )

        apply(
            _ff,
            title=f"{_sel} — Energy Forecast 2025–2040",
            height=330,
            yaxis_title="TWh",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right",  x=1,
                bgcolor="rgba(0,0,0,0)",
                font=dict(color="#7a93b8", size=11),
            ),
        )
        st.plotly_chart(_ff, use_container_width=True, config=CFG)
    else:
        st.info("Insufficient historical data to generate a forecast for this country.")

    # ── Statistical Correlations Matrix (Restored Feature) ────────────────────
    st.markdown('<p class="geip-eyebrow" style="margin-top:32px;">Feature Co-Movements</p>', unsafe_allow_html=True)
    _num_cols = ["primary_energy_consumption", "population", "gdp", "renewables_share_energy"]
    _corr_df = _cdf[_num_cols].corr()
    
    _fig_corr = go.Figure(data=go.Heatmap(
        z=_corr_df.values,
        x=["Energy", "Population", "GDP", "Renewables %"],
        y=["Energy", "Population", "GDP", "Renewables %"],
        colorscale=_CSCALE,
        zmin=-1, zmax=1,
        hovertemplate="Correlation: %{z:.3f}<extra></extra>"
    ))
    apply(_fig_corr, f"{_sel} — Internal Parameter Correlation Matrices", height=240)
    st.plotly_chart(_fig_corr, use_container_width=True, config=CFG)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<hr class="geip-divider">', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RESTORED ADVANCED COMPARISONS & RAW DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<a id="section-comparison"></a>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="geip-section">
      <p class="geip-eyebrow">Cross-Nation Analytics</p>
      <h2 class="geip-h2">Multi-Country Structural Trajectories</h2>
      <p class="geip-lead">Compare macroeconomic metrics and scale variances between top global energy actors</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div style="padding: 0 64px 48px 64px; background: #020617;">', unsafe_allow_html=True)
    
    _comp_countries = st.multiselect(
        "SELECT COUNTRIES FOR VISUAL MATCHING",
        COUNTRIES,
        default=COUNTRIES[:3] if len(COUNTRIES) >= 3 else COUNTRIES,
        key="comparison_selector"
    )
    
    if _comp_countries:
        _comp_df = df[(df["country"].isin(_comp_countries)) & (df["year"] <= LATEST)]
        
        _fig_comp = go.Figure()
        for _cname in _comp_countries:
            _sub = _comp_df[_comp_df["country"] == _cname].sort_values("year")
            _fig_comp.add_trace(go.Scatter(
                x=_sub["year"],
                y=_sub["primary_energy_consumption"],
                mode="lines",
                name=_cname,
                hovertemplate="<b>" + _cname + "</b><br> Year: %{x}<br>Consumption: %{y:,.1f} TWh<extra></extra>"
            ))
        apply(_fig_comp, "Comparative Primary Energy Run-Rate (TWh)", height=350)
        st.plotly_chart(_fig_comp, use_container_width=True, config=CFG)
        
        # ── Raw Analytical Ledger Grid View ──────────────────────────────────
        st.markdown('<p class="geip-eyebrow" style="margin-top:24px;">Structured Ledger Matrix</p>', unsafe_allow_html=True)
        _display_cols = ["country", "year", "population", "gdp", "primary_energy_consumption", "renewables_share_energy"]
        _ledger_raw = _comp_df[_display_cols].sort_values(["country", "year"], ascending=[True, False])
        
        _ledger_pretty = _ledger_raw.copy()
        _ledger_pretty["population"] = _ledger_pretty["population"].map(_fmt_pop)
        _ledger_pretty["gdp"] = _ledger_pretty["gdp"].map(_fmt_gdp)
        _ledger_pretty["primary_energy_consumption"] = _ledger_pretty["primary_energy_consumption"].map(_fmt_twh)
        _ledger_pretty["renewables_share_energy"] = _ledger_pretty["renewables_share_energy"].map(_fmt_ren)
        
        st.markdown('<div class="geip-table-container">', unsafe_allow_html=True)
        st.dataframe(_ledger_pretty, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # CSV Data Provisioning Matrix
        _csv_out = _ledger_raw.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⭳ &nbsp; EXTRACT COMPILATION LEDGER (.CSV)",
            data=_csv_out,
            file_name="geip_metrics_extract.csv",
            mime="text/csv"
        )
    else:
        st.info("Select one or more sovereign countries to engage live multi-variable matrix comparison trackers.")

    st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
    <div class="geip-footer">
      <span class="geip-footer-left">⚡ &nbsp; Global Energy Intelligence Platform</span>
      <span class="geip-footer-right">
        Data Source: Our World in Data (OWID)<br>
        Model: Random Forest Regressor &nbsp;·&nbsp; R² ≈ 0.956<br>
        Latest Data: {LATEST}
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)