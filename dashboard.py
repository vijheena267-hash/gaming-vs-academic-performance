import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

DATA_PATH = "Gaming_Academic_Performance Dataset.csv"  

def _lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a, b, x, maxit=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(-_lbeta(a, b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def pearsonr(x, y):
    """Pearson correlation coefficient and two-tailed p-value."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    if n <= 2:
        return r, 1.0
    if abs(r) >= 1:
        return r, 0.0
    dfree = n - 2
    t_stat = r * math.sqrt(dfree / (1 - r ** 2))
    p = _betai(dfree / 2, 0.5, dfree / (dfree + t_stat ** 2))
    return r, p


def ttest_ind(a, b, equal_var=False):
    """Two-sample t-test. equal_var=False uses Welch's t-test (unequal variances)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if equal_var:
        dfree = na + nb - 2
        pooled = ((na - 1) * va + (nb - 1) * vb) / dfree
        se = math.sqrt(pooled * (1 / na + 1 / nb))
    else:
        se = math.sqrt(va / na + vb / nb)
        dfree = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t_stat = (ma - mb) / se
    p = _betai(dfree / 2, 0.5, dfree / (dfree + t_stat ** 2))
    return t_stat, p


def f_oneway(*groups):
    """One-way ANOVA F-test across 2+ groups."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    k = len(groups)
    n = sum(len(g) for g in groups)
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_between = k - 1
    df_within = n - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within
    p = _betai(df_within / 2, df_between / 2, df_within / (df_within + df_between * f_stat))
    return f_stat, p


def ols(y, x_df):
    """
    Ordinary least squares regression with an intercept, via the normal
    equations. y: 1D array-like target. x_df: DataFrame of predictor columns.
    Returns a dict with params, bse (standard errors), pvalues, rsquared,
    and rsquared_adj - the same fields used from statsmodels previously.
    """
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), x_df.to_numpy(dtype=float)])
    names = ["Intercept"] + list(x_df.columns)
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    dfree = n - k
    r2_adj = 1 - (1 - r2) * (n - 1) / dfree
    sigma2 = ss_res / dfree
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    t_stats = beta / se
    pvals = [_betai(dfree / 2, 0.5, dfree / (dfree + t ** 2)) for t in t_stats]
    return {
        "params": pd.Series(beta, index=names),
        "bse": pd.Series(se, index=names),
        "pvalues": pd.Series(pvals, index=names),
        "rsquared": r2,
        "rsquared_adj": r2_adj,
    }

# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Gaming vs Academic Performance", layout="wide", initial_sidebar_state="expanded")

NAVY = "#0A1628"
NAVY_2 = "#0F2038"
NAVY_3 = "#16294A"
CYAN = "#2DD4E8"
AMBER = "#F5A623"
VIOLET = "#8B7CE8"
TEXT = "#E7EEF7"
MUTED = "#8CA0BC"
GRID = "rgba(255,255,255,0.08)"
PALETTE = [CYAN, AMBER, VIOLET, "#3BD671", "#E34948", "#5DCAA5", "#F27FB0", "#6FA8DC"]

STRESS_ORDER = ["Low", "Medium", "High"]
STRESS_MAP = {"Low": 1, "Medium": 2, "High": 3}

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {NAVY}; color: {TEXT}; }}
    section[data-testid="stSidebar"] {{ background-color: {NAVY_2}; border-right: 1px solid {GRID}; }}
    div[data-testid="stMetric"] {{
        background-color: {NAVY_2};
        border: 1px solid {GRID};
        border-radius: 12px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    div[data-testid="stMetricValue"] {{ color: {CYAN}; }}
    h1, h2, h3, p, label, span {{ color: {TEXT}; }}
    .chart-card {{
        background-color: {NAVY_2};
        border: 1px solid {GRID};
        border-radius: 12px;
        padding: 4px 10px 10px 10px;
        margin-bottom: 8px;
    }}
    .chart-title {{ font-size: 15px; font-weight: 600; margin: 6px 0 0 4px; }}
    .chart-sub {{ font-size: 12px; color: {MUTED}; margin: 0 0 4px 4px; }}
    .dash-header {{
        background: linear-gradient(135deg, {NAVY_2} 0%, {NAVY_3} 100%);
        border: 1px solid {GRID};
        border-left: 4px solid {CYAN};
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }}
    .dash-header-top {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        flex-wrap: wrap;
        gap: 12px;
    }}
    .dash-title {{ font-size: 24px; font-weight: 700; color: {TEXT}; margin: 0 0 4px 0; letter-spacing: 0.2px; }}
    .dash-subtitle {{ font-size: 13px; color: {MUTED}; margin: 0; }}
    .dash-badges {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }}
    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.3px;
        padding: 5px 11px;
        border-radius: 999px;
        border: 1px solid {GRID};
        color: {MUTED};
        background: rgba(255,255,255,0.03);
    }}
    .badge-dot {{ width: 7px; height: 7px; border-radius: 50%; }}
    .badge-live {{ color: {CYAN}; border-color: rgba(45,212,232,0.35); background: {CYAN}1A; }}
    .badge-sample {{ color: {AMBER}; border-color: rgba(245,166,35,0.35); background: {AMBER}1A; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data generation (synthetic fallback, mirrors the real schema)
# ---------------------------------------------------------------------------
@st.cache_data
def generate_data(n: int = 500, seed: int = 1337) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female", "Other"], size=n, p=[0.48, 0.48, 0.04])
    gaming_hours = np.clip(2.6 + rng.standard_normal(n) * 1.7, 0, 9.5)
    addiction_score = np.clip(gaming_hours * 7.2 + rng.standard_normal(n) * 11 + 8, 0, 100)
    study_hours = np.clip(4.6 - gaming_hours * 0.26 + rng.standard_normal(n) * 1.2, 0.3, 8.5)
    sleep_hours = np.clip(7.1 - gaming_hours * 0.14 + rng.standard_normal(n) * 0.9, 3.5, 10)
    stress_numeric = np.clip(
        np.round(1 + addiction_score * 0.02 + (9 - sleep_hours) * 0.15 + rng.standard_normal(n) * 0.6), 1, 3
    ).astype(int)
    stress_level = np.array(STRESS_ORDER)[stress_numeric - 1]
    attendance = np.clip(90 - gaming_hours * 2.6 + rng.standard_normal(n) * 7, 45, 100)
    reaction_time_ms = np.clip(325 - gaming_hours * 7.5 + rng.standard_normal(n) * 28, 150, 500)
    social_activity = np.clip(2.6 - addiction_score * 0.015 + rng.standard_normal(n) * 0.8, 0, 5)
    device_usage = np.clip(gaming_hours * 0.6 + 3 + rng.standard_normal(n) * 1.8, 1, 14)
    grades = np.clip(
        90 - gaming_hours * 2.7 + study_hours * 3.1 + attendance * 0.14 - stress_numeric * 3.5
        + rng.standard_normal(n) * 5.5,
        30, 100,
    )
    age = np.clip(np.round(18 + rng.standard_normal(n) * 1.6), 16, 24).astype(int)
    gaming_genre = rng.choice(["Casual", "FPS", "RPG"], size=n)

    df = pd.DataFrame({
        "student_id": np.arange(1, n + 1),
        "age": age,
        "gender": gender,
        "gaming_hours": gaming_hours.round(2),
        "study_hours": study_hours.round(2),
        "sleep_hours": sleep_hours.round(2),
        "attendance": attendance.round(2),
        "gaming_genre": gaming_genre,
        "social_activity": social_activity.round(2),
        "device_usage": device_usage.round(2),
        "reaction_time_ms": reaction_time_ms.round(2),
        "addiction_score": addiction_score.round(2),
        "stress_level": stress_level,
        "grades": grades.round(2),
    })
    return df


REQUIRED_COLS = [
    "student_id", "age", "gender", "gaming_hours", "study_hours", "sleep_hours",
    "attendance", "gaming_genre", "social_activity", "device_usage",
    "reaction_time_ms", "addiction_score", "stress_level", "grades",
]


def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns needed for numeric stats (stress_numeric)."""
    df = df.copy()
    if df["stress_level"].dtype == object or str(df["stress_level"].dtype).startswith("str"):
        df["stress_numeric"] = df["stress_level"].map(STRESS_MAP)
        # In case labels don't match the expected set exactly, fall back to a
        # rank-based ordinal so the app never crashes on unexpected labels.
        if df["stress_numeric"].isna().any():
            cats = sorted(df["stress_level"].dropna().unique().tolist())
            fallback_map = {c: i + 1 for i, c in enumerate(cats)}
            df["stress_numeric"] = df["stress_level"].map(fallback_map)
    else:
        # Already numeric in the uploaded file
        df["stress_numeric"] = df["stress_level"]
    return df


def load_data() -> tuple[pd.DataFrame, bool]:
    """Returns (dataframe, is_real_data)."""
    uploaded = st.sidebar.file_uploader("Upload your CSV (optional)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.sidebar.error(f"Missing columns: {', '.join(missing)}. Showing sample data instead.")
        else:
            return prep_data(df), True
    if DATA_PATH:
        try:
            df = pd.read_csv(DATA_PATH)
            return prep_data(df), True
        except FileNotFoundError:
            pass
    return prep_data(generate_data()), False


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------
def base_layout(height=320, xlabel=None, ylabel=None):
    return dict(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=12),
        xaxis=dict(title=xlabel, gridcolor=GRID, zerolinecolor=GRID, color=MUTED),
        yaxis=dict(title=ylabel, gridcolor=GRID, zerolinecolor=GRID, color=MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=MUTED)),
        showlegend=False,
    )


def chart_card(title, sub, fig, key):
    st.markdown(f'<div class="chart-title">{title}</div><div class="chart-sub">{sub}</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, key=key)


def histogram_fig(series, nbins, color, xlabel):
    fig = go.Figure(go.Histogram(x=series, nbinsx=nbins, marker_color=color, marker_line_width=0))
    fig.update_layout(**base_layout(xlabel=xlabel, ylabel="Students"))
    return fig


def donut_fig(series, colors):
    counts = series.value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.62,
        marker=dict(colors=colors, line=dict(color=NAVY_2, width=2)),
        textfont=dict(color=TEXT),
    ))
    layout = base_layout(height=300)
    layout["showlegend"] = True
    fig.update_layout(**layout)
    return fig


def scatter_fig(df, xcol, ycol, xlabel, ylabel, color):
    fig = go.Figure(go.Scattergl(
        x=df[xcol], y=df[ycol], mode="markers",
        marker=dict(color=color, size=6, opacity=0.65),
    ))
    fig.update_layout(**base_layout(xlabel=xlabel, ylabel=ylabel))
    return fig


def bar_fig(labels, values, color, horizontal=False, xlabel=None, ylabel=None):
    if horizontal:
        fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=color))
    else:
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color=color))
    fig.update_layout(**base_layout(xlabel=xlabel, ylabel=ylabel))
    return fig


def line_fig(x, y, color, xlabel=None, ylabel=None):
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=6, color=color),
    ))
    fig.update_layout(**base_layout(xlabel=xlabel, ylabel=ylabel))
    return fig


def area_fig(x, y, color, fillcolor, xlabel=None, ylabel=None):
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=fillcolor,
    ))
    fig.update_layout(**base_layout(xlabel=xlabel, ylabel=ylabel))
    return fig


def violin_fig(df, group_col, value_col, colors, order=None, xlabel=None, ylabel=None):
    fig = go.Figure()
    categories = order if order else sorted(df[group_col].dropna().unique().tolist())
    for i, cat in enumerate(categories):
        c = colors[i % len(colors)]
        fig.add_trace(go.Violin(
            y=df.loc[df[group_col] == cat, value_col], name=str(cat),
            line_color=c, fillcolor=c, opacity=0.5,
            box_visible=True, meanline_visible=True, points=False,
        ))
    fig.update_layout(**base_layout(xlabel=xlabel, ylabel=ylabel))
    return fig


def box_fig(df, group_col, value_col, colors, order=None, xlabel=None, ylabel=None):
    fig = go.Figure()
    categories = order if order else sorted(df[group_col].dropna().unique().tolist())
    for i, cat in enumerate(categories):
        c = colors[i % len(colors)]
        fig.add_trace(go.Box(
            y=df.loc[df[group_col] == cat, value_col], name=str(cat),
            marker_color=c, line=dict(color=c),
        ))
    fig.update_layout(**base_layout(xlabel=xlabel, ylabel=ylabel))
    return fig


# ---------------------------------------------------------------------------
# Sidebar: navigation + filters
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div style="display:flex;align-items:center;gap:8px;padding-bottom:16px;'
    f'border-bottom:1px solid {GRID};margin-bottom:16px;">'
    f'<div style="width:10px;height:10px;border-radius:50%;background:{CYAN};"></div>'
    f'<div><div style="font-weight:600;font-size:15px;">GameXAcademics</div>'
    f'<div style="font-size:11px;color:{MUTED};">Student insights dashboard</div></div></div>',
    unsafe_allow_html=True,
)

df_raw, is_real = load_data()
if not is_real:
    st.sidebar.caption("Showing 500 synthetic sample students. Upload a CSV above to use your real data.")

# Derived dynamically so the app works with any set of genres/genders in the data
GENRES = sorted(df_raw["gaming_genre"].dropna().unique().tolist())
GENDERS = sorted(df_raw["gender"].dropna().unique().tolist())
STRESS_LEVELS = [s for s in STRESS_ORDER if s in df_raw["stress_level"].unique()] or \
    sorted(df_raw["stress_level"].dropna().unique().tolist())

section = st.sidebar.radio(
    "Menu",
    ["Overview", "Gaming behavior", "Academic performance", "Health & wellbeing",
     "Correlations & insights", "Data explorer"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")
f_gender = st.sidebar.selectbox("Gender", ["All"] + GENDERS)
f_genre = st.sidebar.selectbox("Gaming genre", ["All"] + GENRES)
f_stress = st.sidebar.selectbox("Stress level", ["All"] + STRESS_LEVELS)

df = df_raw.copy()
if f_gender != "All":
    df = df[df["gender"] == f_gender]
if f_genre != "All":
    df = df[df["gaming_genre"] == f_genre]
if f_stress != "All":
    df = df[df["stress_level"] == f_stress]

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
active_filters = [f for f in [f_gender, f_genre, f_stress] if f != "All"]
filter_text = ", ".join(active_filters) if active_filters else "None"
source_badge_class = "badge-live" if is_real else "badge-sample"
source_label = "Real data (uploaded CSV)" if is_real else "Sample data (synthetic)"
generated_at = datetime.now().strftime("%d %b %Y, %H:%M")

st.markdown(
    f"""
    <div class="dash-header">
        <div class="dash-header-top">
            <div>
                <p class="dash-title">Gaming vs Academic Performance</p>
                <p class="dash-subtitle">Exploring how gaming habits relate to study time, wellbeing and grades</p>
            </div>
        </div>
        <div class="dash-badges">
            <span class="badge {source_badge_class}"><span class="badge-dot" style="background:currentColor;"></span>{source_label}</span>
            <span class="badge">{len(df_raw)} total students</span>
            <span class="badge">{df_raw.shape[1]} columns</span>
            <span class="badge">Active filters: {filter_text}</span>
            <span class="badge">Generated {generated_at}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Students", len(df))
k2.metric("Avg grade", f"{df['grades'].mean():.1f}" if len(df) else "-")
k3.metric("Avg gaming hrs/day", f"{df['gaming_hours'].mean():.1f}h" if len(df) else "-")
k4.metric("Avg study hrs/day", f"{df['study_hours'].mean():.1f}h" if len(df) else "-")
k5.metric("Avg addiction score", f"{df['addiction_score'].mean():.0f}" if len(df) else "-")
k6.metric("Avg stress level", f"{df['stress_numeric'].mean():.1f}/3" if len(df) else "-")

st.markdown("")

if len(df) == 0:
    st.warning("No students match the current filters. Try widening your selection.")
    st.stop()

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
if section == "Overview":
    with st.expander("Key findings", expanded=True):
        corr_game_grade = df["gaming_hours"].corr(df["grades"])
        corr_study_grade = df["study_hours"].corr(df["grades"])
        corr_attendance_grade = df["attendance"].corr(df["grades"])
        corr_sleep_stress = df["sleep_hours"].corr(df["stress_numeric"])
        med_gaming = df["gaming_hours"].median()
        heavy = df[df["gaming_hours"] > med_gaming]
        light = df[df["gaming_hours"] <= med_gaming]

        bullets = [
            f"Gaming hours vs grades: r = {corr_game_grade:.2f} "
            f"({'negative' if corr_game_grade < 0 else 'positive'} relationship).",
            f"Study hours vs grades: r = {corr_study_grade:.2f} "
            f"({'positive' if corr_study_grade > 0 else 'negative'} relationship).",
            f"Attendance vs grades: r = {corr_attendance_grade:.2f}.",
            f"Sleep hours vs stress level: r = {corr_sleep_stress:.2f}.",
            f"Students gaming above the median ({med_gaming:.1f}h/day) average "
            f"{heavy['grades'].mean():.1f} in grades, vs {light['grades'].mean():.1f} for those below it.",
        ]
        for b in bullets:
            st.markdown(f"- {b}")
        st.caption("Correlation values (r) range from -1 to 1. See the Statistical analysis "
                    "section for significance tests behind these numbers.")

    c1, c2 = st.columns(2)
    with c1:
        fig = histogram_fig(df["grades"], 10, CYAN, "Grades")
        chart_card("Grade distribution", "Number of students by grade band", fig, "grade_hist")
    with c2:
        fig = donut_fig(df["gender"], PALETTE)
        chart_card("Gender split", "Share of students by gender", fig, "gender_donut")

    counts = df["gaming_genre"].value_counts().reindex(GENRES, fill_value=0)
    fig = bar_fig(counts.index.tolist(), counts.values.tolist(), AMBER, horizontal=True, xlabel="Students")
    chart_card("Gaming genre popularity", "Number of students preferring each genre", fig, "genre_bar")

elif section == "Gaming behavior":
    c1, c2 = st.columns(2)
    with c1:
        fig = histogram_fig(df["gaming_hours"], 10, AMBER, "Gaming hours/day")
        chart_card("Gaming hours distribution", "Daily gaming hours across students", fig, "gaming_hist")
    with c2:
        fig = histogram_fig(df["device_usage"], 10, VIOLET, "Device usage (hrs/day)")
        chart_card("Device usage distribution", "Daily device usage hours across students", fig, "device_hist")

    c3, c4 = st.columns(2)
    with c3:
        fig = scatter_fig(df, "gaming_hours", "grades", "Gaming hours/day", "Grades", "rgba(45,212,232,0.65)")
        chart_card("Gaming hours vs grades", "Each point is one student", fig, "gaming_grade_scatter")
    with c4:
        fig = go.Figure(go.Scatter(
            x=df["gaming_hours"], y=df["addiction_score"], mode="markers",
            marker=dict(
                size=df["stress_numeric"] * 5 + 4, color="rgba(245,166,35,0.45)",
                line=dict(color=AMBER, width=1),
            ),
        ))
        fig.update_layout(**base_layout(xlabel="Gaming hours/day", ylabel="Addiction score"))
        chart_card("Gaming hours vs addiction score", "Bubble size reflects stress level", fig, "addiction_bubble")

    c5, c6 = st.columns(2)
    with c5:
        fig = violin_fig(df, "gaming_genre", "addiction_score", PALETTE, order=GENRES, ylabel="Addiction score")
        chart_card("Addiction score by genre", "Full distribution shape per gaming genre", fig, "addiction_violin")
    with c6:
        sorted_hours = np.sort(df["gaming_hours"].values)
        cum_pct = np.arange(1, len(sorted_hours) + 1) / len(sorted_hours) * 100
        fig = area_fig(sorted_hours, cum_pct, AMBER, "rgba(245,166,35,0.2)",
                        xlabel="Gaming hours/day", ylabel="Cumulative % of students")
        chart_card("Cumulative gaming hours", "Share of students at or below each hour level", fig, "gaming_cum_area")

elif section == "Academic performance":
    c1, c2 = st.columns(2)
    with c1:
        fig = scatter_fig(df, "study_hours", "grades", "Study hours/day", "Grades", "rgba(45,212,232,0.65)")
        chart_card("Study hours vs grades", "Each point is one student", fig, "study_grade_scatter")
    with c2:
        fig = scatter_fig(df, "attendance", "grades", "Attendance %", "Grades", "rgba(245,166,35,0.65)")
        chart_card("Attendance vs grades", "Each point is one student", fig, "attendance_grade_scatter")

    c3, c4 = st.columns(2)
    with c3:
        fig = histogram_fig(df["study_hours"], 9, CYAN, "Study hours/day")
        chart_card("Study hours distribution", "Daily study hours across students", fig, "study_hist")
    with c4:
        means = df.groupby("gaming_genre")["grades"].mean().reindex(GENRES)
        fig = bar_fig(means.index.tolist(), means.values.tolist(), CYAN, ylabel="Avg grade")
        chart_card("Average grade by gaming genre", "Mean grade per preferred genre", fig, "genre_grade_bar")

    age_trend = df.groupby("age")["grades"].mean().sort_index()
    fig = line_fig(age_trend.index.tolist(), age_trend.values.tolist(), CYAN, xlabel="Age", ylabel="Avg grade")
    chart_card("Average grade by age", "Mean grade trend across student ages", fig, "grade_age_line")

elif section == "Health & wellbeing":
    c1, c2 = st.columns(2)
    with c1:
        fig = box_fig(df, "stress_level", "sleep_hours", PALETTE, order=STRESS_LEVELS, ylabel="Sleep hours")
        chart_card("Sleep hours by stress level", "Distribution of sleep hours per stress category", fig, "sleep_stress_box")
    with c2:
        fig = scatter_fig(df, "gaming_hours", "reaction_time_ms", "Gaming hours/day", "Reaction time (ms)", "rgba(245,166,35,0.65)")
        chart_card("Reaction time vs gaming hours", "Each point is one student", fig, "reaction_scatter")

    c3, c4 = st.columns(2)
    with c3:
        counts = df["stress_level"].value_counts().reindex(STRESS_LEVELS, fill_value=0)
        fig = bar_fig(counts.index.tolist(), counts.values.tolist(), VIOLET, ylabel="Students")
        chart_card("Stress level distribution", "Number of students per stress category", fig, "stress_bar")
    with c4:
        fig = box_fig(df, "stress_level", "grades", PALETTE, order=STRESS_LEVELS, ylabel="Grades")
        chart_card("Grades by stress level", "Spread and outliers per stress category", fig, "grades_stress_box")

elif section == "Correlations & insights":
    numeric_cols = ["gaming_hours", "study_hours", "sleep_hours", "attendance",
                     "addiction_score", "stress_numeric", "reaction_time_ms",
                     "device_usage", "social_activity", "grades"]
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(**base_layout(height=460))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(side="bottom")
    chart_card("Correlation matrix", "Pearson correlation between numeric variables (stress uses Low=1, Medium=2, High=3)", fig, "corr_heatmap")

    q = max(1, int(len(df) * 0.25))
    sorted_df = df.sort_values("grades", ascending=False)
    high = sorted_df.head(q)
    low = sorted_df.tail(q)

    dims = [
        ("Study hours", "study_hours", 9, False),
        ("Sleep hours", "sleep_hours", 10, False),
        ("Attendance", "attendance", 100, False),
        ("Social activity", "social_activity", 5, False),
        ("Low addiction", "addiction_score", 100, True),
        ("Low stress", "stress_numeric", 3, True),
    ]

    def profile(rowset):
        vals = []
        for label, key, mx, invert in dims:
            m = rowset[key].mean() if len(rowset) else 0
            norm = np.clip((m / mx) * 100, 0, 100)
            vals.append(100 - norm if invert else norm)
        return vals

    labels = [d[0] for d in dims]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=profile(high) + [profile(high)[0]], theta=labels + [labels[0]],
                                   fill="toself", name="High performers (top 25%)",
                                   line=dict(color=CYAN), fillcolor="rgba(45,212,232,0.18)"))
    fig.add_trace(go.Scatterpolar(r=profile(low) + [profile(low)[0]], theta=labels + [labels[0]],
                                   fill="toself", name="Low performers (bottom 25%)",
                                   line=dict(color=AMBER), fillcolor="rgba(245,166,35,0.15)"))
    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor=GRID),
            angularaxis=dict(gridcolor=GRID, color=MUTED),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0, font=dict(color=MUTED)),
        margin=dict(l=40, r=40, t=30, b=10),
    )
    chart_card("High vs low performer profile", "Top 25% vs bottom 25% by grade, normalized 0-100", fig, "radar")

elif section == "Data explorer":
    st.markdown("#### Data quality summary")
    dq1, dq2, dq3, dq4 = st.columns(4)
    dq1.metric("Rows (filtered)", len(df))
    dq2.metric("Missing values", int(df[REQUIRED_COLS].isna().sum().sum()))
    dq3.metric("Duplicate rows", int(df[REQUIRED_COLS].duplicated().sum()))
    dq4.metric("Columns", len(REQUIRED_COLS))

    st.markdown("#### Descriptive statistics")
    st.caption("Summary statistics for all numeric columns in the current filtered selection.")
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["student_id"], errors="ignore")
    st.dataframe(numeric_df.describe().T.round(2), use_container_width=True)

    st.markdown("#### Raw / filtered data")
    st.caption(f"Showing {len(df)} rows matching the current sidebar filters.")
    st.dataframe(df[REQUIRED_COLS], use_container_width=True, height=380)

    st.download_button(
        "Download filtered data as CSV",
        data=df[REQUIRED_COLS].to_csv(index=False).encode("utf-8"),
        file_name="filtered_gaming_academic_data.csv",
        mime="text/csv",
    )
