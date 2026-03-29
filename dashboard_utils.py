from __future__ import annotations

import datetime as dt
import html
import math
import pickle
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

CLEANED_CSV_CANDIDATES = [
    DATA_DIR / "Cleaned_dataset" / "WA_Fn-UseC_-HR-Employee-Attrition_capped.csv",
    DATA_DIR / "Cleaned_data" / "WA_Fn-UseC_-HR-Employee-Attrition_capped.csv",
    REPO_ROOT / "WA_Fn-UseC_-HR-Employee-Attrition_capped.csv",
]

MODEL_DIR = REPO_ROOT / "jupyter_notebooks" / "model"
MODEL_CANDIDATES = {
    "Best model": MODEL_DIR / "attrition_minority_yes_best_model.pkl",
    "Business-cost model": MODEL_DIR / "attrition_minority_yes_business_cost_second_best.pkl",
}


@dataclass(frozen=True)
class InteractionRow:
    interaction: str
    kind: str  # "pos" or "neg"
    raw_feature_1: str
    raw_feature_2: str
    spearman_r: float


def _st_cache_data(func):
    if st is None:
        return func

    # Avoid Streamlit cache wrappers when running as a normal python script
    # (e.g., CLI smoke tests) because it produces noisy warnings and provides
    # no benefit without a Streamlit runtime.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

        if get_script_run_ctx() is None:
            return func
    except Exception:
        return func

    return st.cache_data(show_spinner=False)(func)


def _st_cache_resource(func):
    if st is None:
        return func

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

        if get_script_run_ctx() is None:
            return func
    except Exception:
        return func

    return st.cache_resource(show_spinner=False)(func)


@_st_cache_data
def load_cleaned_dataset() -> pd.DataFrame:
    path = next((p for p in CLEANED_CSV_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Could not find cleaned dataset. Looked for: "
            + ", ".join(str(p) for p in CLEANED_CSV_CANDIDATES)
        )

    df = pd.read_csv(path)
    return df


@_st_cache_data
def numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


@_st_cache_data
def correlation_pairs(df: pd.DataFrame) -> pd.DataFrame:
    num = numeric_df(df)
    corr = num.corr(method="spearman")

    corr_long = (
        corr.where(~np.eye(corr.shape[0], dtype=bool)).stack().reset_index(name="spearman_r")
    )
    corr_long.columns = ["feature_1", "feature_2", "spearman_r"]

    corr_long["pair_key"] = corr_long.apply(
        lambda r: tuple(sorted([r["feature_1"], r["feature_2"]])), axis=1
    )
    corr_pairs = corr_long.drop_duplicates(subset=["pair_key"]).drop(columns=["pair_key"])

    # Match the notebook: descending by spearman_r
    corr_pairs = corr_pairs.sort_values("spearman_r", ascending=False).reset_index(drop=True)
    return corr_pairs


@_st_cache_data
def interaction_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Return the mapping from inter_pos/inter_neg to the raw feature pairs.

    This follows the same ordering logic used in Feature_engineering.ipynb:
    - correlations are sorted by spearman_r DESC
    - positive pairs get inter_pos_001.. in that order
    - negative pairs get inter_neg_001.. in that order (still descending, i.e. closest-to-zero first)
    """

    corr_pairs = correlation_pairs(df)

    positive_pairs = corr_pairs[corr_pairs["spearman_r"] > 0].reset_index(drop=True)
    negative_pairs = corr_pairs[corr_pairs["spearman_r"] < 0].reset_index(drop=True)

    rows: list[dict] = []

    for i, row in enumerate(positive_pairs.itertuples(index=False), start=1):
        rows.append(
            {
                "interaction": f"inter_pos_{i:03d}",
                "kind": "pos",
                "raw_feature_1": row.feature_1,
                "raw_feature_2": row.feature_2,
                "spearman_r": float(row.spearman_r),
                "formula": f"({row.feature_1} / {row.feature_2}) * {float(row.spearman_r):.6f}",
            }
        )

    for i, row in enumerate(negative_pairs.itertuples(index=False), start=1):
        rows.append(
            {
                "interaction": f"inter_neg_{i:03d}",
                "kind": "neg",
                "raw_feature_1": row.feature_1,
                "raw_feature_2": row.feature_2,
                "spearman_r": float(row.spearman_r),
                "formula": f"({row.feature_1} * {row.feature_2}) * {float(row.spearman_r):.6f}",
            }
        )

    return pd.DataFrame(rows)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def engineer_interactions(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    target_col: str = "Attrition",
) -> pd.DataFrame:
    """Create `inter_pos_*` and `inter_neg_*` columns and drop the raw numeric inputs.

    This mirrors the notebook logic closely.

    Requirements:
    - `df` must include all raw numeric columns referenced in `mapping`.
    - categorical columns are preserved.

    Returns a new engineered dataframe.
    """

    engineered = df.copy()
    used_original_features: set[str] = set()
    new_cols: dict[str, pd.Series] = {}

    # Build pos and neg in the same order as mapping
    for row in mapping.itertuples(index=False):
        inter = row.interaction
        f1 = row.raw_feature_1
        f2 = row.raw_feature_2
        r = float(row.spearman_r)

        if f1 not in engineered.columns or f2 not in engineered.columns:
            raise KeyError(
                f"Missing required raw numeric columns for interaction {inter}: {f1}, {f2}"
            )

        if row.kind == "pos":
            new_cols[inter] = _safe_divide(engineered[f1], engineered[f2]) * r
        else:
            new_cols[inter] = (engineered[f1] * engineered[f2]) * r

        used_original_features.update([f1, f2])

    if new_cols:
        engineered = pd.concat([engineered, pd.DataFrame(new_cols, index=engineered.index)], axis=1)

    # Drop original numeric features used in engineering, keep target if present
    drop_original = [c for c in sorted(used_original_features) if c in engineered.columns]
    if target_col in drop_original:
        drop_original.remove(target_col)

    engineered = engineered.drop(columns=drop_original)
    return engineered


@_st_cache_resource
def load_pipeline(model_label: str):
    path = MODEL_CANDIDATES.get(model_label)
    if path is None:
        raise KeyError(f"Unknown model label: {model_label}")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    # Some training notebooks persist a dict wrapper (e.g., metadata + model).
    # Normalize to an object that exposes `predict_proba`.
    if isinstance(obj, dict):
        # Common keys first
        for key in ("pipeline", "model", "estimator", "clf", "classifier"):
            candidate = obj.get(key)
            if hasattr(candidate, "predict_proba"):
                return candidate

        # Otherwise, return the first value that looks like an estimator.
        for candidate in obj.values():
            if hasattr(candidate, "predict_proba"):
                return candidate

        raise TypeError(
            "Loaded model pickle is a dict but no value exposes 'predict_proba'. "
            f"Available keys: {sorted(obj.keys())}"
        )

    return obj


def predict_proba_attrition(
    model_label: str,
    df_raw: pd.DataFrame,
    dataset_for_mapping: pd.DataFrame,
    target_col: str = "Attrition",
) -> pd.DataFrame:
    """Return a dataframe with predicted attrition probability for each row.

    The saved sklearn pipeline expects engineered interaction features.
    We reconstruct those interaction features using mapping derived from the training dataset.
    """

    pipe = load_pipeline(model_label)
    mapping = interaction_mapping(dataset_for_mapping)

    # If user already supplied engineered columns, do not re-engineer.
    already_engineered = any(c.startswith("inter_pos_") or c.startswith("inter_neg_") for c in df_raw.columns)

    if already_engineered:
        X = df_raw.copy()
    else:
        X = engineer_interactions(df_raw, mapping=mapping, target_col=target_col)

    if target_col in X.columns:
        X = X.drop(columns=[target_col])

    if not hasattr(pipe, "predict_proba"):
        raise TypeError(
            "Loaded model does not support predict_proba. "
            f"Loaded type: {type(pipe)!r}"
        )

    proba = pipe.predict_proba(X)[:, 1]
    out = df_raw.copy()
    out["pred_attrition_proba"] = proba
    out["pred_attrition_label"] = (proba >= 0.5).astype(int)
    return out


def interaction_importance_table(
    model_label: str,
    dataset_for_mapping: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    pipe = load_pipeline(model_label)

    # Extract LR + feature names after preprocessing
    if not hasattr(pipe, "named_steps"):
        raise TypeError("Expected a sklearn Pipeline with named_steps")

    model = pipe.named_steps.get("model")
    pre = pipe.named_steps.get("preprocessor") or pipe.named_steps.get("preprocess")

    if model is None or pre is None:
        raise KeyError("Pipeline must contain 'preprocessor' (or 'preprocess') and 'model' steps")

    feature_names = list(pre.get_feature_names_out())
    coefs = model.coef_[0]

    mapping = interaction_mapping(dataset_for_mapping).set_index("interaction")

    rows = []
    for fname, c in zip(feature_names, coefs):
        if "inter_pos_" not in fname and "inter_neg_" not in fname:
            continue
        inter = fname.split("__", 1)[-1]
        if inter not in mapping.index:
            continue
        m = mapping.loc[inter]
        rows.append(
            {
                "interaction": inter,
                "kind": m["kind"],
                "raw_feature_1": m["raw_feature_1"],
                "raw_feature_2": m["raw_feature_2"],
                "spearman_r": float(m["spearman_r"]),
                "coef": float(c),
                "abs_coef": float(abs(c)),
                "odds_ratio": float(math.exp(c)),
                "direction": "increases_attrition" if c > 0 else "decreases_attrition",
            }
        )

    df = pd.DataFrame(rows).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df.head(top_n)


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    return missing


AUDIENCE_OPTIONS = ["Non-technical", "Semi-technical", "Technical"]


THEME_OPTIONS = ["Dark", "Light"]


PLOTLY_THEME_NAME = "dark_green"
PLOTLY_THEME_LIGHT_NAME = "light_green"


def theme_selector(*, default: str = "Dark") -> str:
    """Persistent theme selector for all pages.

    Notes:
    - Streamlit's built-in theme (config.toml) is static at startup.
    - This selector applies a lightweight CSS override plus Plotly templates.
    """

    if st is None:
        return default

    if default not in THEME_OPTIONS:
        default = "Dark"

    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = default

    idx = THEME_OPTIONS.index(st.session_state["theme_mode"])
    choice = st.sidebar.radio("Theme", options=THEME_OPTIONS, index=idx, horizontal=True)
    st.session_state["theme_mode"] = choice
    return choice


def apply_app_theme(theme_mode: str) -> None:
    """Apply CSS overrides for Light/Dark modes.

    This doesn't fully replace Streamlit's native theming, but it gives a clean
    and consistent look across the app without requiring a restart.
    """

    if st is None:
        return

    mode = (theme_mode or "Dark").strip().lower()
    if mode.startswith("light"):
        st.markdown(
            """
<style>
/* Light mode overrides */
.stApp {
    background-color: #F7FBF8;
    color: #0B1F14;
}

/* Increase base readability */
.stApp, .stApp p, .stApp li {
    font-size: 1.02rem;
    line-height: 1.55;
}

/* Main text (markdown + captions) */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #0B1F14;
}

[data-testid="stCaptionContainer"] {
    color: rgba(11, 31, 20, 0.75);
}

/* Help text / small UI text */
[data-testid="stHelp"],
small,
.stMarkdown small {
    color: rgba(11, 31, 20, 0.80);
}

/* Widget labels */
[data-testid="stWidgetLabel"] {
    color: #0B1F14;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #EAF5EF;
}

[data-testid="stSidebar"] * {
    color: #0B1F14;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #0B1F14;
}

/* Inputs */
div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.75);
}

input, textarea {
    color: #0B1F14 !important;
}

/* Buttons */
button[kind="primary"],
button[kind="secondary"],
button {
    color: #0B1F14 !important;
}

/* Download buttons: make text + background clearly readable */
div[data-testid="stDownloadButton"] button {
    background: #146B3A !important;
    border: 1px solid rgba(11, 31, 20, 0.25) !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 1.02rem !important;
    padding: 0.60rem 0.90rem !important;
    border-radius: 10px !important;
}

div[data-testid="stDownloadButton"] button * {
    color: #FFFFFF !important;
}

div[data-testid="stDownloadButton"] button:hover {
    background: #0F5A30 !important;
    border-color: rgba(11, 31, 20, 0.35) !important;
}

/* Links */
a {
    color: #146B3A;
}

code, pre {
    color: #0B1F14;
    background: rgba(255,255,255,0.70);
    border: 1px solid rgba(11, 31, 20, 0.10);
    border-radius: 8px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: rgba(255,255,255,0.65);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.60);
    border: 1px solid rgba(11, 31, 20, 0.12);
    border-radius: 8px;
    padding: 8px 10px;
}
</style>
""",
            unsafe_allow_html=True,
        )
    else:
        # Dark mode overrides (Streamlit defaults can be a bit low-contrast depending on browser/theme)
        st.markdown(
            """
<style>
.stApp {
    background-color: #06130C;
    color: #E8F5E9;
}

/* Increase base readability */
.stApp, .stApp p, .stApp li {
    font-size: 1.02rem;
    line-height: 1.55;
}

/* Main text (markdown + captions) */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #E8F5E9;
}

[data-testid="stCaptionContainer"] {
    color: rgba(232, 245, 233, 0.78);
}

/* Help text / small UI text */
[data-testid="stHelp"],
small,
.stMarkdown small {
    color: rgba(232, 245, 233, 0.82);
}

/* Widget labels */
[data-testid="stWidgetLabel"] {
    color: #E8F5E9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0B2015;
}

[data-testid="stSidebar"] * {
    color: #E8F5E9;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #E8F5E9;
}

/* Inputs */
div[data-baseweb="select"] > div {
    background-color: rgba(16, 42, 29, 0.55);
}

input, textarea {
    color: #E8F5E9 !important;
}

/* Buttons */
button[kind="primary"],
button[kind="secondary"],
button {
    color: #E8F5E9 !important;
}

/* Download buttons: make text + background clearly readable */
div[data-testid="stDownloadButton"] button {
    background: rgba(46, 204, 113, 0.92) !important;
    border: 1px solid rgba(46, 204, 113, 0.95) !important;
    color: #06130C !important;
    font-weight: 800 !important;
    font-size: 1.02rem !important;
    padding: 0.60rem 0.90rem !important;
    border-radius: 10px !important;
}

div[data-testid="stDownloadButton"] button * {
    color: #06130C !important;
}

div[data-testid="stDownloadButton"] button:hover {
    background: rgba(46, 204, 113, 0.98) !important;
    border-color: rgba(46, 204, 113, 1.0) !important;
}

/* Links */
a {
    color: #7DCEA0;
}

code, pre {
    color: #E8F5E9;
    background: rgba(16,42,29,0.55);
    border: 1px solid rgba(232, 245, 233, 0.10);
    border-radius: 8px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: rgba(16, 42, 29, 0.35);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(16,42,29,0.45);
    border: 1px solid rgba(232, 245, 233, 0.10);
    border-radius: 8px;
    padding: 8px 10px;
}
</style>
""",
            unsafe_allow_html=True,
        )


def _register_plotly_template(name: str, *, base_name: str, overlay_layout: go.Layout) -> None:
    if name in pio.templates:
        return

    base = pio.templates[base_name] if base_name in pio.templates else None
    tpl = go.layout.Template(base) if base is not None else go.layout.Template()
    tpl.layout.update(overlay_layout)
    pio.templates[name] = tpl


def configure_plotly_theme(theme_mode: str | None = None) -> None:
    """Configure Plotly defaults to match the selected app theme."""

    mode = (theme_mode or "Dark").strip().lower()

    dark_overlay = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8F5E9"),
        # High-contrast qualitative palette so grouped traces are clearly distinguishable.
        # (The earlier monochrome-green palette made categories hard to tell apart.)
        colorway=[
            "#2ECC71",  # green
            "#3498DB",  # blue
            "#E74C3C",  # red
            "#F1C40F",  # yellow
            "#9B59B6",  # purple
            "#1ABC9C",  # teal
            "#E67E22",  # orange
            "#EC407A",  # pink
            "#95A5A6",  # gray
        ],
        xaxis=dict(
            gridcolor="rgba(232,245,233,0.12)",
            zerolinecolor="rgba(232,245,233,0.20)",
            linecolor="rgba(232,245,233,0.25)",
        ),
        yaxis=dict(
            gridcolor="rgba(232,245,233,0.12)",
            zerolinecolor="rgba(232,245,233,0.20)",
            linecolor="rgba(232,245,233,0.25)",
        ),
        legend=dict(
            bgcolor="rgba(16,42,29,0.35)",
            bordercolor="rgba(232,245,233,0.10)",
            borderwidth=1,
            title=dict(font=dict(color="#E8F5E9")),
        ),
        annotationdefaults=dict(
            font=dict(color="#E8F5E9", size=12),
            bgcolor="rgba(16,42,29,0.55)",
            bordercolor="rgba(232,245,233,0.12)",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    light_overlay = go.Layout(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#0B1F14"),
        colorway=[
            "#1E8449",
            "#2980B9",
            "#C0392B",
            "#B7950B",
            "#7D3C98",
            "#117A65",
            "#AF601A",
            "#AD1457",
            "#616A6B",
        ],
        xaxis=dict(
            gridcolor="rgba(11,31,20,0.10)",
            zerolinecolor="rgba(11,31,20,0.18)",
            linecolor="rgba(11,31,20,0.22)",
        ),
        yaxis=dict(
            gridcolor="rgba(11,31,20,0.10)",
            zerolinecolor="rgba(11,31,20,0.18)",
            linecolor="rgba(11,31,20,0.22)",
        ),
        legend=dict(
            bgcolor="rgba(234,245,239,0.70)",
            bordercolor="rgba(11,31,20,0.10)",
            borderwidth=1,
            title=dict(font=dict(color="#0B1F14")),
        ),
        annotationdefaults=dict(
            font=dict(color="#0B1F14", size=12),
            bgcolor="rgba(255,255,255,0.80)",
            bordercolor="rgba(11,31,20,0.12)",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    _register_plotly_template(PLOTLY_THEME_NAME, base_name="plotly_dark", overlay_layout=dark_overlay)
    _register_plotly_template(
        PLOTLY_THEME_LIGHT_NAME, base_name="plotly_white", overlay_layout=light_overlay
    )

    pio.templates.default = PLOTLY_THEME_LIGHT_NAME if mode.startswith("light") else PLOTLY_THEME_NAME


def audience_selector(*, default: str = "Semi-technical") -> str:
    """Persistent audience selector for all pages.

    Stores selection in st.session_state["audience"]. If Streamlit is not available,
    returns the default.
    """

    if st is None:
        return default

    if default not in AUDIENCE_OPTIONS:
        default = "Semi-technical"

    if "audience" not in st.session_state:
        st.session_state["audience"] = default

    # Render in sidebar; keep consistent across pages
    idx = AUDIENCE_OPTIONS.index(st.session_state["audience"])
    choice = st.sidebar.radio("Audience", options=AUDIENCE_OPTIONS, index=idx)
    st.session_state["audience"] = choice
    return choice


def render_audience_markdown(blocks: dict[str, str], *, audience: str) -> None:
    """Render the markdown for the current audience.

    `blocks` should have keys matching AUDIENCE_OPTIONS.
    """

    if st is None:
        return

    md = blocks.get(audience) or blocks.get("Semi-technical") or next(iter(blocks.values()), "")
    if md:
        st.markdown(md)


def apply_global_filters(
    df: pd.DataFrame,
    *,
    key_prefix: str = "global",
    enabled_by_default: bool = True,
) -> pd.DataFrame:
    """Render a global filter panel in the sidebar and return the filtered dataframe.

    - Filter selections are persisted in st.session_state.
    - Designed to be called on every page so filters apply consistently.
    - If Streamlit is not available, returns df unchanged.
    """

    if st is None:
        return df

    if df is None or df.empty:
        return df

    # Keep key naming stable across pages.
    def _k(name: str) -> str:
        return f"{key_prefix}__{name}"

    enable_key = _k("enabled")
    if enable_key not in st.session_state:
        st.session_state[enable_key] = bool(enabled_by_default)

    with st.sidebar:
        st.markdown("### Global filters")
        enabled = st.toggle(
            "Enable global filters",
            key=enable_key,
            help="When enabled, filters apply to charts and tables on every page.",
        )

        with st.expander("Filters", expanded=False):
            st.caption("Tip: Keep groupings small (e.g., ≤ 12 categories) for clearer plots.")

            # Choose a pragmatic set of columns if they exist.
            preferred_cat_cols = [
                "Attrition",
                "Department",
                "JobRole",
                "OverTime",
                "Gender",
                "MaritalStatus",
                "BusinessTravel",
                "JobLevel",
                "EducationField",
            ]
            preferred_num_cols = [
                "Age",
                "MonthlyIncome",
                "DistanceFromHome",
                "YearsAtCompany",
                "TotalWorkingYears",
            ]

            cat_cols = [c for c in preferred_cat_cols if c in df.columns]
            num_cols = [c for c in preferred_num_cols if c in df.columns]

            # Track keys to enable a clean reset.
            reset_keys: list[str] = [enable_key]

            if cat_cols:
                st.markdown("**Categorical**")
                for col in cat_cols:
                    key = _k(f"cat__{col}")
                    reset_keys.append(key)
                    opts = (
                        df[col]
                        .astype(str)
                        .fillna("(missing)")
                        .unique()
                        .tolist()
                    )
                    opts = sorted(opts)
                    st.multiselect(
                        col,
                        options=opts,
                        default=st.session_state.get(key, []),
                        key=key,
                    )

            if num_cols:
                st.markdown("**Numeric**")
                for col in num_cols:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    if s.empty:
                        continue
                    # Use robust bounds to avoid a single outlier dominating the slider.
                    lo = float(np.percentile(s, 1))
                    hi = float(np.percentile(s, 99))
                    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                        lo = float(s.min())
                        hi = float(s.max())
                    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                        continue

                    key = _k(f"num__{col}")
                    reset_keys.append(key)
                    default_val = st.session_state.get(key, (lo, hi))
                    try:
                        default_val = (float(default_val[0]), float(default_val[1]))
                    except Exception:
                        default_val = (lo, hi)

                    st.slider(
                        col,
                        min_value=float(lo),
                        max_value=float(hi),
                        value=(
                            max(float(lo), min(float(hi), default_val[0])),
                            max(float(lo), min(float(hi), default_val[1])),
                        ),
                        key=key,
                    )

            if st.button("Reset filters", type="secondary"):
                for k in reset_keys:
                    st.session_state.pop(k, None)
                st.session_state[enable_key] = bool(enabled_by_default)
                st.rerun()

    if not bool(st.session_state.get(enable_key, enabled_by_default)):
        return df

    filtered = df

    # Apply categorical filters
    for col in [c for c in df.columns if _k(f"cat__{c}") in st.session_state]:
        picked = st.session_state.get(_k(f"cat__{col}"), [])
        if picked:
            s = filtered[col].astype(str).fillna("(missing)")
            filtered = filtered[s.isin([str(v) for v in picked])]

    # Apply numeric filters
    for col in [c for c in df.columns if _k(f"num__{c}") in st.session_state]:
        rng = st.session_state.get(_k(f"num__{col}"))
        if not rng or len(rng) != 2:
            continue
        try:
            lo, hi = float(rng[0]), float(rng[1])
        except Exception:
            continue
        s = pd.to_numeric(filtered[col], errors="coerce")
        filtered = filtered[(s >= lo) & (s <= hi)]

    # Small summary so users understand what they're seeing.
    try:
        if len(filtered) != len(df):
            st.sidebar.caption(f"Filtered rows: {len(filtered):,} / {len(df):,}")
    except Exception:
        pass

    return filtered


def dataframe_to_excel_bytes(df: pd.DataFrame, *, sheet_name: str = "data") -> bytes:
    """Convert a dataframe to an .xlsx file (as bytes) for download."""

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "data")
    return bio.getvalue()


def download_dataframe(
    df: pd.DataFrame,
    *,
    file_stem: str,
    label: str = "Download",
    csv_kwargs: dict | None = None,
    excel_sheet_name: str = "data",
) -> None:
    """Render CSV + Excel + TXT download buttons for a dataframe."""

    if st is None:
        return

    csv_kwargs = csv_kwargs or {}
    csv_bytes = df.to_csv(index=False, **csv_kwargs).encode("utf-8")
    # TXT export defaults to TSV for good readability + compatibility.
    txt_bytes = df.to_csv(index=False, sep="\t").encode("utf-8")
    xlsx_bytes = dataframe_to_excel_bytes(df, sheet_name=excel_sheet_name)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            f"{label} CSV",
            data=csv_bytes,
            file_name=f"{file_stem}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            f"{label} Excel",
            data=xlsx_bytes,
            file_name=f"{file_stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            f"{label} TXT",
            data=txt_bytes,
            file_name=f"{file_stem}.txt",
            mime="text/plain",
            use_container_width=True,
        )


def _audience_slug(audience: str) -> str:
    return (
        (audience or "audience")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _coerce_to_report_figures(
    figures: Sequence[tuple[str, object]] | None,
) -> list[tuple[str, go.Figure]]:
    out: list[tuple[str, go.Figure]] = []
    if not figures:
        return out
    for title, fig in figures:
        if fig is None:
            continue
        if isinstance(fig, go.Figure):
            out.append((title, fig))
            continue
        # Many plotly express figs are also go.Figure, but keep a defensive branch
        if hasattr(fig, "to_plotly_json"):
            out.append((title, go.Figure(fig)))
            continue
        raise TypeError(f"Unsupported figure type for report: {type(fig)!r}")
    return out


def build_plotly_html_report_bytes(
    *,
    title: str,
    audience_markdown: dict[str, str],
    theme_mode: str | None,
    selected_audience: str | None,
    figures: Sequence[tuple[str, object]] | None = None,
    tables: Sequence[tuple[str, pd.DataFrame]] | None = None,
    include_plotlyjs: str | bool = True,
) -> bytes:
    """Build a self-contained-ish HTML report including Plotly figures.

    - If `selected_audience` is None, includes *all* audiences grouped.
    - If provided, includes only that audience block.
    - Plotly figures are embedded as interactive divs.
    """

    report_figs = _coerce_to_report_figures(figures)
    report_tables = list(tables) if tables else []

    mode = (theme_mode or "Dark").strip().lower()
    is_light = mode.startswith("light")

    bg = "#F7FBF8" if is_light else "#06130C"
    fg = "#0B1F14" if is_light else "#E8F5E9"
    muted = "rgba(11,31,20,0.72)" if is_light else "rgba(232,245,233,0.75)"
    card = "rgba(255,255,255,0.65)" if is_light else "rgba(16,42,29,0.45)"
    border = "rgba(11,31,20,0.12)" if is_light else "rgba(232,245,233,0.10)"
    link = "#146B3A" if is_light else "#7DCEA0"

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def render_md_block(md: str) -> str:
        # Avoid adding a markdown dependency; preserve readability with pre-wrap.
        safe = html.escape(md or "")
        return f'<div class="md">{safe}</div>'

    def render_table(title_: str, df: pd.DataFrame) -> str:
        try:
            table_html = df.to_html(index=False, escape=True)
        except Exception:
            table_html = html.escape(str(df))
            table_html = f"<pre>{table_html}</pre>"
        return (
            f"<section class='card'>"
            f"<h3>{html.escape(title_)}</h3>"
            f"<div class='table'>{table_html}</div>"
            f"</section>"
        )

    # Build Plotly fragments; include plotlyjs only once.
    plotly_fragments: list[str] = []
    for i, (fig_title, fig) in enumerate(report_figs):
        inc = include_plotlyjs if i == 0 else False
        div = pio.to_html(fig, include_plotlyjs=inc, full_html=False)
        plotly_fragments.append(
            f"<section class='card'><h3>{html.escape(fig_title)}</h3>{div}</section>"
        )

    # Audience sections
    audience_sections: list[tuple[str, str]]
    if selected_audience:
        audience_sections = [(selected_audience, audience_markdown.get(selected_audience, ""))]
    else:
        # Keep stable ordering
        audience_sections = [(a, audience_markdown.get(a, "")) for a in AUDIENCE_OPTIONS]

    audience_html = "".join(
        (
            f"<section class='card'>"
            f"<h2>{html.escape(audience_name)}</h2>"
            f"{render_md_block(md)}"
            f"</section>"
        )
        for audience_name, md in audience_sections
        if (md or "").strip()
    )

    tables_html = "".join(render_table(tname, tdf) for tname, tdf in report_tables if tdf is not None)

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: {bg};
      --fg: {fg};
      --muted: {muted};
      --card: {card};
      --border: {border};
      --link: {link};
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: var(--bg);
      color: var(--fg);
      line-height: 1.55;
    }}
    a {{ color: var(--link); }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 22px 18px; }}
    header {{ margin-bottom: 14px; }}
    h1 {{ margin: 0 0 6px 0; font-size: 1.7rem; }}
    .meta {{ color: var(--muted); font-size: 0.95rem; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 14px;
      overflow-x: auto;
    }}
    h2 {{ margin: 0 0 8px 0; font-size: 1.25rem; }}
    h3 {{ margin: 0 0 8px 0; font-size: 1.1rem; }}
    .md {{ white-space: pre-wrap; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.95rem; }}
    th, td {{ border: 1px solid var(--border); padding: 6px 8px; text-align: left; }}
    th {{ background: rgba(0,0,0,0.06); }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>{html.escape(title)}</h1>
      <div class="meta">Generated: {html.escape(ts)}{(' · Audience: ' + html.escape(selected_audience)) if selected_audience else ' · Grouped by audience'}</div>
    </header>

    <div class="grid">
      {audience_html}
      {''.join(plotly_fragments)}
      {tables_html}
    </div>
  </div>
</body>
</html>
"""
    return doc.encode("utf-8")


def download_plotly_html_report(
    *,
    title: str,
    file_stem: str,
    audience: str,
    audience_markdown: dict[str, str],
    theme_mode: str | None,
    figures: Sequence[tuple[str, object]] | None = None,
    tables: Sequence[tuple[str, pd.DataFrame]] | None = None,
    include_plotlyjs: str | bool = True,
) -> None:
    """Render download buttons for HTML reports (current audience + grouped)."""

    if st is None:
        return

    safe_aud = _audience_slug(audience)
    current_bytes = build_plotly_html_report_bytes(
        title=title,
        audience_markdown=audience_markdown,
        theme_mode=theme_mode,
        selected_audience=audience,
        figures=figures,
        tables=tables,
        include_plotlyjs=include_plotlyjs,
    )
    grouped_bytes = build_plotly_html_report_bytes(
        title=title,
        audience_markdown=audience_markdown,
        theme_mode=theme_mode,
        selected_audience=None,
        figures=figures,
        tables=tables,
        include_plotlyjs=include_plotlyjs,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download report (current audience)",
            data=current_bytes,
            file_name=f"{file_stem}__{safe_aud}.html",
            mime="text/html",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download report (all audiences)",
            data=grouped_bytes,
            file_name=f"{file_stem}__all_audiences.html",
            mime="text/html",
            use_container_width=True,
        )
