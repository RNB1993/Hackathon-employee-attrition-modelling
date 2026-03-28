from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

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

/* Main text (markdown + captions) */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #0B1F14;
}

[data-testid="stCaptionContainer"] {
    color: rgba(11, 31, 20, 0.75);
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

/* Buttons */
button[kind="primary"],
button[kind="secondary"],
button {
    color: #0B1F14;
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
        colorway=[
            "#2ECC71",
            "#27AE60",
            "#A3E4D7",
            "#58D68D",
            "#1E8449",
            "#52BE80",
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
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    light_overlay = go.Layout(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#0B1F14"),
        colorway=[
            "#1E8449",
            "#27AE60",
            "#2ECC71",
            "#16A085",
            "#52BE80",
            "#7DCEA0",
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
    """Render CSV + Excel download buttons for a dataframe."""

    if st is None:
        return

    csv_kwargs = csv_kwargs or {}
    csv_bytes = df.to_csv(index=False, **csv_kwargs).encode("utf-8")
    xlsx_bytes = dataframe_to_excel_bytes(df, sheet_name=excel_sheet_name)

    c1, c2 = st.columns(2)
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
