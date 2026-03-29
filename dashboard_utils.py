from __future__ import annotations

import html
import math
import pickle
from dataclasses import dataclass
from datetime import datetime as dt
from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

CLEANED_CSV_CANDIDATES: list[Path] = [
    DATA_DIR / "Cleaned_dataset" / "WA_Fn-UseC_-HR-Employee-Attrition_capped.csv",
    DATA_DIR / "Cleaned_data" / "WA_Fn-UseC_-HR-Employee-Attrition_capped.csv",
]

MODEL_DIR = REPO_ROOT / "jupyter_notebooks" / "model"
MODEL_CANDIDATES: dict[str, Path] = {
    "Best model": MODEL_DIR / "attrition_minority_yes_best_model.pkl",
    "Business-cost model": MODEL_DIR / "attrition_minority_yes_business_cost_second_best.pkl",
}


@dataclass(frozen=True)
class InteractionRow:
    name: str
    description: str


def _st_cache_data():
    return getattr(st, "cache_data", st.cache)


def _st_cache_resource():
    return getattr(st, "cache_resource", st.cache)


def set_page_config(*, page_title: str, layout: str = "wide") -> None:
    st.set_page_config(page_title=page_title, page_icon="📊", layout=layout)


def theme_selector(*, key: str = "theme_mode") -> str:
    default = st.session_state.get(key, "Light")
    theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0 if default == "Light" else 1, key=key)
    return theme


def audience_selector(*, key: str = "audience") -> str:
    default = st.session_state.get(key, "Non-technical")
    options = ["Non-technical", "Semi-technical", "Technical"]
    idx = options.index(default) if default in options else 0
    return st.sidebar.selectbox("Audience", options, index=idx, key=key)


def apply_app_theme(theme_mode: str) -> None:
    # Keep it lightweight: mainly tighten spacing and make charts fit well.
    base = "#0e1117" if theme_mode == "Dark" else "#ffffff"
    st.markdown(
        f"""
<style>
section.main > div {{ padding-top: 1.25rem; }}
.block-container {{ padding-top: 1.25rem; }}
[data-testid='stAppViewContainer'] {{ background: {base}; }}
</style>
""",
        unsafe_allow_html=True,
    )


def configure_plotly_theme(theme_mode: str) -> None:
    pio.templates.default = "plotly_dark" if theme_mode == "Dark" else "plotly"


def render_audience_markdown(audience: str, md_map: Mapping[str, str]) -> None:
    st.markdown(md_map.get(audience, ""))


@_st_cache_data()(show_spinner=False)
def load_cleaned_dataset() -> pd.DataFrame:
    for candidate in CLEANED_CSV_CANDIDATES:
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError(
        "Could not find cleaned dataset. Tried: " + ", ".join(str(p) for p in CLEANED_CSV_CANDIDATES)
    )


def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df

    with st.sidebar.expander("Global filters", expanded=False):
        if "Attrition" in filtered.columns:
            opts = sorted(filtered["Attrition"].dropna().astype(str).unique().tolist())
            selected = st.multiselect("Attrition", opts, default=opts)
            if selected:
                filtered = filtered[filtered["Attrition"].astype(str).isin(selected)]

        if "Department" in filtered.columns:
            opts = sorted(filtered["Department"].dropna().astype(str).unique().tolist())
            selected = st.multiselect("Department", opts, default=opts)
            if selected:
                filtered = filtered[filtered["Department"].astype(str).isin(selected)]

        if "JobRole" in filtered.columns:
            opts = sorted(filtered["JobRole"].dropna().astype(str).unique().tolist())
            selected = st.multiselect("JobRole", opts, default=opts)
            if selected:
                filtered = filtered[filtered["JobRole"].astype(str).isin(selected)]

        if "Age" in filtered.columns:
            ages = pd.to_numeric(filtered["Age"], errors="coerce")
            if ages.notna().any():
                min_age, max_age = int(ages.min()), int(ages.max())
                low, high = st.slider("Age range", min_age, max_age, (min_age, max_age))
                filtered = filtered[ages.between(low, high)]

    return filtered


def download_dataframe(df: pd.DataFrame, *, label: str = "Download CSV", filename: str = "data.csv") -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")


def download_plotly_html_report(
    figures: Sequence[go.Figure],
    *,
    label: str = "Download Plotly HTML report",
    filename: str = "report.html",
    title: str = "Dashboard report",
) -> None:
    parts = [
        "<!doctype html>",
        "<html><head>",
        f"<meta charset='utf-8'/><title>{html.escape(title)}</title>",
        "</head><body>",
        f"<h2>{html.escape(title)}</h2>",
        f"<p>Generated: {dt.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>",
    ]
    for fig in figures:
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        parts.append("<hr/>")
    parts.append("</body></html>")

    content = "\n".join(parts).encode("utf-8")
    st.download_button(label, content, file_name=filename, mime="text/html")


def short_plot_state_description(state: Mapping[str, object]) -> str:
    items = [f"- **{k}**: {v}" for k, v in state.items() if v is not None and v != ""]
    return "\n".join(items) if items else ""


def numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


def correlation_pairs(num_df: pd.DataFrame, *, method: str = "spearman") -> pd.DataFrame:
    if num_df.empty:
        return pd.DataFrame(columns=["feature_a", "feature_b", "r"])
    corr = num_df.corr(method=method)
    rows: list[dict[str, object]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append({"feature_a": cols[i], "feature_b": cols[j], "r": float(corr.iloc[i, j])})
    return pd.DataFrame(rows).sort_values("r", key=lambda s: s.abs(), ascending=False)


def metrics_bar_figure(metric_df: pd.DataFrame, *, x: str, y: str, color: str | None = None, title: str = "") -> go.Figure:
    fig = go.Figure()
    if color and color in metric_df.columns:
        for val in metric_df[color].dropna().unique().tolist():
            sub = metric_df[metric_df[color] == val]
            fig.add_bar(name=str(val), x=sub[x], y=sub[y])
    else:
        fig.add_bar(x=metric_df[x], y=metric_df[y])

    fig.update_layout(title=title, barmode="group", height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


@_st_cache_resource()(show_spinner=False)
def load_pipeline(model_label: str):
    path = MODEL_CANDIDATES.get(model_label)
    if not path or not path.exists():
        raise FileNotFoundError(f"Model not found for '{model_label}': {path}")

    obj = pickle.loads(path.read_bytes())
    if hasattr(obj, "predict_proba"):
        return obj, {"artifact_name": path.name}
    if isinstance(obj, dict) and "estimator" in obj:
        return obj["estimator"], obj

    raise TypeError(f"Unsupported model artifact type: {type(obj)}")


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in required:
        if col not in out.columns:
            out[col] = 0
    return out


def engineer_interactions(df: pd.DataFrame) -> pd.DataFrame:
    # The published dataset already contains engineered interaction columns like inter_pos_###.
    # Keep this as a placeholder so the prediction page can be resilient if an older dataset is used.
    return df


def predict_proba_attrition(pipeline, X: pd.DataFrame, *, positive_label: str = "Yes") -> np.ndarray:
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("Pipeline does not support predict_proba")

    proba = pipeline.predict_proba(X)
    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        return proba[:, 1]

    classes = list(classes)
    if positive_label in classes:
        return proba[:, classes.index(positive_label)]
    return proba[:, 1]


def probability_indicator_figure(probability: float, *, title: str = "Attrition probability") -> go.Figure:
    probability = float(max(0.0, min(1.0, probability)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#d62728"},
                "steps": [
                    {"range": [0, 33], "color": "#2ca02c"},
                    {"range": [33, 66], "color": "#ffdd57"},
                    {"range": [66, 100], "color": "#ff7f0e"},
                ],
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def interaction_mapping(df: pd.DataFrame) -> pd.DataFrame:
    # Best-effort mapping: list interaction columns present.
    cols = [c for c in df.columns if c.startswith("inter_")]
    return pd.DataFrame({"interaction": cols})


def interaction_importance_table(pipeline, *, top_n: int = 25) -> pd.DataFrame:
    # Attempt to compute importances in the transformed feature space.
    if not hasattr(pipeline, "named_steps"):
        return pd.DataFrame(columns=["feature", "importance"])

    preprocess = pipeline.named_steps.get("preprocess")
    model = pipeline.named_steps.get("model")
    if preprocess is None or model is None:
        return pd.DataFrame(columns=["feature", "importance"])

    try:
        feature_names = list(preprocess.get_feature_names_out())
    except Exception:
        feature_names = None

    importance = None
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2:
            coef = coef[0]
        importance = np.abs(coef)
    elif hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)

    if importance is None:
        return pd.DataFrame(columns=["feature", "importance"])

    if feature_names and len(feature_names) == len(importance):
        df_imp = pd.DataFrame({"feature": feature_names, "importance": importance})
    else:
        df_imp = pd.DataFrame({"feature": [f"f_{i}" for i in range(len(importance))], "importance": importance})

    return df_imp.sort_values("importance", ascending=False).head(int(top_n)).reset_index(drop=True)
