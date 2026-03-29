from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard_utils import (
    apply_app_theme,
    apply_global_filters,
    audience_selector,
    configure_plotly_theme,
    correlation_pairs,
    download_dataframe,
    download_plotly_html_report,
    interaction_importance_table,
    interaction_mapping,
    load_cleaned_dataset,
    numeric_df,
    render_audience_markdown,
    short_plot_state_description,
    theme_selector,
)

st.set_page_config(page_title="Feature Engineering", layout="wide")

theme_mode = theme_selector()
apply_app_theme(theme_mode)
configure_plotly_theme(theme_mode)

st.title("Feature Engineering — Correlations & Interactions")

df = load_cleaned_dataset()
df = apply_global_filters(df)
num = numeric_df(df)

audience = audience_selector()

AUDIENCE_MD = {
    "Non-technical": """
This page shows which factors tend to move together and how we built extra *combined* features.

These patterns can suggest which levers matter, but correlation does not prove cause.
""",
    "Semi-technical": """
Correlation view + engineered interaction features used by the prediction model.

- Heatmap: Spearman correlation (numeric)
- Interaction mapping: how new features were constructed
""",
    "Technical": """
Feature engineering audit:

- Spearman correlation matrix
- Pairwise correlation list (deduped)
- Interaction mapping to raw feature pairs
- Model coefficient ranking for interaction features
""",
}

render_audience_markdown(AUDIENCE_MD, audience=audience)

with st.expander("How to read this page", expanded=False):
    st.markdown(
        """
- The heatmap shows **Spearman correlation** between numeric variables (−1 to +1).
- The correlation pairs table highlights the strongest relationships.
- The interaction mapping explains the engineered features used by the prediction model.
"""
    )


def _download_caption() -> None:
    st.caption("Choose a format: CSV / Excel / TXT.")

with st.sidebar:
    st.header("Controls")
    abs_r_min = st.slider("Min |Spearman r|", 0.0, 1.0, 0.3, 0.05)
    top_n = st.slider("Top N correlations", 10, 200, 30, 10)
    model_label = st.selectbox("Model for interaction ranking", ["Best model", "Business-cost model"])

st.subheader("Spearman correlation heatmap (numeric)")
# Plotly heatmap
corr = num.corr(method="spearman")
fig = px.imshow(
    corr,
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    aspect="auto",
)
fig.update_layout(height=650)
st.plotly_chart(fig, width="stretch")

heatmap_state = {
    "chart": "Heatmap",
    "title": "Spearman correlation heatmap (numeric)",
    "global_filters_enabled": bool(st.session_state.get("global__enabled", True)),
    "n_rows": int(len(df)),
}
heatmap_state["short_description"] = short_plot_state_description(heatmap_state)
if heatmap_state["short_description"]:
    st.caption(f"Plot summary: {heatmap_state['short_description']}")

st.subheader("Top correlated pairs")
pairs = correlation_pairs(df)
filtered = pairs[(pairs["spearman_r"].abs() >= abs_r_min)].copy().head(top_n)
st.dataframe(filtered, width="stretch")
_download_caption()
download_dataframe(filtered, file_stem="correlation_pairs_filtered", label="Download table")

st.subheader("Interaction feature mapping")
map_df = interaction_mapping(df)
st.caption("This maps each engineered `inter_pos_*` / `inter_neg_*` to its raw numeric feature pair.")
st.dataframe(map_df.head(50), width="stretch")
_download_caption()
download_dataframe(map_df, file_stem="interaction_mapping", label="Download mapping")

st.subheader("Top interaction predictors (from saved model)")
imp = interaction_importance_table(model_label=model_label, dataset_for_mapping=df, top_n=25)
st.dataframe(imp, width="stretch")
_download_caption()
download_dataframe(imp, file_stem="interaction_importance_top25", label="Download ranking")

fig2 = px.bar(
    imp.sort_values("coef"),
    x="coef",
    y="interaction",
    color="direction",
    orientation="h",
    hover_data=["raw_feature_1", "raw_feature_2", "spearman_r", "odds_ratio"],
)
fig2.update_layout(height=700)
st.plotly_chart(fig2, width="stretch")

bar_state = {
    "chart": "Bar",
    "title": "Top interaction predictors (from saved model)",
    "model": model_label,
    "x": "coef",
    "y": "interaction",
    "color": "direction",
    "global_filters_enabled": bool(st.session_state.get("global__enabled", True)),
    "n_rows": int(len(df)),
}
bar_state["short_description"] = short_plot_state_description(bar_state)
if bar_state["short_description"]:
    st.caption(f"Plot summary: {bar_state['short_description']}")

st.subheader("Download page report (HTML)")
st.caption("Interactive HTML report (includes key charts and tables).")

plot_summaries_df = pd.DataFrame(
    [
        {"plot": "Spearman correlation heatmap", "description": heatmap_state.get("short_description", "")},
        {"plot": "Top interaction predictors", "description": bar_state.get("short_description", "")},
    ]
)
download_plotly_html_report(
    title="Feature Engineering — Correlations & Interactions",
    file_stem="report_feature_engineering",
    audience=audience,
    audience_markdown=AUDIENCE_MD,
    theme_mode=theme_mode,
    figures=[
        ("Spearman correlation heatmap", fig),
        ("Top interaction predictors", fig2),
    ],
    tables=[
        ("Plot summaries", plot_summaries_df),
        ("Top correlated pairs (filtered)", filtered),
        ("Interaction mapping (first 200 rows)", map_df.head(200)),
        ("Interaction importance (top 25)", imp),
    ],
)
