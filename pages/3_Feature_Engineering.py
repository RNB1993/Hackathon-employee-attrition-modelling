from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard_utils import (
    apply_app_theme,
    audience_selector,
    configure_plotly_theme,
    correlation_pairs,
    download_dataframe,
    interaction_importance_table,
    interaction_mapping,
    load_cleaned_dataset,
    numeric_df,
    render_audience_markdown,
    theme_selector,
)

st.set_page_config(page_title="Feature Engineering", layout="wide")

theme_mode = theme_selector()
apply_app_theme(theme_mode)
configure_plotly_theme(theme_mode)

st.title("Feature Engineering — Correlations & Interactions")

df = load_cleaned_dataset()
num = numeric_df(df)

audience = audience_selector()

render_audience_markdown(
    {
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
    },
    audience=audience,
)

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
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top correlated pairs")
pairs = correlation_pairs(df)
filtered = pairs[(pairs["spearman_r"].abs() >= abs_r_min)].copy().head(top_n)
st.dataframe(filtered, use_container_width=True)
download_dataframe(filtered, file_stem="correlation_pairs_filtered", label="Download table")

st.subheader("Interaction feature mapping")
map_df = interaction_mapping(df)
st.caption("This maps each engineered `inter_pos_*` / `inter_neg_*` to its raw numeric feature pair.")
st.dataframe(map_df.head(50), use_container_width=True)
download_dataframe(map_df, file_stem="interaction_mapping", label="Download mapping")

st.subheader("Top interaction predictors (from saved model)")
imp = interaction_importance_table(model_label=model_label, dataset_for_mapping=df, top_n=25)
st.dataframe(imp, use_container_width=True)
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
st.plotly_chart(fig2, use_container_width=True)
