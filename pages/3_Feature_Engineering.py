from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st

from dashboard_utils import (
    correlation_pairs,
    interaction_importance_table,
    interaction_mapping,
    load_cleaned_dataset,
    numeric_df,
)

st.set_page_config(page_title="Feature Engineering", layout="wide")

st.title("Feature Engineering — Correlations & Interactions")

df = load_cleaned_dataset()
num = numeric_df(df)

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
pairs = pairs[pairs["feature_1"] != "Attrition"]

filtered = pairs[(pairs["spearman_r"].abs() >= abs_r_min)].copy().head(top_n)
st.dataframe(filtered, use_container_width=True)

st.subheader("Interaction feature mapping")
map_df = interaction_mapping(df)
st.caption("This maps each engineered `inter_pos_*` / `inter_neg_*` to its raw numeric feature pair.")
st.dataframe(map_df.head(50), use_container_width=True)

st.subheader("Top interaction predictors (from saved model)")
imp = interaction_importance_table(model_label=model_label, dataset_for_mapping=df, top_n=25)
st.dataframe(imp, use_container_width=True)

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
