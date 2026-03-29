from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import dashboard_utils as du


AUDIENCE_MD = {
    "Non-technical": """
This page shows an *estimated* attrition probability using the trained model.

Use it to understand what profiles the model considers higher risk.
""".strip(),
    "Semi-technical": """
Predictions are produced by a trained sklearn pipeline (preprocessing + model).

We use rows from the cleaned dataset as example inputs.
""".strip(),
    "Technical": """
The model expects the engineered interaction columns (e.g. `inter_pos_###`) present in the cleaned dataset.
""".strip(),
}


def main() -> None:
    du.set_page_config(page_title="Prediction", layout="wide")

    theme_mode = du.theme_selector()
    audience = du.audience_selector()
    du.apply_app_theme(theme_mode)
    du.configure_plotly_theme(theme_mode)

    st.title("Prediction — Attrition Probability")

    with st.expander("How to read this page", expanded=True):
        du.render_audience_markdown(audience, AUDIENCE_MD)

    base_df_full = du.load_cleaned_dataset()
    base_df = du.apply_global_filters(base_df_full)

    st.sidebar.header("Model")
    model_label = st.sidebar.selectbox("Choose model", list(du.MODEL_CANDIDATES.keys()), index=0)

    overlay_opacity = st.sidebar.slider(
        "Overlay opacity (grouped histograms)",
        min_value=0.05,
        max_value=0.95,
        value=0.45,
        step=0.05,
        help="Lower opacity makes overlapping groups easier to distinguish.",
    )

    pipeline, meta = du.load_pipeline(model_label)

    st.caption(f"Model file: `{meta.get('artifact_name', '')}`")

    if base_df.empty:
        st.info("No data after filters.")
        return

    idx_max = len(base_df) - 1
    row_idx = st.sidebar.number_input("Row index", min_value=0, max_value=int(idx_max), value=0, step=1)

    row = base_df.iloc[[int(row_idx)]].copy()
    required = getattr(pipeline, "feature_names_in_", None)
    if required is not None:
        row = du.ensure_required_columns(row, required)

    proba = float(du.predict_proba_attrition(pipeline, row)[0])

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(du.probability_indicator_figure(proba), use_container_width=True)

    with right:
        st.subheader("Input row (preview)")
        st.dataframe(row.iloc[:, : min(20, row.shape[1])], use_container_width=True, height=320)
        if row.shape[1] > 20:
            st.caption(f"Showing first 20 of {row.shape[1]} columns")

    st.divider()

    st.subheader("Distribution context")
    num_cols = [c for c in ["Age", "MonthlyIncome", "DistanceFromHome", "YearsAtCompany"] if c in base_df.columns]
    if num_cols:
        feature = st.selectbox("Numeric feature", num_cols, index=0)
        if "Attrition" in base_df.columns:
            fig = px.histogram(base_df, x=feature, color="Attrition", barmode="overlay", opacity=overlay_opacity)
        else:
            fig = px.histogram(base_df, x=feature)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        du.download_plotly_html_report([fig], filename="prediction_context.html", title=f"Prediction context — {feature}")
    else:
        st.info("No common numeric columns available to plot context.")

    st.subheader("Export")
    du.download_dataframe(row, label="Download selected row (CSV)", filename="prediction_row.csv")


if __name__ == "__main__":
    main()
