from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import dashboard_utils as du


AUDIENCE_MD = {
    "Non-technical": """
This page highlights correlations (numeric variables) and model-driven feature importance.

Use it to spot which combinations of factors tend to move together.
""".strip(),
    "Semi-technical": """
- Correlations use **Spearman r** (rank correlation) to reduce sensitivity to outliers.
- Interaction ranking is computed from model coefficients / importances when available.
""".strip(),
    "Technical": """
Correlations are computed on numeric-only columns. Model ranking is in the transformed feature space (post-preprocessing).
""".strip(),
}


def main() -> None:
    du.set_page_config(page_title="Feature Engineering", layout="wide")

    theme_mode = du.theme_selector()
    audience = du.audience_selector()
    du.apply_app_theme(theme_mode)
    du.configure_plotly_theme(theme_mode)

    st.title("Feature Engineering — Correlations & Interactions")

    with st.expander("How to read this page", expanded=True):
        du.render_audience_markdown(audience, AUDIENCE_MD)

    df = du.load_cleaned_dataset()
    df = du.apply_global_filters(df)

    num = du.numeric_df(df)

    st.sidebar.header("Controls")
    abs_r_min = st.sidebar.slider("Min |Spearman r|", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    top_n = st.sidebar.slider("Top N correlations", min_value=10, max_value=200, value=40, step=10)
    model_label = st.sidebar.selectbox("Model for interaction ranking", list(du.MODEL_CANDIDATES.keys()), index=0)

    st.subheader("Spearman correlation heatmap (numeric)")
    if num.empty:
        st.info("No numeric columns available after filters.")
    else:
        corr = num.corr(method="spearman")
        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Spearman correlation heatmap (numeric)",
        )
        fig.update_layout(height=600, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
        du.download_plotly_html_report([fig], filename="correlation_heatmap.html", title="Correlation heatmap")

    st.subheader("Top correlation pairs")
    pairs = du.correlation_pairs(num, method="spearman")
    if not pairs.empty:
        pairs = pairs[pairs["r"].abs() >= abs_r_min].head(int(top_n))
        st.dataframe(pairs, use_container_width=True, height=360)
        du.download_dataframe(pairs, label="Download correlation pairs (CSV)", filename="correlation_pairs.csv")
    else:
        st.info("Not enough numeric columns to compute pairs.")

    st.subheader("Model feature ranking (best effort)")
    try:
        pipeline, meta = du.load_pipeline(model_label)
        imp = du.interaction_importance_table(pipeline, top_n=35)
    except Exception as e:
        st.warning(f"Could not load model ranking: {e}")
        imp = pd.DataFrame(columns=["feature", "importance"])

    if imp.empty:
        st.info("No importances available for the selected model.")
    else:
        st.dataframe(imp, use_container_width=True, height=360)
        bar = px.bar(imp.head(25)[::-1], x="importance", y="feature", orientation="h", title="Top feature importances")
        bar.update_layout(height=640)
        st.plotly_chart(bar, use_container_width=True)
        du.download_dataframe(imp, label="Download importance table (CSV)", filename="feature_importance.csv")
        du.download_plotly_html_report([bar], filename="feature_importance.html", title="Feature importance")


if __name__ == "__main__":
    main()
