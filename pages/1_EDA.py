from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard_utils import (
    audience_selector,
    configure_plotly_theme,
    download_dataframe,
    load_cleaned_dataset,
    render_audience_markdown,
)

st.set_page_config(page_title="EDA", layout="wide")

configure_plotly_theme()

st.title("EDA — Univariate & Bivariate (Plotly)")

df = load_cleaned_dataset()

audience = audience_selector()

render_audience_markdown(
    {
        "Non-technical": """
This page helps you *visually explore* patterns in the HR dataset.

- Use **Univariate** to see how one field is distributed.
- Use **Bivariate** to compare two fields (e.g., a factor vs attrition).
""",
        "Semi-technical": """
Use this page for interactive exploratory data analysis (EDA) with Plotly.

- Choose a column for **Univariate** distribution plots
- Use **Bivariate** to explore relationships and potential predictors
""",
        "Technical": """
Interactive EDA with Plotly (hist/box/violin/scatter) and optional facets.

Tip: use **color** + facets for quick stratified inspection before modeling.
""",
    },
    audience=audience,
)

all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes("number").columns.tolist()
categorical_cols = [c for c in all_cols if c not in numeric_cols]

st.subheader("Chart controls")
st.caption(
    "Controls are shown here (not only in the sidebar) so they're visible across all audience levels. "
    "Use Color + Facets to group and compare segments."
)

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.3, 1.3])
with c1:
    plot_kind = st.radio("Plot", ["Univariate", "Bivariate"], horizontal=True)
with c2:
    # Optional, used mainly for hover / defaults
    target_candidates = [c for c in df.columns if c.lower() == "attrition"]
    target_default = target_candidates[0] if target_candidates else all_cols[0]
    target = st.selectbox("Target (optional)", options=all_cols, index=all_cols.index(target_default))
with c3:
    color = st.selectbox("Hue / Color", options=[None] + all_cols, index=0)
with c4:
    facet_col = st.selectbox("Facet col", options=[None] + categorical_cols, index=0)

c5, c6 = st.columns([1.3, 1.3])
with c5:
    facet_row = st.selectbox("Facet row", options=[None] + categorical_cols, index=0)
with c6:
    st.caption("Tip: facets work best with categorical columns.")

if plot_kind == "Univariate":
    left, right = st.columns([1, 2])
    with left:
        col = st.selectbox("Column", options=all_cols)
        if col in numeric_cols:
            chart = st.selectbox("Chart", options=["Histogram", "Box", "Violin"], index=0)
            nbins = st.slider("Bins (hist)", 10, 100, 30)
        else:
            chart = st.selectbox("Chart", options=["Bar"], index=0)
            nbins = None

    with right:
        if chart == "Histogram":
            fig = px.histogram(
                df,
                x=col,
                color=color,
                facet_row=facet_row,
                facet_col=facet_col,
                nbins=nbins,
                barmode="overlay" if color else "relative",
                marginal="box",
            )
        elif chart == "Box":
            fig = px.box(
                df,
                x=color if color in categorical_cols else None,
                y=col,
                color=color,
                facet_row=facet_row,
                facet_col=facet_col,
                points="outliers",
            )
        elif chart == "Violin":
            fig = px.violin(
                df,
                x=color if color in categorical_cols else None,
                y=col,
                color=color,
                facet_row=facet_row,
                facet_col=facet_col,
                box=True,
                points="all",
            )
        else:
            counts = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="count")
            fig = px.bar(counts, x=col, y="count")

        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

else:
    left, right = st.columns([1, 2])
    with left:
        x = st.selectbox("X", options=all_cols, index=all_cols.index(target) if target in all_cols else 0)
        y_candidates = [c for c in all_cols if c != x]
        y = st.selectbox("Y", options=y_candidates, index=0)
        chart = st.selectbox("Chart", options=["Scatter", "Box", "Bar"], index=0)
        st.caption(
            "Scatter works best for numeric-numeric; Box is great for categorical vs numeric; "
            "Bar aggregates by group (e.g., mean of Y by X)."
        )

    with right:
        if chart == "Scatter":
            fig = px.scatter(
                df,
                x=x,
                y=y,
                color=color,
                facet_row=facet_row,
                facet_col=facet_col,
                hover_data=[target] if target in df.columns else None,
            )
        elif chart == "Box":
            fig = px.box(
                df,
                x=x,
                y=y,
                color=color,
                facet_row=facet_row,
                facet_col=facet_col,
                points="outliers",
            )
        else:
            # Bar: aggregate mean(y) by x (works best when x is categorical)
            if y in numeric_cols:
                agg = df.groupby(x, dropna=False)[y].mean().reset_index(name=f"mean_{y}")
                fig = px.bar(agg, x=x, y=f"mean_{y}", color=color if color == x else None)
            else:
                ct = df.groupby([x, y], dropna=False).size().reset_index(name="count")
                fig = px.bar(ct, x=x, y="count", color=y, barmode="group")

        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Dataset preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Download dataset")
download_dataframe(df, file_stem="hr_attrition_cleaned", label="Download dataset")
