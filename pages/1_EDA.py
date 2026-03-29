from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import dashboard_utils as du


AUDIENCE_MD = {
    "Non-technical": """
This page helps you visually explore the dataset.

- Use **Univariate** plots to understand a single variable.
- Use **Bivariate** plots to compare two variables.
""".strip(),
    "Semi-technical": """
Focus on distribution shape (skew, outliers) and how patterns differ by Attrition.
""".strip(),
    "Technical": """
Use facets and color/hue to quickly spot interactions and potential confounders.
""".strip(),
}


def _available_columns(df: pd.DataFrame) -> list[str]:
    return sorted(df.columns.tolist())


def main() -> None:
    du.set_page_config(page_title="EDA", layout="wide")

    theme_mode = du.theme_selector()
    audience = du.audience_selector()
    du.apply_app_theme(theme_mode)
    du.configure_plotly_theme(theme_mode)

    st.title("EDA — Univariate & Bivariate (Plotly)")

    with st.expander("How to read this page", expanded=True):
        du.render_audience_markdown(audience, AUDIENCE_MD)

    df = du.load_cleaned_dataset()
    df = du.apply_global_filters(df)

    all_cols = _available_columns(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in all_cols if c not in numeric_cols]

    st.subheader("Chart controls")

    left, right = st.columns([1, 2])

    with left:
        plot_kind = st.radio("Plot", ["Univariate", "Bivariate"], horizontal=True)
        target_optional = st.selectbox("Target (optional)", [None, "Attrition"], index=1 if "Attrition" in df.columns else 0)

        if plot_kind == "Univariate":
            x_col = st.selectbox("Column", all_cols, index=all_cols.index("Age") if "Age" in all_cols else 0)
            chart_type = st.selectbox("Chart type", ["Histogram", "Box", "Bar"], index=0)
            hue = st.selectbox("Hue / Color", [None] + categorical_cols, index=(1 if target_optional in categorical_cols else 0))
            facet_col = st.selectbox("Facet col", [None] + categorical_cols, index=0)
            facet_row = st.selectbox("Facet row", [None] + categorical_cols, index=0)
        else:
            x_col = st.selectbox("X", all_cols, index=all_cols.index("Age") if "Age" in all_cols else 0)
            y_col = st.selectbox("Y", all_cols, index=all_cols.index("MonthlyIncome") if "MonthlyIncome" in all_cols else min(1, len(all_cols) - 1))
            chart_type = st.selectbox("Chart type", ["Scatter", "Box", "Violin"], index=0)
            hue = st.selectbox("Hue / Color", [None] + categorical_cols, index=(1 if target_optional in categorical_cols else 0))
            facet_col = st.selectbox("Facet col", [None] + categorical_cols, index=0)
            facet_row = st.selectbox("Facet row", [None] + categorical_cols, index=0)

        st.caption("Tip: facets work best with categorical columns.")

    with right:
        plot_state = {
            "Plot": plot_kind,
            "Chart": chart_type,
            "X": x_col,
            "Y": y_col if plot_kind == "Bivariate" else None,
            "Hue": hue,
            "Facet col": facet_col,
            "Facet row": facet_row,
        }

        if df.empty:
            st.info("No data after filters.")
            return

        fig = None

        if plot_kind == "Univariate":
            if chart_type == "Histogram":
                fig = px.histogram(df, x=x_col, color=hue, facet_col=facet_col, facet_row=facet_row, nbins=30, barmode="overlay")
            elif chart_type == "Box":
                fig = px.box(df, x=hue, y=x_col, color=hue, facet_col=facet_col, facet_row=facet_row)
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_col, color=hue, facet_col=facet_col, facet_row=facet_row)
        else:
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col, color=hue, facet_col=facet_col, facet_row=facet_row, trendline=None)
            elif chart_type == "Box":
                fig = px.box(df, x=x_col, y=y_col, color=hue, facet_col=facet_col, facet_row=facet_row)
            elif chart_type == "Violin":
                fig = px.violin(df, x=x_col, y=y_col, color=hue, facet_col=facet_col, facet_row=facet_row, box=True)

        if fig is None:
            st.info("Choose chart settings to render a plot.")
            return

        fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        desc = du.short_plot_state_description(plot_state)
        if desc:
            st.markdown(desc)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            du.download_dataframe(df, label="Download filtered dataset (CSV)", filename="eda_filtered.csv")
        with c2:
            du.download_plotly_html_report([fig], filename="eda_plot.html", title="EDA Plot")


if __name__ == "__main__":
    main()
