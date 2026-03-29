from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import streamlit as st

import dashboard_utils as du


AUDIENCE_MD = {
    "Non-technical": """
This page provides simple statistical tests to support (or challenge) visual patterns.

Treat results as *directional* rather than definitive.
""".strip(),
    "Semi-technical": """
Numeric: Welch t-test + Mann–Whitney. Categorical: Chi-square + Cramér’s V.

Assumption checks are shown for context.
""".strip(),
    "Technical": """
Use this page to quickly screen many columns; then validate with deeper modelling / controlled analysis.
""".strip(),
}


def _cramers_v(confusion: pd.DataFrame) -> float:
    if confusion.empty:
        return float("nan")
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    if n == 0:
        return float("nan")
    r, k = confusion.shape
    return math.sqrt((chi2 / n) / max(1, min(k - 1, r - 1)))


def _numeric_report(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    target = df[target_col].astype(str)
    classes = [c for c in ["No", "Yes"] if c in target.unique().tolist()]
    if len(classes) < 2:
        classes = target.dropna().unique().tolist()[:2]
    if len(classes) < 2:
        return pd.DataFrame()

    num_cols = df.select_dtypes(include="number").columns.tolist()
    rows: list[dict[str, object]] = []

    a = classes[0]
    b = classes[1]

    for col in num_cols:
        xa = pd.to_numeric(df.loc[target == a, col], errors="coerce").dropna()
        xb = pd.to_numeric(df.loc[target == b, col], errors="coerce").dropna()
        if len(xa) < 5 or len(xb) < 5:
            continue

        t_stat, t_p = stats.ttest_ind(xa, xb, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(xa, xb, alternative="two-sided")

        rows.append(
            {
                "feature": col,
                f"mean_{a}": float(xa.mean()),
                f"mean_{b}": float(xb.mean()),
                "welch_t_p": float(t_p),
                "mannwhitney_p": float(u_p),
                "n_a": int(len(xa)),
                "n_b": int(len(xb)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("mannwhitney_p", ascending=True)


def _categorical_report(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    target = df[target_col].astype(str)
    classes = [c for c in ["No", "Yes"] if c in target.unique().tolist()]
    if len(classes) < 2:
        classes = target.dropna().unique().tolist()[:2]
    if len(classes) < 2:
        return pd.DataFrame()

    cat_cols = [c for c in df.columns if c != target_col and df[c].dtype == object]
    rows: list[dict[str, object]] = []

    for col in cat_cols:
        ct = pd.crosstab(df[col].astype(str), target)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue

        chi2, p, dof, _ = stats.chi2_contingency(ct)
        rows.append(
            {
                "feature": col,
                "chi2_p": float(p),
                "cramers_v": float(_cramers_v(ct)),
                "dof": int(dof),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("chi2_p", ascending=True)


def main() -> None:
    du.set_page_config(page_title="Stats", layout="wide")

    theme_mode = du.theme_selector()
    audience = du.audience_selector()
    du.apply_app_theme(theme_mode)
    du.configure_plotly_theme(theme_mode)

    st.title("Stats — Quick Tests")
    st.caption("Stat testing page.\n\nNumeric: Welch t-test + Mann–Whitney; Categorical: Chi-square + Cramér’s V.\nAssumption checks shown for context.")

    with st.expander("How to read this page", expanded=True):
        du.render_audience_markdown(audience, AUDIENCE_MD)

    df = du.load_cleaned_dataset()
    df = du.apply_global_filters(df)

    if "Attrition" not in df.columns:
        st.error("Expected column 'Attrition' not found.")
        return

    target_col = "Attrition"

    overlay_opacity = st.sidebar.slider(
        "Overlay opacity (grouped histograms)",
        min_value=0.05,
        max_value=0.95,
        value=0.45,
        step=0.05,
        help="Lower opacity makes overlapping groups easier to distinguish.",
    )

    tabs = st.tabs(["Numeric", "Categorical"])

    with tabs[0]:
        rep = _numeric_report(df, target_col)
        if rep.empty:
            st.info("No numeric columns or insufficient data after filters.")
        else:
            st.subheader("Numeric tests")
            st.dataframe(rep, use_container_width=True, height=340)
            du.download_dataframe(rep, label="Download numeric report (CSV)", filename="stats_numeric.csv")

            feat = st.selectbox("Feature to visualize", rep["feature"].tolist(), index=0)
            fig = px.histogram(df, x=feat, color=target_col, barmode="overlay", opacity=overlay_opacity)
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

            du.download_plotly_html_report([fig], filename="stats_numeric_plot.html", title=f"Stats plot — {feat}")

    with tabs[1]:
        rep = _categorical_report(df, target_col)
        if rep.empty:
            st.info("No categorical columns or insufficient data after filters.")
        else:
            st.subheader("Categorical tests")
            st.dataframe(rep, use_container_width=True, height=340)
            du.download_dataframe(rep, label="Download categorical report (CSV)", filename="stats_categorical.csv")

            feat = st.selectbox("Feature to visualize", rep["feature"].tolist(), index=0, key="cat_feat")
            plot_df = (
                df.assign(_count=1)
                .groupby([feat, target_col], as_index=False)["_count"]
                .sum()
            )
            fig = px.bar(plot_df, x=feat, y="_count", color=target_col, barmode="group")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

            du.download_plotly_html_report([fig], filename="stats_categorical_plot.html", title=f"Stats plot — {feat}")


if __name__ == "__main__":
    main()
