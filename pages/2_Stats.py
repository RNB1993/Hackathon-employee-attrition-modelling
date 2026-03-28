from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

from dashboard_utils import (
    apply_app_theme,
    audience_selector,
    configure_plotly_theme,
    download_dataframe,
    load_cleaned_dataset,
    render_audience_markdown,
    theme_selector,
)

st.set_page_config(page_title="Stats", layout="wide")

theme_mode = theme_selector()
apply_app_theme(theme_mode)
configure_plotly_theme(theme_mode)

st.title("Stats — Quick Tests")

df = load_cleaned_dataset()

audience = audience_selector()

render_audience_markdown(
    {
        "Non-technical": """
This page runs simple checks to see whether a factor differs between employees who **left** vs **stayed**.

Use it to support discussion—not as a final decision tool.
""",
        "Semi-technical": """
Quick hypothesis tests / association checks.

- Numeric vs target: Welch t-test and Mann–Whitney
- Categorical vs target: Chi-square + Cramér’s V
""",
        "Technical": """
Stat testing page.

Numeric: Welch t-test + Mann–Whitney; Categorical: Chi-square + Cramér’s V.
Assumption checks shown for context.
""",
    },
    audience=audience,
)

# Normalize target to binary 0/1
if "Attrition" in df.columns and df["Attrition"].dtype == object:
    y = (df["Attrition"].astype(str).str.lower() == "yes").astype(int)
    df = df.copy()
    df["Attrition_bin"] = y
    target_col = "Attrition_bin"
elif "Attrition" in df.columns:
    target_col = "Attrition"
else:
    target_col = st.selectbox("Target column", options=df.columns)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

with st.sidebar:
    st.header("Test selection")
    test_kind = st.radio(
        "Test",
        ["Numeric vs Target (t-test / Mann-Whitney)", "Categorical vs Target (Chi-square)"],
    )

if test_kind.startswith("Numeric"):
    col = st.selectbox("Numeric column", options=[c for c in num_cols if c != target_col])

    g0 = df[df[target_col] == 0][col].dropna()
    g1 = df[df[target_col] == 1][col].dropna()

    st.subheader("Group distributions")
    fig = px.histogram(df, x=col, color=target_col, barmode="overlay", marginal="box")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Assumption checks")
    sh0 = stats.shapiro(g0.sample(min(len(g0), 500), random_state=0)) if len(g0) >= 3 else None
    sh1 = stats.shapiro(g1.sample(min(len(g1), 500), random_state=0)) if len(g1) >= 3 else None
    lev = stats.levene(g0, g1) if len(g0) >= 2 and len(g1) >= 2 else None

    st.write(
        {
            "n_group0": int(len(g0)),
            "n_group1": int(len(g1)),
            "shapiro_p_group0": None if sh0 is None else float(sh0.pvalue),
            "shapiro_p_group1": None if sh1 is None else float(sh1.pvalue),
            "levene_p": None if lev is None else float(lev.pvalue),
        }
    )

    st.subheader("Tests")
    # Welch t-test is a good default when variances differ
    t_res = stats.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
    try:
        mw = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        mw_p = float(mw.pvalue)
    except Exception:
        mw_p = None

    results = {
        "welch_ttest_stat": float(t_res.statistic),
        "welch_ttest_p": float(t_res.pvalue),
        "mann_whitney_p": mw_p,
        "mean_group0": float(g0.mean()) if len(g0) else None,
        "mean_group1": float(g1.mean()) if len(g1) else None,
        "n_group0": int(len(g0)),
        "n_group1": int(len(g1)),
        "column": col,
        "target_col": target_col,
    }
    st.write(results)

    st.subheader("Download")
    download_dataframe(pd.DataFrame([results]), file_stem="stats_numeric_results", label="Download results")

else:
    cat = st.selectbox("Categorical column", options=[c for c in cat_cols if c != target_col])

    ct = pd.crosstab(df[cat], df[target_col], dropna=False)
    chi2, p, dof, expected = stats.chi2_contingency(ct)

    # Cramér's V
    n = ct.to_numpy().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    cramers_v = np.sqrt(phi2 / max(1, min(k - 1, r - 1)))

    st.subheader("Contingency table")
    st.dataframe(ct, use_container_width=True)

    st.subheader("Download")
    download_dataframe(ct.reset_index(), file_stem="stats_contingency_table", label="Download contingency")

    st.subheader("Chi-square results")
    chi_results = {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "cramers_v": float(cramers_v),
        "categorical_col": cat,
        "target_col": target_col,
    }
    st.write(chi_results)
    download_dataframe(pd.DataFrame([chi_results]), file_stem="stats_chi_square_results", label="Download results")

    st.subheader("Bar chart")
    plot_df = ct.reset_index().melt(id_vars=cat, var_name=target_col, value_name="count")
    fig = px.bar(plot_df, x=cat, y="count", color=target_col, barmode="group")
    st.plotly_chart(fig, use_container_width=True)
