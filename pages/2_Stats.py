from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

from dashboard_utils import (
    apply_app_theme,
    apply_global_filters,
    audience_selector,
    configure_plotly_theme,
    download_dataframe,
    download_plotly_html_report,
    load_cleaned_dataset,
    metrics_bar_figure,
    render_audience_markdown,
    short_plot_state_description,
    theme_selector,
)

st.set_page_config(page_title="Stats", layout="wide")

theme_mode = theme_selector()
apply_app_theme(theme_mode)
configure_plotly_theme(theme_mode)

st.title("Stats — Quick Tests")

df = load_cleaned_dataset()
df = apply_global_filters(df)

audience = audience_selector()

AUDIENCE_MD = {
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
}

render_audience_markdown(AUDIENCE_MD, audience=audience)

with st.expander("How to read this page", expanded=False):
    st.markdown(
        """
- **Numeric vs Target**: compares the numeric feature between employees who stayed vs left.
- **Categorical vs Target**: checks whether the category distribution differs by target.
- Treat p-values as *signals*, not final proof; sample size and data quality matter.
"""
    )

report_fig = None
report_tables: list[tuple[str, pd.DataFrame]] = []
report_settings: dict[str, object] = {}

overlay_opacity = st.slider(
    "Overlay opacity (grouped histograms)",
    min_value=0.15,
    max_value=0.95,
    value=0.65,
    step=0.05,
    help="Lower opacity makes overlapping groups easier to distinguish.",
)

# Normalize target + build human-readable labels for plotting/tables
target_col = None
target_bin_col: str | None = None
target_label_col: str | None = None
target_display_name: str | None = None

if "Attrition" in df.columns:
    target_display_name = "Attrition"
    if df["Attrition"].dtype == object:
        df = df.copy()
        df["Attrition_label"] = (
            df["Attrition"].astype(str).fillna("(missing)").str.strip().str.title()
        )
        df["Attrition_bin"] = (df["Attrition"].astype(str).str.lower() == "yes").astype(int)
        target_bin_col = "Attrition_bin"
        target_label_col = "Attrition_label"
        target_col = target_bin_col
    else:
        target_col = "Attrition"
else:
    target_col = st.selectbox("Target column", options=df.columns)
    target_display_name = str(target_col)

if target_col is not None and target_label_col is None:
    # If target is numeric binary, prefer No/Yes labels.
    try:
        vals = set(pd.to_numeric(df[target_col], errors="coerce").dropna().unique().tolist())
    except Exception:
        vals = set()
    if vals.issubset({0.0, 1.0}):
        df = df.copy()
        df["_target_label"] = (
            pd.to_numeric(df[target_col], errors="coerce")
            .map({0.0: "No", 1.0: "Yes"})
            .fillna("(missing)")
        )
        target_label_col = "_target_label"

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

    # Use the binary target if present; otherwise, fall back to the selected target.
    _t = target_bin_col or target_col

    try:
        tvals = pd.to_numeric(df[_t], errors="coerce").dropna().unique().tolist()
        tvals = sorted(set(float(v) for v in tvals))
    except Exception:
        tvals = []
    if len(tvals) != 2 or not set(tvals).issubset({0.0, 1.0}):
        st.error(
            "Numeric vs Target tests require a binary target (0/1). "
            f"Current target '{_t}' has values: {tvals[:10]}" + ("..." if len(tvals) > 10 else "")
        )
        st.stop()

    g0 = df[df[_t] == 0][col].dropna()
    g1 = df[df[_t] == 1][col].dropna()

    st.subheader("Group distributions")

    plot_df = df.copy()
    # Prefer a human-readable target label if we have one.
    if target_label_col and target_label_col in plot_df.columns:
        plot_df["_target_group"] = plot_df[target_label_col].astype(str)
    else:
        plot_df["_target_group"] = plot_df[target_col].astype(str)
    color_col = "_target_group"

    fig = px.histogram(plot_df, x=col, color=color_col, barmode="overlay", marginal="box")
    fig.update_traces(opacity=overlay_opacity)
    fig.update_layout(legend_title_text=str(target_display_name or target_col))

    show_summary_lines = st.checkbox(
        "Show mean/median lines on histogram",
        value=False,
        help="Adds reference lines to help compare distributions.",
    )
    if show_summary_lines:
        try:
            overall = pd.to_numeric(plot_df[col], errors="coerce").dropna()
            if len(overall):
                fig.add_vline(x=float(overall.mean()), line_dash="dash", line_width=3, line_color="#D62728")
                fig.add_vline(x=float(overall.median()), line_dash="dot", line_width=3, line_color="#1F77B4")
        except Exception:
            pass

    plot_state = {
        "test_kind": test_kind,
        "numeric_column": col,
        "target_display": target_display_name,
        "target_used_for_tests": _t,
        "overlay_opacity": float(overlay_opacity),
        "summary_lines": "Mean, Median" if show_summary_lines else "(none)",
        "global_filters_enabled": bool(st.session_state.get("global__enabled", True)),
        "n_rows": int(len(df)),
    }
    plot_state["short_description"] = short_plot_state_description(plot_state)

    st.plotly_chart(fig, width="stretch")
    if plot_state["short_description"]:
        st.caption(f"Plot summary: {plot_state['short_description']}")
    report_fig = fig

    report_settings = plot_state

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
        "target_col": target_display_name or target_col,
    }
    st.write(results)

    show_metrics_plot = st.checkbox(
        "Show metrics plot",
        value=True,
        help="Visualizes key numeric results (group means + significance).",
    )
    if show_metrics_plot:
        def _neglog10(pval: float | None) -> float | None:
            try:
                if pval is None:
                    return None
                p_ = float(pval)
                if not np.isfinite(p_) or p_ <= 0:
                    return None
                return float(-np.log10(p_))
            except Exception:
                return None

        mfig = metrics_bar_figure(
            {
                "mean(group=0)": results.get("mean_group0"),
                "mean(group=1)": results.get("mean_group1"),
                "-log10(p) Welch": _neglog10(results.get("welch_ttest_p")),
                "-log10(p) MW": _neglog10(results.get("mann_whitney_p")),
            },
            title="Key test metrics",
        )
        st.plotly_chart(mfig, width="stretch")

    report_tables.append(("Test results", pd.DataFrame([results])))

    st.subheader("Download")
    st.caption("Choose a format: CSV / Excel / TXT.")
    download_dataframe(pd.DataFrame([results]), file_stem="stats_numeric_results", label="Download results")

else:
    cat = st.selectbox("Categorical column", options=[c for c in cat_cols if c != target_col])

    # Prefer labeled target columns so tables/charts show No/Yes instead of 0/1.
    _t_label = target_label_col if (target_label_col and target_label_col in df.columns) else target_col
    ct = pd.crosstab(df[cat], df[_t_label], dropna=False)
    chi2, p, dof, expected = stats.chi2_contingency(ct)

    # Cramér's V
    n = ct.to_numpy().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    cramers_v = np.sqrt(phi2 / max(1, min(k - 1, r - 1)))

    st.subheader("Contingency table")
    st.dataframe(ct, width="stretch")

    report_tables.append(("Contingency table", ct.reset_index()))

    st.subheader("Download")
    st.caption("Choose a format: CSV / Excel / TXT.")
    download_dataframe(ct.reset_index(), file_stem="stats_contingency_table", label="Download contingency")

    st.subheader("Chi-square results")
    chi_results = {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "cramers_v": float(cramers_v),
        "categorical_col": cat,
        "target_col": target_display_name or target_col,
    }
    st.write(chi_results)

    show_metrics_plot = st.checkbox(
        "Show metrics plot",
        value=True,
        help="Visualizes effect size and significance summary.",
    )
    if show_metrics_plot:
        def _neglog10(pval: float | None) -> float | None:
            try:
                if pval is None:
                    return None
                p_ = float(pval)
                if not np.isfinite(p_) or p_ <= 0:
                    return None
                return float(-np.log10(p_))
            except Exception:
                return None

        mfig = metrics_bar_figure(
            {
                "Cramér's V": chi_results.get("cramers_v"),
                "-log10(p)": _neglog10(chi_results.get("p_value")),
                "chi2": chi_results.get("chi2"),
            },
            title="Chi-square summary metrics",
        )
        st.plotly_chart(mfig, width="stretch")
    st.caption("Choose a format: CSV / Excel / TXT.")
    download_dataframe(pd.DataFrame([chi_results]), file_stem="stats_chi_square_results", label="Download results")

    report_tables.append(("Chi-square results", pd.DataFrame([chi_results])))

    report_settings = {
        "test_kind": test_kind,
        "categorical_column": cat,
        "target_display": target_display_name,
        "target_used_for_tables": str(_t_label),
        "global_filters_enabled": bool(st.session_state.get("global__enabled", True)),
        "n_rows": int(len(df)),
    }
    report_settings["short_description"] = short_plot_state_description(report_settings)

    st.subheader("Bar chart")
    plot_df = ct.reset_index().melt(id_vars=cat, var_name="target", value_name="count")
    plot_df["_target_group"] = plot_df["target"].astype(str)
    fig = px.bar(plot_df, x=cat, y="count", color="_target_group", barmode="group")
    fig.update_layout(legend_title_text=str(target_display_name or target_col))
    st.plotly_chart(fig, width="stretch")
    if report_settings.get("short_description"):
        st.caption(f"Plot summary: {report_settings['short_description']}")
    report_fig = fig

st.subheader("Download page report (HTML)")
st.caption("Interactive HTML report (includes charts and result tables).")

settings_df = pd.DataFrame(
    [{"setting": str(k), "value": "" if v is None else str(v)} for k, v in (report_settings or {}).items()]
)

download_plotly_html_report(
    title="Stats — Quick Tests",
    file_stem="report_stats",
    audience=audience,
    audience_markdown=AUDIENCE_MD,
    theme_mode=theme_mode,
    figures=[("Selected test chart", report_fig)] if report_fig is not None else None,
    tables=[("Chart description / settings", settings_df)] + report_tables,
)
