from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard_utils import (
    MODEL_CANDIDATES,
    apply_app_theme,
    apply_global_filters,
    audience_selector,
    configure_plotly_theme,
    download_dataframe,
    download_plotly_html_report,
    interaction_importance_table,
    load_cleaned_dataset,
    metrics_bar_figure,
    render_audience_markdown,
    theme_selector,
)

st.set_page_config(page_title="Executive Summary", layout="wide")

theme_mode = theme_selector()
apply_app_theme(theme_mode)
configure_plotly_theme(theme_mode)

st.title("Executive Summary")

base_df_full = load_cleaned_dataset()
base_df = apply_global_filters(base_df_full)

audience = audience_selector()

AUDIENCE_MD = {
    "Non-technical": """
This page gives a quick snapshot of **headcount**, **attrition rate**, and the strongest patterns in the data.

Use the sidebar to apply global filters (e.g., Department, JobRole) and watch the KPIs and charts update.
""",
    "Semi-technical": """
High-level KPI view for the cleaned IBM HR dataset.

Includes segment-level attrition rates and (optionally) the top interaction features from the trained model.
""",
    "Technical": """
Executive snapshot using the cleaned dataset + optional model coefficient inspection.

Global filters are applied consistently across pages via `apply_global_filters()`.
""",
}

render_audience_markdown(AUDIENCE_MD, audience=audience)


def _attrition_mask(df: pd.DataFrame) -> pd.Series:
    if "Attrition" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    s = df["Attrition"].astype(str).str.strip().str.lower()
    return s.isin({"yes", "1", "true", "y"})


def _kpi_block(df: pd.DataFrame) -> dict[str, float | int | None]:
    if df is None or df.empty:
        return {
            "n": 0,
            "n_attr": 0,
            "attr_rate": None,
            "avg_age": None,
            "avg_income": None,
            "avg_years": None,
        }

    m = _attrition_mask(df)
    n = int(len(df))
    n_attr = int(m.sum())
    attr_rate = float(n_attr / n) if n else None

    def _mean(col: str):
        if col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.mean()) if len(s) else None

    return {
        "n": n,
        "n_attr": n_attr,
        "attr_rate": attr_rate,
        "avg_age": _mean("Age"),
        "avg_income": _mean("MonthlyIncome"),
        "avg_years": _mean("YearsAtCompany"),
    }


kpi_all = _kpi_block(base_df_full)
kpi = _kpi_block(base_df)

with st.expander("Key KPIs", expanded=True):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Employees", f"{kpi['n']:,}", delta=f"All: {kpi_all['n']:,}")

    with c2:
        rate = kpi["attr_rate"]
        rate_all = kpi_all["attr_rate"]
        st.metric(
            "Attrition rate",
            f"{rate:.1%}" if rate is not None else "—",
            delta=(f"All: {rate_all:.1%}" if rate_all is not None else None),
        )

    with c3:
        st.metric(
            "Avg age",
            f"{kpi['avg_age']:.1f}" if kpi["avg_age"] is not None else "—",
            delta=(f"All: {kpi_all['avg_age']:.1f}" if kpi_all["avg_age"] is not None else None),
        )

    with c4:
        st.metric(
            "Avg monthly income",
            (f"${kpi['avg_income']:,.0f}" if kpi["avg_income"] is not None else "—"),
            delta=(
                f"All: ${kpi_all['avg_income']:,.0f}" if kpi_all["avg_income"] is not None else None
            ),
        )

    c5, c6 = st.columns(2)
    with c5:
        st.metric(
            "Attritions (count)",
            f"{kpi['n_attr']:,}",
            delta=f"All: {kpi_all['n_attr']:,}",
        )
    with c6:
        st.metric(
            "Avg years at company",
            f"{kpi['avg_years']:.1f}" if kpi["avg_years"] is not None else "—",
            delta=(
                f"All: {kpi_all['avg_years']:.1f}" if kpi_all["avg_years"] is not None else None
            ),
        )

st.divider()

st.subheader("Attrition by segment")

seg_c1, seg_c2 = st.columns(2)

with seg_c1:
    col = "Department" if "Department" in base_df.columns else None
    if col:
        tmp = base_df.copy()
        tmp["_is_attr"] = _attrition_mask(tmp).astype(int)
        grp = tmp.groupby(col, dropna=False)["_is_attr"].agg(["mean", "count"]).reset_index()
        grp = grp.sort_values("mean", ascending=False)
        fig = px.bar(
            grp,
            x=col,
            y="mean",
            text="count",
            title="Attrition rate by Department",
        )
        fig.update_layout(yaxis_tickformat=",.0%", height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No Department column found in dataset.")

with seg_c2:
    col = "JobRole" if "JobRole" in base_df.columns else None
    if col:
        tmp = base_df.copy()
        tmp["_is_attr"] = _attrition_mask(tmp).astype(int)
        grp = tmp.groupby(col, dropna=False)["_is_attr"].agg(["mean", "count"]).reset_index()
        grp = grp.sort_values(["mean", "count"], ascending=[False, False]).head(15)
        fig = px.bar(
            grp,
            x=col,
            y="mean",
            text="count",
            title="Attrition rate by JobRole (top 15)",
        )
        fig.update_layout(yaxis_tickformat=",.0%", height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No JobRole column found in dataset.")

st.subheader("Key distributions")

d1, d2 = st.columns(2)

with d1:
    if "MonthlyIncome" in base_df.columns and "Attrition" in base_df.columns:
        fig = px.violin(
            base_df,
            x="Attrition",
            y="MonthlyIncome",
            box=True,
            points="outliers",
            title="MonthlyIncome distribution (by Attrition)",
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    elif "MonthlyIncome" in base_df.columns:
        fig = px.histogram(base_df, x="MonthlyIncome", nbins=40, title="MonthlyIncome distribution")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No MonthlyIncome column found.")

with d2:
    if "OverTime" in base_df.columns and "Attrition" in base_df.columns:
        tmp = base_df.copy()
        tmp["_is_attr"] = _attrition_mask(tmp).astype(int)
        grp = tmp.groupby("OverTime", dropna=False)["_is_attr"].agg(["mean", "count"]).reset_index()
        grp = grp.sort_values("mean", ascending=False)
        fig = px.bar(grp, x="OverTime", y="mean", text="count", title="Attrition rate by OverTime")
        fig.update_layout(yaxis_tickformat=",.0%", height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        # Simple fallback: show Age histogram
        if "Age" in base_df.columns:
            fig = px.histogram(base_df, x="Age", nbins=30, title="Age distribution")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No OverTime/Age columns found.")

st.divider()

st.subheader("Model insight (optional)")

with st.sidebar:
    st.markdown("### Summary settings")
    show_model = st.toggle(
        "Show model interaction importance",
        value=True,
        help="Uses the trained pipeline coefficients to rank engineered interaction features.",
    )
    model_label = st.selectbox("Model", options=list(MODEL_CANDIDATES.keys()), index=0)
    top_n = st.slider("Top interactions", 10, 50, 20, 5)

importance_df = None
if show_model:
    try:
        importance_df = interaction_importance_table(model_label, dataset_for_mapping=base_df_full, top_n=int(top_n))
        st.caption("Top engineered interactions by absolute coefficient (log-odds).")
        st.dataframe(importance_df, width="stretch")

        # Small plot summary
        plot_df = importance_df.head(12).copy()
        plot_df["label"] = plot_df["interaction"].astype(str) + " (" + plot_df["direction"].astype(str) + ")"
        fig = px.bar(
            plot_df[::-1],
            x="abs_coef",
            y="label",
            orientation="h",
            title="Top interaction strength (abs coef)",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.info(f"Model insight not available for this pipeline: {e}")

st.subheader("Downloads")

with st.expander("Download filtered dataset", expanded=False):
    st.caption("Exports the current filtered dataset (raw cleaned columns).")
    download_dataframe(base_df, file_stem="hr_attrition_filtered", label="Download data")

with st.expander("Download executive report (HTML)", expanded=False):
    st.caption("Includes key charts and KPI tables as a shareable HTML report.")

    # Rebuild simple KPI table
    kpi_table = pd.DataFrame(
        [
            {"metric": "employees", "value": kpi["n"]},
            {"metric": "attritions", "value": kpi["n_attr"]},
            {"metric": "attrition_rate", "value": kpi["attr_rate"]},
            {"metric": "avg_age", "value": kpi["avg_age"]},
            {"metric": "avg_monthly_income", "value": kpi["avg_income"]},
            {"metric": "avg_years_at_company", "value": kpi["avg_years"]},
        ]
    )

    figs: list[tuple[str, object]] = []
    # Recreate one or two stable figures for reporting
    try:
        if "OverTime" in base_df.columns and "Attrition" in base_df.columns:
            tmp = base_df.copy()
            tmp["_is_attr"] = _attrition_mask(tmp).astype(int)
            grp = tmp.groupby("OverTime", dropna=False)["_is_attr"].agg(["mean", "count"]).reset_index()
            rep_fig = px.bar(grp, x="OverTime", y="mean", text="count", title="Attrition rate by OverTime")
            rep_fig.update_layout(yaxis_tickformat=",.0%")
            figs.append(("Attrition rate by OverTime", rep_fig))
    except Exception:
        pass

    try:
        if "Department" in base_df.columns:
            tmp = base_df.copy()
            tmp["_is_attr"] = _attrition_mask(tmp).astype(int)
            grp = tmp.groupby("Department", dropna=False)["_is_attr"].agg(["mean", "count"]).reset_index()
            grp = grp.sort_values("mean", ascending=False)
            rep_fig = px.bar(grp, x="Department", y="mean", text="count", title="Attrition rate by Department")
            rep_fig.update_layout(yaxis_tickformat=",.0%")
            figs.append(("Attrition rate by Department", rep_fig))
    except Exception:
        pass

    download_plotly_html_report(
        title="Executive Summary — IBM HR Attrition",
        file_stem="report_executive_summary",
        audience=audience,
        audience_markdown=AUDIENCE_MD,
        theme_mode=theme_mode,
        figures=figs if figs else None,
        tables=[
            ("Key KPIs", kpi_table),
            ("Filtered data (first 200 rows)", base_df.head(200).reset_index(drop=True)),
        ],
    )
