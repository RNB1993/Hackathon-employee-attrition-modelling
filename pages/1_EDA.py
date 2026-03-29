from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard_utils import (
    apply_app_theme,
    apply_global_filters,
    audience_selector,
    configure_plotly_theme,
    download_dataframe,
    download_plotly_html_report,
    load_cleaned_dataset,
    render_audience_markdown,
    short_plot_state_description,
    theme_selector,
)

st.set_page_config(page_title="EDA", layout="wide")

theme_mode = theme_selector()
apply_app_theme(theme_mode)
configure_plotly_theme(theme_mode)

st.title("EDA — Univariate & Bivariate (Plotly)")

df = load_cleaned_dataset()
df = apply_global_filters(df)

audience = audience_selector()

AUDIENCE_MD = {
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
}

render_audience_markdown(AUDIENCE_MD, audience=audience)

with st.expander("How to read this page", expanded=False):
    st.markdown(
        """
- **Univariate** shows the distribution of a single column.
- **Bivariate** compares two columns; use **Color** and **Facets** to compare segments.
- If the chart becomes busy (too many groups), reduce the grouping or pick a higher-level category.
"""
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

# If the user didn't pick a hue, default to the chosen target so group distinctions are visible.
color_used = color
if color_used is None and target in df.columns:
    color_used = target

# If color is a small-cardinality numeric column (e.g., 0/1), force discrete colors.
plot_df = df
color_col = color_used
if color_col is not None and color_col in df.columns:
    try:
        nunique = int(df[color_col].nunique(dropna=False))
    except Exception:
        nunique = 999
    if (color_col in numeric_cols) and nunique <= 12:
        plot_df = df.copy()
        # Prefer human-readable labels for common binary encodings.
        try:
            vals = set(pd.to_numeric(plot_df[color_col], errors="coerce").dropna().unique().tolist())
        except Exception:
            vals = set()
        if vals.issubset({0.0, 1.0}):
            plot_df["_color_group"] = (
                pd.to_numeric(plot_df[color_col], errors="coerce")
                .map({0.0: "No", 1.0: "Yes"})
                .fillna("(missing)")
            )
        else:
            plot_df["_color_group"] = plot_df[color_col].astype("Int64").astype(str)
        color_col = "_color_group"


def _warn_many_groups(*, col_name: str | None, display_name: str, max_groups: int) -> None:
    if not col_name:
        return
    if col_name not in plot_df.columns:
        return
    try:
        n = int(plot_df[col_name].nunique(dropna=False))
    except Exception:
        return
    if n > max_groups:
        st.warning(
            f"{display_name} has {n} unique groups. This can hide distinctions or make facets unreadable. "
            f"Consider choosing a column with ≤ {max_groups} groups or remove that grouping."
        )

grouping_summary = []
if color_used:
    grouping_summary.append(f"color={color_used}")
if facet_col:
    grouping_summary.append(f"facet_col={facet_col}")
if facet_row:
    grouping_summary.append(f"facet_row={facet_row}")
st.caption(
    "Grouping applied: " + (", ".join(grouping_summary) if grouping_summary else "(none)")
)

_warn_many_groups(col_name=color_col, display_name=f"Color ({color_used})" if color_used else "Color", max_groups=12)
_warn_many_groups(col_name=facet_col, display_name=f"Facet col ({facet_col})" if facet_col else "Facet col", max_groups=8)
_warn_many_groups(col_name=facet_row, display_name=f"Facet row ({facet_row})" if facet_row else "Facet row", max_groups=8)

overlay_opacity = st.slider(
    "Overlay opacity (when color is used)",
    min_value=0.15,
    max_value=0.95,
    value=0.65,
    step=0.05,
    help="Lower opacity helps distinguish overlapping distributions when using color grouping.",
)

show_summary_lines = st.checkbox(
    "Show summary reference lines",
    value=False,
    help="Adds mean/median reference lines to help interpret distributions and scatter plots.",
)
summary_lines = []
if show_summary_lines:
    summary_lines = st.multiselect(
        "Summary lines",
        options=["Mean", "Median"],
        default=["Mean"],
        help="Shown as vertical (distribution) or vertical+horizontal (scatter) lines.",
    )


def _add_vlines(fig, s, *, title_prefix: str = "") -> None:
    try:
        ser = pd.to_numeric(s, errors="coerce").dropna()
        if ser.empty:
            return
        if "Mean" in summary_lines:
            m = float(ser.mean())
            fig.add_vline(x=m, line_width=3, line_dash="dash", line_color="#D62728")
            fig.add_annotation(x=m, y=1.02, xref="x", yref="paper", text=f"{title_prefix}mean", showarrow=False)
        if "Median" in summary_lines:
            med = float(ser.median())
            fig.add_vline(x=med, line_width=3, line_dash="dot", line_color="#1F77B4")
            fig.add_annotation(x=med, y=1.06, xref="x", yref="paper", text=f"{title_prefix}median", showarrow=False)
    except Exception:
        return


def _add_scatter_mean_lines(fig, *, x_series, y_series) -> None:
    try:
        xs = pd.to_numeric(x_series, errors="coerce").dropna()
        ys = pd.to_numeric(y_series, errors="coerce").dropna()
        if xs.empty or ys.empty:
            return
        if "Mean" in summary_lines:
            fig.add_vline(x=float(xs.mean()), line_width=2, line_dash="dash", line_color="rgba(214,39,40,0.85)")
            fig.add_hline(y=float(ys.mean()), line_width=2, line_dash="dash", line_color="rgba(214,39,40,0.85)")
        if "Median" in summary_lines:
            fig.add_vline(x=float(xs.median()), line_width=2, line_dash="dot", line_color="rgba(31,119,180,0.85)")
            fig.add_hline(y=float(ys.median()), line_width=2, line_dash="dot", line_color="rgba(31,119,180,0.85)")
    except Exception:
        return

facet_col_spacing = 0.08 if facet_col else 0.04
facet_row_spacing = 0.10 if facet_row else 0.06


def _style_facet_annotations(fig) -> None:
    try:
        is_light = (theme_mode or "Dark").strip().lower().startswith("light")
        fg = "#0B1F14" if is_light else "#E8F5E9"
        bg = "rgba(255,255,255,0.80)" if is_light else "rgba(16,42,29,0.60)"
        border = "rgba(11,31,20,0.14)" if is_light else "rgba(232,245,233,0.14)"

        def _upd(a):
            # Facet labels are annotations; style them like "facet strips".
            a.update(font=dict(size=13, color=fg), bgcolor=bg, bordercolor=border, borderwidth=1)

        fig.for_each_annotation(_upd)
    except Exception:
        return

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
                plot_df,
                x=col,
                color=color_col,
                facet_row=facet_row,
                facet_col=facet_col,
                facet_col_spacing=facet_col_spacing,
                facet_row_spacing=facet_row_spacing,
                nbins=nbins,
                barmode="overlay" if color_col else "relative",
                marginal="box",
            )
            if color_col:
                fig.update_traces(opacity=overlay_opacity)
            if show_summary_lines and summary_lines:
                _add_vlines(fig, plot_df[col])
        elif chart == "Box":
            fig = px.box(
                plot_df,
                x=color if color in categorical_cols else None,
                y=col,
                color=color_col,
                facet_row=facet_row,
                facet_col=facet_col,
                facet_col_spacing=facet_col_spacing,
                facet_row_spacing=facet_row_spacing,
                points="outliers",
            )
            if show_summary_lines and "Mean" in summary_lines:
                try:
                    fig.update_traces(boxmean=True)
                except Exception:
                    pass
        elif chart == "Violin":
            fig = px.violin(
                plot_df,
                x=color if color in categorical_cols else None,
                y=col,
                color=color_col,
                facet_row=facet_row,
                facet_col=facet_col,
                facet_col_spacing=facet_col_spacing,
                facet_row_spacing=facet_row_spacing,
                box=True,
                points="all",
            )
        else:
            counts = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="count")
            fig = px.bar(counts, x=col, y="count")

        _style_facet_annotations(fig)

        # Improve legend clarity when color grouping is active
        if color_used:
            fig.update_layout(legend_title_text=str(color_used))

        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

else:
    left, right = st.columns([1, 2])
    with left:
        x = st.selectbox("X", options=all_cols, index=all_cols.index(target) if target in all_cols else 0)
        y_candidates = [c for c in all_cols if c != x]
        y = st.selectbox("Y", options=y_candidates, index=0)
        chart = st.selectbox("Chart", options=["Scatter", "Box", "Bar", "Distribution"], index=0)
        trendline = "None"
        if chart == "Scatter" and x in numeric_cols and y in numeric_cols:
            trendline = st.selectbox("Trendline", options=["None", "OLS", "LOWESS"], index=1)

        bar_agg = "Mean"
        if chart == "Bar" and y in numeric_cols:
            bar_agg = st.selectbox(
                "Bar aggregation",
                options=["Mean", "Median", "Sum", "Count"],
                index=0,
                help="When Y is numeric, bars summarize Y by groups of X (and optional Color/Facets).",
            )

        dist_var = None
        if chart == "Distribution":
            numeric_opts = [c for c in [y, x] if c in numeric_cols]
            # Fall back to any numeric column if needed
            if not numeric_opts:
                numeric_opts = numeric_cols[:]
            if numeric_opts:
                dist_var = st.selectbox(
                    "Distribution variable",
                    options=numeric_opts,
                    index=0,
                    help="Histogram distribution plot. Use facets to split by groups (e.g., facet_col=X).",
                )
        st.caption(
            "Scatter works best for numeric-numeric; Box is great for categorical vs numeric; "
            "Bar aggregates by group (e.g., mean of Y by X). Distribution shows a histogram for a numeric variable."
        )

    with right:
        if chart == "Distribution":
            if dist_var is None or dist_var not in plot_df.columns:
                st.info("Pick a numeric distribution variable to plot.")
                fig = px.histogram(plot_df, x=y if y in numeric_cols else None)
            else:
                fig = px.histogram(
                    plot_df,
                    x=dist_var,
                    color=color_col,
                    facet_row=facet_row,
                    facet_col=facet_col,
                    facet_col_spacing=facet_col_spacing,
                    facet_row_spacing=facet_row_spacing,
                    nbins=40,
                    barmode="overlay" if color_col else "relative",
                    title=f"Distribution: {dist_var}",
                )
                if color_col:
                    fig.update_traces(opacity=overlay_opacity)
                if show_summary_lines and summary_lines:
                    _add_vlines(fig, plot_df[dist_var])
                _style_facet_annotations(fig)
                if color_used:
                    fig.update_layout(legend_title_text=str(color_used))

        elif chart == "Scatter":
            trend = None
            if trendline == "OLS":
                trend = "ols"
            elif trendline == "LOWESS":
                trend = "lowess"

            fig = px.scatter(
                plot_df,
                x=x,
                y=y,
                color=color_col,
                facet_row=facet_row,
                facet_col=facet_col,
                facet_col_spacing=facet_col_spacing,
                facet_row_spacing=facet_row_spacing,
                hover_data=[target] if target in df.columns else None,
                trendline=trend,
                trendline_scope="overall" if trend else None,
                trendline_color_override="#F1C40F",
            )
            # Style markers without affecting the trendline traces.
            fig.update_traces(
                marker=dict(size=7, opacity=0.82, line=dict(width=0)),
                selector=dict(mode="markers"),
            )

            # Make the trendline highly visible.
            if trend:
                fig.update_traces(
                    line=dict(width=4),
                    selector=dict(mode="lines"),
                )
            if show_summary_lines and summary_lines:
                _add_scatter_mean_lines(fig, x_series=plot_df[x], y_series=plot_df[y])
        elif chart == "Box":
            fig = px.box(
                plot_df,
                x=x,
                y=y,
                color=color_col,
                facet_row=facet_row,
                facet_col=facet_col,
                facet_col_spacing=facet_col_spacing,
                facet_row_spacing=facet_row_spacing,
                points="outliers",
            )
        else:
            # Bar:
            # - If Y is numeric: aggregate mean(Y) by X and (optionally) color + facets.
            # - If Y is categorical: count rows by X and Y (and facets).
            if y in numeric_cols:
                group_cols = [c for c in [facet_row, facet_col, x, color_col] if c]
                group_cols = list(dict.fromkeys(group_cols))  # preserve order, drop duplicates

                agg_name = str(bar_agg or "Mean").strip().lower()
                if agg_name == "count":
                    agg = plot_df.groupby(group_cols, dropna=False).size().reset_index(name="count")
                    y_col = "count"
                else:
                    func_map = {
                        "mean": "mean",
                        "median": "median",
                        "sum": "sum",
                    }
                    func = func_map.get(agg_name, "mean")
                    y_col = f"{func}_{y}"
                    agg = (
                        plot_df.groupby(group_cols, dropna=False)[y]
                        .agg(func)
                        .reset_index(name=y_col)
                    )

                fig = px.bar(
                    agg,
                    x=x,
                    y=y_col,
                    color=color_col,
                    facet_row=facet_row,
                    facet_col=facet_col,
                    facet_col_spacing=facet_col_spacing,
                    facet_row_spacing=facet_row_spacing,
                    barmode="group" if color_col else "relative",
                )
            else:
                group_cols = [c for c in [facet_row, facet_col, x, y] if c]
                group_cols = list(dict.fromkeys(group_cols))

                ct = plot_df.groupby(group_cols, dropna=False).size().reset_index(name="count")
                fig = px.bar(
                    ct,
                    x=x,
                    y="count",
                    color=y,
                    facet_row=facet_row,
                    facet_col=facet_col,
                    facet_col_spacing=facet_col_spacing,
                    facet_row_spacing=facet_row_spacing,
                    barmode="group",
                )

        _style_facet_annotations(fig)

        if color_used:
            fig.update_layout(legend_title_text=str(color_used))

        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Download page report (HTML)")
st.caption("Interactive HTML report (includes the selected chart and key tables).")

chart_settings = {
    "plot_kind": plot_kind,
    "chart": chart,
    "x": (col if plot_kind == "Univariate" else x),
    "y": (None if plot_kind == "Univariate" else y),
    "color": color_used,
    "facet_col": facet_col,
    "facet_row": facet_row,
    "bar_aggregation": (bar_agg if (plot_kind == "Bivariate" and chart == "Bar" and y in numeric_cols) else None),
    "trendline": (trendline if (plot_kind == "Bivariate" and chart == "Scatter") else None),
    "summary_lines": ", ".join(summary_lines) if summary_lines else "(none)",
    "global_filters_enabled": bool(st.session_state.get("global__enabled", True)),
    "n_rows": int(len(df)),
}

chart_settings["short_description"] = short_plot_state_description(chart_settings)
if chart_settings["short_description"]:
    st.caption(f"Plot summary: {chart_settings['short_description']}")

chart_settings_df = pd.DataFrame(
    [{"setting": k, "value": "" if v is None else str(v)} for k, v in chart_settings.items()]
)

download_plotly_html_report(
    title="EDA — Univariate & Bivariate (Plotly)",
    file_stem="report_eda",
    audience=audience,
    audience_markdown=AUDIENCE_MD,
    theme_mode=theme_mode,
    figures=[("Selected chart", fig)],
    tables=[
        ("Chart description / settings", chart_settings_df),
        ("Dataset preview (first 50 rows)", df.head(50)),
    ],
)

st.subheader("Dataset preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Download dataset")
st.caption("Choose a format: CSV / Excel / TXT.")
download_dataframe(df, file_stem="hr_attrition_cleaned", label="Download dataset")
