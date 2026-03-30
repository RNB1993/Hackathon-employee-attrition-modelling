from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_utils import (
    MODEL_CANDIDATES,
    apply_app_theme,
    apply_global_filters,
    audience_selector,
    configure_plotly_theme,
    download_dataframe,
    download_plotly_html_report,
    engineer_interactions,
    ensure_required_columns,
    interaction_mapping,
    load_cleaned_dataset,
    load_pipeline,
    metrics_bar_figure,
    predict_proba_attrition,
    probability_indicator_figure,
    render_audience_markdown,
    short_plot_state_description,
    theme_selector,
)

st.set_page_config(page_title="Prediction", layout="wide")

theme_mode = theme_selector()
apply_app_theme(theme_mode)
configure_plotly_theme(theme_mode)

st.title("Prediction — Attrition Probability")

base_df_full = load_cleaned_dataset()
base_df = apply_global_filters(base_df_full)

if base_df is None or base_df.empty:
    st.warning(
        "No rows match the current global filters. "
        "Reset or broaden filters in the sidebar to use Prediction."
    )
    st.stop()

audience = audience_selector()

with st.sidebar:
    st.header("Model")
    model_label = st.selectbox("Choose model", list(MODEL_CANDIDATES.keys()))
    try:
        st.caption(f"Model file: {MODEL_CANDIDATES[model_label].name}")
    except Exception:
        pass

AUDIENCE_MD = {
    "Non-technical": """
This page estimates **attrition risk** as a probability.

- Higher probability means the profile is more similar to employees who left in the historical data.
- Use this to *triage and explore*, not as a final decision.
""",
    "Semi-technical": """
Predict attrition probability using the saved model pipeline.

You can:
- start from an existing employee record, then **key in edits**
- upload a CSV to score many employees
""",
    "Technical": """
Model scoring page (sklearn Pipeline). Features are engineered interaction terms derived from Spearman correlation pairs.

Includes scenario analysis (one-feature sensitivity) using the same feature engineering pipeline.
""",
}

render_audience_markdown(AUDIENCE_MD, audience=audience)

with st.expander("How to read this page", expanded=False):
    st.markdown(
        """
- The model outputs a **probability** (0–100%) of attrition risk.
- Use **group scoring** to compare segments; use **single employee** to explore scenarios.
- This is a decision-support tool; always combine with HR context and fairness checks.
"""
    )

overlay_opacity = st.sidebar.slider(
    "Overlay opacity (grouped histograms)",
    min_value=0.15,
    max_value=0.95,
    value=0.65,
    step=0.05,
    help="Lower opacity makes overlapping groups easier to distinguish.",
)

threshold = st.sidebar.slider(
    "Classification threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Controls how the predicted probability is turned into a 0/1 label. Does not change the probability itself.",
)


def _read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV defensively (encoding + delimiter inference)."""

    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if not raw:
        raise ValueError("Uploaded file is empty")

    # Basic guardrail to avoid accidental huge uploads.
    max_bytes = 8 * 1024 * 1024
    if len(raw) > max_bytes:
        raise ValueError(f"CSV is too large ({len(raw) / (1024 * 1024):.1f} MB). Please upload ≤ 8 MB.")

    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                BytesIO(raw),
                encoding=enc,
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as e:
            last_err = e

    raise ValueError(f"Could not parse CSV. Last error: {last_err}")


@st.cache_resource(show_spinner=False)
def _get_pipe(model_label: str):
    return load_pipeline(model_label)


def _predict_proba_from_raw(raw_df: pd.DataFrame, *, pipe, mapping: pd.DataFrame) -> np.ndarray:
    X = engineer_interactions(raw_df, mapping=mapping)
    if "Attrition" in X.columns:
        X = X.drop(columns=["Attrition"])
    return pipe.predict_proba(X)[:, 1]


def _plot_feature_context(base: pd.DataFrame, feature: str, value) -> go.Figure | None:
    if feature not in base.columns:
        st.info(f"No plot available: missing column {feature}")
        return None

    s = base[feature]
    if pd.api.types.is_numeric_dtype(s):
        # Default to grouping by Attrition when available so the distribution shows distinctions.
        color_col = "Attrition" if "Attrition" in base.columns else None
        fig = px.histogram(
            base,
            x=feature,
            color=color_col,
            nbins=30,
            barmode="overlay" if color_col else "relative",
            title=f"{feature} distribution" + (" (grouped by Attrition)" if color_col else ""),
        )
        if color_col:
            fig.update_traces(opacity=overlay_opacity)
            fig.update_layout(legend_title_text=color_col)
        try:
            fig.add_vline(x=float(value), line_width=3, line_dash="dash", line_color="red")
        except Exception:
            pass
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

        state = {
            "chart": "Histogram",
            "x": feature,
            "color": color_col,
            "global_filters_enabled": bool(st.session_state.get("global__enabled", True)),
            "n_rows": int(len(base)),
        }
        state["short_description"] = short_plot_state_description(state)
        if state["short_description"]:
            st.caption(f"Plot summary: {state['short_description']}")

        try:
            pct = float((s.dropna() < float(value)).mean())
            st.caption(f"Entered value percentile (approx): {pct:.0%}")
        except Exception:
            pass
        return fig
    else:
        counts = s.astype(str).fillna("(missing)").value_counts().reset_index()
        counts.columns = [feature, "count"]
        counts["is_selected"] = counts[feature].astype(str) == str(value)
        fig = px.bar(
            counts,
            x=feature,
            y="count",
            color="is_selected",
            color_discrete_map={True: "#EF553B", False: "#636EFA"},
            title=f"{feature} counts",
        )
        fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

        state = {
            "chart": "Bar",
            "x": feature,
            "y": "count",
            "color": "is_selected",
            "global_filters_enabled": bool(st.session_state.get("global__enabled", True)),
            "n_rows": int(len(base)),
        }
        state["short_description"] = short_plot_state_description(state)
        if state["short_description"]:
            st.caption(f"Plot summary: {state['short_description']}")
        return fig


pipe = _get_pipe(model_label)
# IMPORTANT: keep mapping stable by deriving it from the full dataset,
# not a filtered view (filters are for exploration, not redefining features).
mapping = interaction_mapping(base_df_full)
required_raw_numeric = sorted(
    set(mapping["raw_feature_1"].tolist() + mapping["raw_feature_2"].tolist())
)

# For raw uploads, require all the cleaned dataset columns (except target).
required_raw_cols = [c for c in base_df_full.columns if c != "Attrition"]

# If user uploads already-engineered interactions, raw numeric columns are optional.
required_non_numeric = [c for c in required_raw_cols if c not in required_raw_numeric]

st.subheader("Option A — Select an employee (or group) by features")
st.caption(
    "Instead of picking a row index, filter by employee attributes. You can score a whole group, "
    "then optionally pick one employee for detailed what-if analysis."
)

# --- Build a lightweight filtering UI ---
filterable_cat_cols = [
    c
    for c in [
        "Department",
        "JobRole",
        "JobLevel",
        "BusinessTravel",
        "OverTime",
        "Gender",
        "MaritalStatus",
        "EducationField",
    ]
    if c in base_df.columns
]

filterable_num_cols = [
    c
    for c in [
        "Age",
        "MonthlyIncome",
        "DistanceFromHome",
        "YearsAtCompany",
        "TotalWorkingYears",
    ]
    if c in base_df.columns
]

with st.expander("Filter employees", expanded=True):
    cat_filters: dict[str, list[str]] = {}
    if filterable_cat_cols:
        st.markdown("**Categorical filters**")
        for col in filterable_cat_cols:
            opts = sorted(base_df[col].astype(str).fillna("(missing)").unique().tolist())
            picked = st.multiselect(f"{col}", options=opts, default=[])
            if picked:
                cat_filters[col] = picked

    num_filters: dict[str, tuple[float, float]] = {}
    if filterable_num_cols:
        st.markdown("**Numeric filters**")
        for col in filterable_num_cols:
            s = pd.to_numeric(base_df[col], errors="coerce").dropna()
            if s.empty:
                continue
            lo = float(np.percentile(s, 1))
            hi = float(np.percentile(s, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                continue
            picked = st.slider(col, min_value=lo, max_value=hi, value=(lo, hi))
            # Only apply if user narrowed the range
            if picked[0] > lo or picked[1] < hi:
                num_filters[col] = (float(picked[0]), float(picked[1]))

filtered = base_df.copy()
for col, picked in cat_filters.items():
    filtered = filtered[filtered[col].astype(str).fillna("(missing)").isin(picked)]
for col, (lo, hi) in num_filters.items():
    s = pd.to_numeric(filtered[col], errors="coerce")
    filtered = filtered[(s >= lo) & (s <= hi)]

st.write(f"Filtered employees: **{len(filtered):,}** / {len(base_df):,}")

if len(filtered) == 0:
    st.warning("No employees match the current filters. Widen filters to continue.")
    st.stop()

st.dataframe(filtered.head(30), width="stretch")

st.markdown("### Score the filtered group (optional)")
score_group = st.button("Score filtered group", type="secondary")
group_out = None
if score_group:
    try:
        group_out = predict_proba_attrition(
            model_label,
            filtered,
            dataset_for_mapping=base_df_full,
            threshold=float(threshold),
        )
        st.success("Group predictions computed.")

        group_color = "Attrition" if "Attrition" in group_out.columns else None
        if group_color:
            st.caption("Grouped by: Attrition")

        fig = px.histogram(
            group_out,
            x="pred_attrition_proba",
            color=group_color,
            nbins=30,
            barmode="overlay" if group_color else "relative",
            title="Predicted attrition probability — filtered group",
        )
        if group_color:
            fig.update_traces(opacity=overlay_opacity)
            fig.update_layout(legend_title_text=group_color)
        fig.update_layout(
            xaxis_tickformat=",.0%",
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        show_summary_lines = st.checkbox(
            "Show mean/median lines on histogram (group)",
            value=False,
            help="Adds reference lines to the probability distribution.",
        )
        if show_summary_lines:
            try:
                s = pd.to_numeric(group_out["pred_attrition_proba"], errors="coerce").dropna()
                if len(s):
                    fig.add_vline(x=float(s.mean()), line_dash="dash", line_width=3, line_color="#D62728")
                    fig.add_vline(x=float(s.median()), line_dash="dot", line_width=3, line_color="#1F77B4")
                    st.caption(f"Mean: {float(s.mean()):.1%} | Median: {float(s.median()):.1%}")
            except Exception:
                pass

        st.plotly_chart(fig, width="stretch")

        group_state = {
            "chart": "Histogram",
            "x": "pred_attrition_proba",
            "color": group_color,
            "title": "Predicted attrition probability — filtered group",
            "global_filters_enabled": bool(st.session_state.get("global__enabled", True)),
            "n_rows": int(len(group_out)),
        }
        group_state["short_description"] = short_plot_state_description(group_state)
        if group_state["short_description"]:
            st.caption(f"Plot summary: {group_state['short_description']}")

        show_metrics_plot = st.checkbox(
            "Show metrics plot (group)",
            value=True,
            help="Plots summary statistics for the scored group.",
        )
        if show_metrics_plot:
            thr = float(threshold)
            try:
                s = pd.to_numeric(group_out["pred_attrition_proba"], errors="coerce").dropna()
                metrics = {
                    "mean proba": float(s.mean()) if len(s) else None,
                    "median proba": float(s.median()) if len(s) else None,
                    "p90 proba": float(s.quantile(0.90)) if len(s) else None,
                    f"share ≥ {thr:.0%}": float((s >= thr).mean()) if len(s) else None,
                }
                mfig = metrics_bar_figure(metrics, title="Group scoring summary")
                mfig.update_layout(yaxis_tickformat=",.0%")
                st.plotly_chart(mfig, width="stretch")
            except Exception:
                pass

        st.subheader("Download group predictions")
        st.caption("Choose a format: CSV / Excel / TXT.")
        download_dataframe(group_out, file_stem="filtered_group_predictions", label="Download predictions")

        st.subheader("Download group report (HTML)")
        st.caption("Interactive HTML report (includes charts and key tables).")
        summary = (
            group_out[["pred_attrition_proba"]]
            .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            .reset_index()
            .rename(columns={"index": "metric"})
        )

        settings_df = pd.DataFrame(
            [
                {"setting": "model", "value": model_label},
                {"setting": "rows_scored", "value": str(len(group_out))},
                {"setting": "global_filters_enabled", "value": str(bool(st.session_state.get("global__enabled", True)))},
                {"setting": "threshold", "value": f"{float(threshold):.2f}"},
                {"setting": "short_description", "value": group_state.get("short_description", "")},
            ]
        )
        download_plotly_html_report(
            title=f"Prediction — Group Scoring ({model_label})",
            file_stem="report_prediction_group",
            audience=audience,
            audience_markdown=AUDIENCE_MD,
            theme_mode=theme_mode,
            figures=[("Group probability distribution", fig)],
            tables=[
                ("Chart description / settings", settings_df),
                ("Summary stats", summary),
                ("Predictions (first 200 rows)", group_out.head(200)),
            ],
        )
    except Exception as e:
        st.error(f"Group prediction failed: {e}")

st.markdown("### Pick one employee for detailed analysis")
id_col = "EmployeeNumber" if "EmployeeNumber" in filtered.columns else None
if id_col:
    labels = (
        filtered[[id_col] + [c for c in ["JobRole", "Department", "Age"] if c in filtered.columns]]
        .astype(str)
        .fillna("")
        .apply(lambda r: " | ".join([f"{id_col}={r[id_col]}"] + [f"{c}={r[c]}" for c in r.index if c != id_col and r[c] != ""]), axis=1)
    )
    selected_label = st.selectbox("Employee", options=labels.tolist(), index=0)
    selected_idx = labels[labels == selected_label].index[0]
    row_base = filtered.loc[[selected_idx]].copy()
else:
    # Fall back to filtered position index
    pos = st.number_input("Employee position in filtered table", 0, max(0, len(filtered) - 1), 0, 1)
    row_base = filtered.iloc[[int(pos)]].copy()

# If the base employee changes, drop previously-entered overrides to avoid confusion.
row_key = str(row_base.index[0])
if st.session_state.get("row_overrides_row_key") != row_key:
    st.session_state.pop("row_overrides", None)
    st.session_state["row_overrides_row_key"] = row_key

st.write("Selected row (raw):")
st.dataframe(row_base, width="stretch")

st.markdown("### Key in edits (override features)")
editable_cols = [c for c in base_df.columns if c not in {"Attrition"}]
numeric_cols = base_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in editable_cols if c not in numeric_cols]

override_cols = st.multiselect(
    "Which features do you want to override?",
    options=editable_cols,
    default=[],
)

row = row_base.copy()
if override_cols:
    with st.form("override_form"):
        st.caption("Enter new values for the selected features, then click Apply.")
        overrides: dict[str, object] = {}
        for col in override_cols:
            if col in numeric_cols:
                base_val = row_base.iloc[0][col]
                default_val = 0.0 if pd.isna(base_val) else float(base_val)
                overrides[col] = st.number_input(col, value=default_val)
            else:
                opts = sorted(base_df[col].astype(str).fillna("(missing)").unique().tolist())
                current = str(row_base.iloc[0][col]) if not pd.isna(row_base.iloc[0][col]) else "(missing)"
                idx = opts.index(current) if current in opts else 0
                overrides[col] = st.selectbox(col, options=opts, index=idx)

        applied = st.form_submit_button("Apply overrides")
        if applied:
            for col, val in overrides.items():
                if col in numeric_cols:
                    row[col] = float(val)
                else:
                    # Keep as string to match training preprocessing
                    row[col] = str(val)
            st.session_state["row_overrides"] = {c: row.iloc[0][c] for c in override_cols}

    # Re-apply persisted overrides when navigating widgets / reruns
    if "row_overrides" in st.session_state:
        for col, val in st.session_state["row_overrides"].items():
            if col in row.columns:
                row[col] = val

if override_cols:
    st.write("Row after overrides:")
    st.dataframe(row, width="stretch")

try:
    pred_row = predict_proba_attrition(
        model_label,
        row,
        dataset_for_mapping=base_df_full,
        threshold=float(threshold),
    )
    st.success("Prediction computed.")
    proba = float(pred_row["pred_attrition_proba"].iloc[0])
    st.metric("Predicted attrition probability", f"{proba:.1%}")

    st.caption(
        f"Label threshold: {float(threshold):.0%} (predicted label = 1 when probability ≥ threshold)."
    )

    show_metrics_plot = st.checkbox(
        "Show probability gauge",
        value=True,
        help="Shows the predicted probability as a gauge with a 50% threshold.",
    )
    if show_metrics_plot:
        gfig = probability_indicator_figure(
            proba,
            title="Predicted attrition probability",
            threshold=float(threshold),
        )
        st.plotly_chart(gfig, width="stretch")
        gauge_state = {
            "chart": "Gauge",
            "title": "Predicted attrition probability",
            "threshold": float(threshold),
            "predicted_probability": float(proba),
            "global_filters_enabled": bool(st.session_state.get("global__enabled", True)),
            "n_rows": 1,
        }
        gauge_state["short_description"] = short_plot_state_description(gauge_state)
        if gauge_state["short_description"]:
            st.caption(f"Plot summary: {gauge_state['short_description']}")
    st.dataframe(pred_row[["pred_attrition_proba", "pred_attrition_label"]], width="stretch")

    st.subheader("Download prediction")
    st.caption("Choose a format: CSV / Excel / TXT.")
    download_dataframe(pred_row, file_stem="single_employee_prediction", label="Download prediction")

    st.subheader("Explain this prediction (based on keyed-in features)")
    default_features = override_cols[:]
    features_to_plot = st.multiselect(
        "Choose features to visualize",
        options=editable_cols,
        default=default_features,
        help="If you overrode features above, those are suggested here.",
    )

    context_figs: list[tuple[str, object]] = []

    if not features_to_plot:
        st.info("Select one or more features to see input-dependent plots.")
    else:
        for feat in features_to_plot:
            st.markdown(f"#### {feat}")
            fig_ctx = _plot_feature_context(base_df, feat, row.iloc[0][feat])
            if fig_ctx is not None:
                context_figs.append((f"Feature context — {feat}", fig_ctx))

    st.subheader("What-if sensitivity (change one feature)")
    sens_numeric = [c for c in features_to_plot if c in numeric_cols]
    sens_categorical = [c for c in features_to_plot if c in categorical_cols]
    sens_kind = st.radio(
        "Sensitivity type",
        ["Numeric feature", "Categorical feature"],
        horizontal=True,
        index=0,
        help="Plots how the predicted probability changes when one feature changes and all others are held constant.",
    )

    sens_fig = None

    if sens_kind == "Numeric feature":
        if not sens_numeric:
            st.info("Pick at least one numeric feature in the visualization selector above.")
        else:
            f = st.selectbox("Numeric feature", options=sens_numeric)
            n_points = st.slider("Points", 10, 80, 25, 5)
            lo = float(base_df[f].quantile(0.05))
            hi = float(base_df[f].quantile(0.95))
            grid = np.linspace(lo, hi, n_points)

            scenarios = pd.concat([row] * n_points, ignore_index=True)
            scenarios[f] = grid
            probs = _predict_proba_from_raw(scenarios, pipe=pipe, mapping=mapping)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=grid, y=probs, mode="lines", name="Predicted proba"))
            try:
                fig.add_vline(x=float(row.iloc[0][f]), line_dash="dash", line_color="red")
            except Exception:
                pass
            fig.update_layout(
                xaxis_title=f,
                yaxis_title="Attrition probability",
                yaxis_tickformat=",.0%",
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, width="stretch")
            sens_state = {
                "chart": "Line",
                "title": "What-if sensitivity (numeric feature)",
                "x": f,
                "y": "pred_attrition_proba",
                "global_filters_enabled": bool(st.session_state.get("global__enabled", True)),
                "n_rows": int(n_points),
            }
            sens_state["short_description"] = short_plot_state_description(sens_state)
            if sens_state["short_description"]:
                st.caption(f"Plot summary: {sens_state['short_description']}")
            sens_fig = fig
    else:
        if not sens_categorical:
            st.info("Pick at least one categorical feature in the visualization selector above.")
        else:
            f = st.selectbox("Categorical feature", options=sens_categorical)
            cats = sorted(base_df[f].astype(str).fillna("(missing)").unique().tolist())
            scenarios = pd.concat([row] * len(cats), ignore_index=True)
            scenarios[f] = cats
            probs = _predict_proba_from_raw(scenarios, pipe=pipe, mapping=mapping)
            plot_df = pd.DataFrame({f: cats, "pred_attrition_proba": probs})
            fig = px.bar(plot_df, x=f, y="pred_attrition_proba")
            fig.update_layout(yaxis_tickformat=",.0%", height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            sens_state = {
                "chart": "Bar",
                "title": "What-if sensitivity (categorical feature)",
                "x": f,
                "y": "pred_attrition_proba",
                "global_filters_enabled": bool(st.session_state.get("global__enabled", True)),
                "n_rows": int(len(cats)),
            }
            sens_state["short_description"] = short_plot_state_description(sens_state)
            if sens_state["short_description"]:
                st.caption(f"Plot summary: {sens_state['short_description']}")
            sens_fig = fig

    st.subheader("Download prediction report (HTML)")
    st.caption("Interactive HTML report (includes plots and input/output tables).")
    report_figs: list[tuple[str, object]] = []
    report_figs.extend(context_figs)
    if sens_fig is not None:
        report_figs.append(("What-if sensitivity", sens_fig))

    settings_df = pd.DataFrame(
        [
            {"setting": "model", "value": model_label},
            {"setting": "predicted_probability", "value": f"{proba:.4f}"},
            {"setting": "threshold", "value": f"{float(threshold):.2f}"},
            {"setting": "global_filters_enabled", "value": str(bool(st.session_state.get("global__enabled", True)))},
            {
                "setting": "short_description",
                "value": f"Single employee predicted probability: {proba:.1%} (threshold {float(threshold):.0%}).",
            },
        ]
    )

    download_plotly_html_report(
        title=f"Prediction — Single Employee ({model_label})",
        file_stem="report_prediction_single",
        audience=audience,
        audience_markdown=AUDIENCE_MD,
        theme_mode=theme_mode,
        figures=report_figs if report_figs else None,
        tables=[
            ("Chart description / settings", settings_df),
            ("Input row (raw)", row_base.reset_index(drop=True)),
            ("Input row (after overrides)", row.reset_index(drop=True)),
            ("Prediction output", pred_row.reset_index(drop=True)),
        ],
    )
except Exception as e:
    st.error(f"Prediction failed: {e}")

st.divider()

st.subheader("Option B — Upload CSV (raw feature columns)")

with st.expander("Download a scoring template", expanded=False):
    st.caption(
        "Download a CSV template with the cleaned dataset columns expected by the scoring pipeline. "
        "You can fill in your own employee rows and re-upload."
    )
    template = base_df_full[required_raw_cols].head(50).copy()
    download_dataframe(template, file_stem="scoring_template", label="Download template")

uploaded = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded is not None:
    try:
        up = _read_uploaded_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()

    st.write("Uploaded preview:")
    st.dataframe(up.head(20), width="stretch")

    already_engineered = any(
        c.startswith("inter_pos_") or c.startswith("inter_neg_") for c in up.columns
    )

    if already_engineered:
        missing = ensure_required_columns(up, required_non_numeric)
        if missing:
            st.error(
                "Uploaded CSV contains engineered interaction columns, but is missing other required raw columns "
                "(typically categorical features expected by the pipeline). "
                f"Missing ({len(missing)}): {missing[:25]}" + ("..." if len(missing) > 25 else "")
            )
            st.stop()
    else:
        missing = ensure_required_columns(up, required_raw_cols)
        if missing:
            st.error(
                "Uploaded CSV is missing required columns for raw scoring. "
                "Use the template download above, or upload a file that already contains engineered `inter_pos_*`/`inter_neg_*` columns. "
                f"Missing ({len(missing)}): {missing[:25]}" + ("..." if len(missing) > 25 else "")
            )
            st.stop()

    try:
        out = predict_proba_attrition(
            model_label,
            up,
            dataset_for_mapping=base_df_full,
            threshold=float(threshold),
        )
        st.success("Predictions computed.")
        st.dataframe(out[["pred_attrition_proba", "pred_attrition_label"]].head(50), width="stretch")

        batch_color = "Attrition" if "Attrition" in out.columns else None
        if batch_color:
            st.caption("Grouped by: Attrition")
        dist = px.histogram(
            out,
            x="pred_attrition_proba",
            color=batch_color,
            nbins=30,
            barmode="overlay" if batch_color else "relative",
            title="Predicted probability distribution",
        )
        if batch_color:
            dist.update_traces(opacity=overlay_opacity)
            dist.update_layout(legend_title_text=batch_color)
        dist.update_layout(xaxis_tickformat=",.0%", height=320, margin=dict(l=10, r=10, t=40, b=10))
        show_summary_lines = st.checkbox(
            "Show mean/median lines on histogram (batch)",
            value=False,
            help="Adds reference lines to the probability distribution.",
        )
        if show_summary_lines:
            try:
                s = pd.to_numeric(out["pred_attrition_proba"], errors="coerce").dropna()
                if len(s):
                    dist.add_vline(x=float(s.mean()), line_dash="dash", line_width=3, line_color="#D62728")
                    dist.add_vline(x=float(s.median()), line_dash="dot", line_width=3, line_color="#1F77B4")
                    st.caption(f"Mean: {float(s.mean()):.1%} | Median: {float(s.median()):.1%}")
            except Exception:
                pass

        st.plotly_chart(dist, width="stretch")

        batch_state = {
            "chart": "Histogram",
            "x": "pred_attrition_proba",
            "color": batch_color,
            "title": "Predicted probability distribution",
            "global_filters_enabled": bool(st.session_state.get("global__enabled", True)),
            "n_rows": int(len(out)),
        }
        batch_state["short_description"] = short_plot_state_description(batch_state)
        if batch_state["short_description"]:
            st.caption(f"Plot summary: {batch_state['short_description']}")

        show_metrics_plot = st.checkbox(
            "Show metrics plot (batch)",
            value=True,
            help="Plots summary statistics for the uploaded batch.",
        )
        if show_metrics_plot:
            thr = float(threshold)
            try:
                s = pd.to_numeric(out["pred_attrition_proba"], errors="coerce").dropna()
                metrics = {
                    "mean proba": float(s.mean()) if len(s) else None,
                    "median proba": float(s.median()) if len(s) else None,
                    "p90 proba": float(s.quantile(0.90)) if len(s) else None,
                    f"share ≥ {thr:.0%}": float((s >= thr).mean()) if len(s) else None,
                }
                mfig = metrics_bar_figure(metrics, title="Batch scoring summary")
                mfig.update_layout(yaxis_tickformat=",.0%")
                st.plotly_chart(mfig, width="stretch")
            except Exception:
                pass

        st.subheader("Download predictions")
        st.caption("Choose a format: CSV / Excel / TXT.")
        download_dataframe(out, file_stem="attrition_predictions", label="Download predictions")

        st.subheader("Download batch report (HTML)")
        st.caption("Interactive HTML report (includes charts and key tables).")
        summary = (
            out[["pred_attrition_proba"]]
            .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            .reset_index()
            .rename(columns={"index": "metric"})
        )

        settings_df = pd.DataFrame(
            [
                {"setting": "model", "value": model_label},
                {"setting": "rows_scored", "value": str(len(out))},
                {"setting": "global_filters_enabled", "value": str(bool(st.session_state.get("global__enabled", True)))},
                {"setting": "threshold", "value": f"{float(threshold):.2f}"},
                {"setting": "short_description", "value": batch_state.get("short_description", "")},
            ]
        )
        download_plotly_html_report(
            title=f"Prediction — Batch Scoring ({model_label})",
            file_stem="report_prediction_batch",
            audience=audience,
            audience_markdown=AUDIENCE_MD,
            theme_mode=theme_mode,
            figures=[("Batch probability distribution", dist)],
            tables=[
                ("Chart description / settings", settings_df),
                ("Summary stats", summary),
                ("Predictions (first 200 rows)", out.head(200)),
            ],
        )

        st.subheader("Inspect one scored row (plots depend on selected features)")
        idx = st.number_input(
            "Row to inspect (uploaded)",
            min_value=0,
            max_value=max(0, len(out) - 1),
            value=0,
            step=1,
        )
        inspect = out.iloc[[int(idx)]].copy()
        st.metric("Selected row probability", f"{float(inspect['pred_attrition_proba'].iloc[0]):.1%}")

        inspect_features = st.multiselect(
            "Features to visualize",
            options=[c for c in up.columns if c != "Attrition"],
            default=[c for c in override_cols if c in up.columns],
        )
        for feat in inspect_features:
            st.markdown(f"#### {feat}")
            _plot_feature_context(base_df, feat, inspect.iloc[0][feat])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
