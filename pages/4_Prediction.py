from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_utils import (
    MODEL_CANDIDATES,
    apply_app_theme,
    audience_selector,
    configure_plotly_theme,
    download_dataframe,
    download_plotly_html_report,
    engineer_interactions,
    ensure_required_columns,
    interaction_mapping,
    load_cleaned_dataset,
    load_pipeline,
    predict_proba_attrition,
    render_audience_markdown,
    theme_selector,
)

st.set_page_config(page_title="Prediction", layout="wide")

theme_mode = theme_selector()
apply_app_theme(theme_mode)
configure_plotly_theme(theme_mode)

st.title("Prediction — Attrition Probability")

base_df = load_cleaned_dataset()

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

overlay_opacity = st.sidebar.slider(
    "Overlay opacity (grouped histograms)",
    min_value=0.15,
    max_value=0.95,
    value=0.65,
    step=0.05,
    help="Lower opacity makes overlapping groups easier to distinguish.",
)


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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)
        return fig


pipe = _get_pipe(model_label)
mapping = interaction_mapping(base_df)
required_raw_numeric = sorted(
    set(mapping["raw_feature_1"].tolist() + mapping["raw_feature_2"].tolist())
)

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

st.dataframe(filtered.head(30), use_container_width=True)

st.markdown("### Score the filtered group (optional)")
score_group = st.button("Score filtered group", type="secondary")
group_out = None
if score_group:
    try:
        group_out = predict_proba_attrition(model_label, filtered, dataset_for_mapping=base_df)
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
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Download group predictions")
        download_dataframe(group_out, file_stem="filtered_group_predictions", label="Download")

        st.subheader("Download group report (HTML)")
        summary = (
            group_out[["pred_attrition_proba"]]
            .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            .reset_index()
            .rename(columns={"index": "metric"})
        )
        download_plotly_html_report(
            title=f"Prediction — Group Scoring ({model_label})",
            file_stem="report_prediction_group",
            audience=audience,
            audience_markdown=AUDIENCE_MD,
            theme_mode=theme_mode,
            figures=[("Group probability distribution", fig)],
            tables=[
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
st.dataframe(row_base, use_container_width=True)

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
    st.dataframe(row, use_container_width=True)

try:
    pred_row = predict_proba_attrition(model_label, row, dataset_for_mapping=base_df)
    st.success("Prediction computed.")
    proba = float(pred_row["pred_attrition_proba"].iloc[0])
    st.metric("Predicted attrition probability", f"{proba:.1%}")
    st.dataframe(pred_row[["pred_attrition_proba", "pred_attrition_label"]], use_container_width=True)

    st.subheader("Download prediction")
    download_dataframe(pred_row, file_stem="single_employee_prediction", label="Download")

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
            st.plotly_chart(fig, use_container_width=True)
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
            st.plotly_chart(fig, use_container_width=True)
            sens_fig = fig

    st.subheader("Download prediction report (HTML)")
    report_figs: list[tuple[str, object]] = []
    report_figs.extend(context_figs)
    if sens_fig is not None:
        report_figs.append(("What-if sensitivity", sens_fig))

    download_plotly_html_report(
        title=f"Prediction — Single Employee ({model_label})",
        file_stem="report_prediction_single",
        audience=audience,
        audience_markdown=AUDIENCE_MD,
        theme_mode=theme_mode,
        figures=report_figs if report_figs else None,
        tables=[
            ("Input row (raw)", row_base.reset_index(drop=True)),
            ("Input row (after overrides)", row.reset_index(drop=True)),
            ("Prediction output", pred_row.reset_index(drop=True)),
        ],
    )
except Exception as e:
    st.error(f"Prediction failed: {e}")

st.divider()

st.subheader("Option B — Upload CSV (raw feature columns)")
uploaded = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded is not None:
    up = pd.read_csv(uploaded)
    st.write("Uploaded preview:")
    st.dataframe(up.head(20), use_container_width=True)

    already_engineered = any(
        c.startswith("inter_pos_") or c.startswith("inter_neg_") for c in up.columns
    )

    if not already_engineered:
        missing = ensure_required_columns(up, required_raw_numeric)
        if missing:
            st.error(
                "Uploaded CSV is missing raw numeric columns required to build engineered interactions. "
                "Alternatively, upload a file that already contains engineered `inter_pos_*`/`inter_neg_*` columns. "
                f"Missing ({len(missing)}): {missing[:25]}" + ("..." if len(missing) > 25 else "")
            )
            st.stop()

    try:
        out = predict_proba_attrition(model_label, up, dataset_for_mapping=base_df)
        st.success("Predictions computed.")
        st.dataframe(out[["pred_attrition_proba", "pred_attrition_label"]].head(50), use_container_width=True)

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
        st.plotly_chart(dist, use_container_width=True)

        st.subheader("Download predictions")
        download_dataframe(out, file_stem="attrition_predictions", label="Download")

        st.subheader("Download batch report (HTML)")
        summary = (
            out[["pred_attrition_proba"]]
            .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            .reset_index()
            .rename(columns={"index": "metric"})
        )
        download_plotly_html_report(
            title=f"Prediction — Batch Scoring ({model_label})",
            file_stem="report_prediction_batch",
            audience=audience,
            audience_markdown=AUDIENCE_MD,
            theme_mode=theme_mode,
            figures=[("Batch probability distribution", dist)],
            tables=[
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
