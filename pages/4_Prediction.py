from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_utils import (
    MODEL_CANDIDATES,
    audience_selector,
    configure_plotly_theme,
    download_dataframe,
    engineer_interactions,
    ensure_required_columns,
    interaction_mapping,
    load_cleaned_dataset,
    load_pipeline,
    predict_proba_attrition,
    render_audience_markdown,
)

st.set_page_config(page_title="Prediction", layout="wide")

configure_plotly_theme()

st.title("Prediction — Attrition Probability")

base_df = load_cleaned_dataset()

audience = audience_selector()

with st.sidebar:
    st.header("Model")
    model_label = st.selectbox("Choose model", list(MODEL_CANDIDATES.keys()))

render_audience_markdown(
    {
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
    },
    audience=audience,
)


@st.cache_resource(show_spinner=False)
def _get_pipe(model_label: str):
    return load_pipeline(model_label)


def _predict_proba_from_raw(raw_df: pd.DataFrame, *, pipe, mapping: pd.DataFrame) -> np.ndarray:
    X = engineer_interactions(raw_df, mapping=mapping)
    if "Attrition" in X.columns:
        X = X.drop(columns=["Attrition"])
    return pipe.predict_proba(X)[:, 1]


def _plot_feature_context(base: pd.DataFrame, feature: str, value) -> None:
    if feature not in base.columns:
        st.info(f"No plot available: missing column {feature}")
        return

    s = base[feature]
    if pd.api.types.is_numeric_dtype(s):
        fig = px.histogram(base, x=feature, nbins=30, title=f"{feature} distribution")
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


pipe = _get_pipe(model_label)
mapping = interaction_mapping(base_df)

st.subheader("Option A — Predict on an existing employee from the dataset")
row_id = st.number_input("Row index", min_value=0, max_value=len(base_df) - 1, value=0, step=1)
row_base = base_df.iloc[[int(row_id)]].copy()

# If the base row changes, drop previously-entered overrides to avoid confusion.
if st.session_state.get("row_overrides_row_id") != int(row_id):
    st.session_state.pop("row_overrides", None)
    st.session_state["row_overrides_row_id"] = int(row_id)

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

    if not features_to_plot:
        st.info("Select one or more features to see input-dependent plots.")
    else:
        for feat in features_to_plot:
            st.markdown(f"#### {feat}")
            _plot_feature_context(base_df, feat, row.iloc[0][feat])

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
        mapping = interaction_mapping(base_df)
        required_raw_numeric = sorted(
            set(mapping["raw_feature_1"].tolist() + mapping["raw_feature_2"].tolist())
        )

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

        st.subheader("Download predictions")
        download_dataframe(out, file_stem="attrition_predictions", label="Download")

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
