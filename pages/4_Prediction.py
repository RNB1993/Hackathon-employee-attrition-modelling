from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard_utils import (
    MODEL_CANDIDATES,
    ensure_required_columns,
    interaction_mapping,
    load_cleaned_dataset,
    predict_proba_attrition,
)

st.set_page_config(page_title="Prediction", layout="wide")

st.title("Prediction — Attrition Probability")

base_df = load_cleaned_dataset()

with st.sidebar:
    st.header("Model")
    model_label = st.selectbox("Choose model", list(MODEL_CANDIDATES.keys()))

st.subheader("Option A — Predict on an existing employee from the dataset")
row_id = st.number_input("Row index", min_value=0, max_value=len(base_df) - 1, value=0, step=1)
row = base_df.iloc[[int(row_id)]].copy()

st.write("Selected row (raw):")
st.dataframe(row, use_container_width=True)

try:
    pred_row = predict_proba_attrition(model_label, row, dataset_for_mapping=base_df)
    st.success("Prediction computed.")
    st.dataframe(pred_row[["pred_attrition_proba", "pred_attrition_label"]], use_container_width=True)
except Exception as e:
    st.error(f"Prediction failed: {e}")

st.divider()

st.subheader("Option B — Upload CSV (raw feature columns)")
uploaded = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded is not None:
    up = pd.read_csv(uploaded)
    st.write("Uploaded preview:")
    st.dataframe(up.head(20), use_container_width=True)

    mapping = interaction_mapping(base_df)
    required_raw_numeric = sorted(set(mapping["raw_feature_1"].tolist() + mapping["raw_feature_2"].tolist()))

    missing = ensure_required_columns(up, required_raw_numeric)
    if missing:
        st.error(
            "Uploaded CSV is missing raw numeric columns required to build engineered interactions. "
            f"Missing ({len(missing)}): {missing[:25]}" + ("..." if len(missing) > 25 else "")
        )
        st.stop()

    try:
        out = predict_proba_attrition(model_label, up, dataset_for_mapping=base_df)
        st.success("Predictions computed.")
        st.dataframe(out[["pred_attrition_proba", "pred_attrition_label"]].head(50), use_container_width=True)

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv, file_name="attrition_predictions.csv")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
