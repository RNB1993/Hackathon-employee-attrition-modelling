from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

CLEANED_CSV_CANDIDATES = [
    DATA_DIR / "Cleaned_dataset" / "WA_Fn-UseC_-HR-Employee-Attrition_capped.csv",
    DATA_DIR / "Cleaned_data" / "WA_Fn-UseC_-HR-Employee-Attrition_capped.csv",
    REPO_ROOT / "WA_Fn-UseC_-HR-Employee-Attrition_capped.csv",
]

MODEL_DIR = REPO_ROOT / "jupyter_notebooks" / "model"
MODEL_CANDIDATES = {
    "Best model": MODEL_DIR / "attrition_minority_yes_best_model.pkl",
    "Business-cost model": MODEL_DIR / "attrition_minority_yes_business_cost_second_best.pkl",
}


@dataclass(frozen=True)
class InteractionRow:
    interaction: str
    kind: str  # "pos" or "neg"
    raw_feature_1: str
    raw_feature_2: str
    spearman_r: float


def _st_cache_data(func):
    if st is None:
        return func
    return st.cache_data(show_spinner=False)(func)


def _st_cache_resource(func):
    if st is None:
        return func
    return st.cache_resource(show_spinner=False)(func)


@_st_cache_data
def load_cleaned_dataset() -> pd.DataFrame:
    path = next((p for p in CLEANED_CSV_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Could not find cleaned dataset. Looked for: "
            + ", ".join(str(p) for p in CLEANED_CSV_CANDIDATES)
        )

    df = pd.read_csv(path)
    return df


@_st_cache_data
def numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


@_st_cache_data
def correlation_pairs(df: pd.DataFrame) -> pd.DataFrame:
    num = numeric_df(df)
    corr = num.corr(method="spearman")

    corr_long = (
        corr.where(~np.eye(corr.shape[0], dtype=bool)).stack().reset_index(name="spearman_r")
    )
    corr_long.columns = ["feature_1", "feature_2", "spearman_r"]

    corr_long["pair_key"] = corr_long.apply(
        lambda r: tuple(sorted([r["feature_1"], r["feature_2"]])), axis=1
    )
    corr_pairs = corr_long.drop_duplicates(subset=["pair_key"]).drop(columns=["pair_key"])

    # Match the notebook: descending by spearman_r
    corr_pairs = corr_pairs.sort_values("spearman_r", ascending=False).reset_index(drop=True)
    return corr_pairs


@_st_cache_data
def interaction_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Return the mapping from inter_pos/inter_neg to the raw feature pairs.

    This follows the same ordering logic used in Feature_engineering.ipynb:
    - correlations are sorted by spearman_r DESC
    - positive pairs get inter_pos_001.. in that order
    - negative pairs get inter_neg_001.. in that order (still descending, i.e. closest-to-zero first)
    """

    corr_pairs = correlation_pairs(df)

    positive_pairs = corr_pairs[corr_pairs["spearman_r"] > 0].reset_index(drop=True)
    negative_pairs = corr_pairs[corr_pairs["spearman_r"] < 0].reset_index(drop=True)

    rows: list[dict] = []

    for i, row in enumerate(positive_pairs.itertuples(index=False), start=1):
        rows.append(
            {
                "interaction": f"inter_pos_{i:03d}",
                "kind": "pos",
                "raw_feature_1": row.feature_1,
                "raw_feature_2": row.feature_2,
                "spearman_r": float(row.spearman_r),
                "formula": f"({row.feature_1} / {row.feature_2}) * {float(row.spearman_r):.6f}",
            }
        )

    for i, row in enumerate(negative_pairs.itertuples(index=False), start=1):
        rows.append(
            {
                "interaction": f"inter_neg_{i:03d}",
                "kind": "neg",
                "raw_feature_1": row.feature_1,
                "raw_feature_2": row.feature_2,
                "spearman_r": float(row.spearman_r),
                "formula": f"({row.feature_1} * {row.feature_2}) * {float(row.spearman_r):.6f}",
            }
        )

    return pd.DataFrame(rows)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def engineer_interactions(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    target_col: str = "Attrition",
) -> pd.DataFrame:
    """Create `inter_pos_*` and `inter_neg_*` columns and drop the raw numeric inputs.

    This mirrors the notebook logic closely.

    Requirements:
    - `df` must include all raw numeric columns referenced in `mapping`.
    - categorical columns are preserved.

    Returns a new engineered dataframe.
    """

    engineered = df.copy()
    used_original_features: set[str] = set()

    # Build pos and neg in the same order as mapping
    for row in mapping.itertuples(index=False):
        inter = row.interaction
        f1 = row.raw_feature_1
        f2 = row.raw_feature_2
        r = float(row.spearman_r)

        if f1 not in engineered.columns or f2 not in engineered.columns:
            raise KeyError(
                f"Missing required raw numeric columns for interaction {inter}: {f1}, {f2}"
            )

        if row.kind == "pos":
            engineered[inter] = _safe_divide(engineered[f1], engineered[f2]) * r
        else:
            engineered[inter] = (engineered[f1] * engineered[f2]) * r

        used_original_features.update([f1, f2])

    # Drop original numeric features used in engineering, keep target if present
    drop_original = [c for c in sorted(used_original_features) if c in engineered.columns]
    if target_col in drop_original:
        drop_original.remove(target_col)

    engineered = engineered.drop(columns=drop_original)
    return engineered


@_st_cache_resource
def load_pipeline(model_label: str):
    path = MODEL_CANDIDATES.get(model_label)
    if path is None:
        raise KeyError(f"Unknown model label: {model_label}")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def predict_proba_attrition(
    model_label: str,
    df_raw: pd.DataFrame,
    dataset_for_mapping: pd.DataFrame,
    target_col: str = "Attrition",
) -> pd.DataFrame:
    """Return a dataframe with predicted attrition probability for each row.

    The saved sklearn pipeline expects engineered interaction features.
    We reconstruct those interaction features using mapping derived from the training dataset.
    """

    pipe = load_pipeline(model_label)
    mapping = interaction_mapping(dataset_for_mapping)

    # If user already supplied engineered columns, do not re-engineer.
    already_engineered = any(c.startswith("inter_pos_") or c.startswith("inter_neg_") for c in df_raw.columns)

    if already_engineered:
        X = df_raw.copy()
    else:
        X = engineer_interactions(df_raw, mapping=mapping, target_col=target_col)

    if target_col in X.columns:
        X = X.drop(columns=[target_col])

    proba = pipe.predict_proba(X)[:, 1]
    out = df_raw.copy()
    out["pred_attrition_proba"] = proba
    out["pred_attrition_label"] = (proba >= 0.5).astype(int)
    return out


def interaction_importance_table(
    model_label: str,
    dataset_for_mapping: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    pipe = load_pipeline(model_label)

    # Extract LR + feature names after preprocessing
    if not hasattr(pipe, "named_steps"):
        raise TypeError("Expected a sklearn Pipeline with named_steps")

    model = pipe.named_steps.get("model")
    pre = pipe.named_steps.get("preprocessor") or pipe.named_steps.get("preprocess")

    if model is None or pre is None:
        raise KeyError("Pipeline must contain 'preprocessor' (or 'preprocess') and 'model' steps")

    feature_names = list(pre.get_feature_names_out())
    coefs = model.coef_[0]

    mapping = interaction_mapping(dataset_for_mapping).set_index("interaction")

    rows = []
    for fname, c in zip(feature_names, coefs):
        if "inter_pos_" not in fname and "inter_neg_" not in fname:
            continue
        inter = fname.split("__", 1)[-1]
        if inter not in mapping.index:
            continue
        m = mapping.loc[inter]
        rows.append(
            {
                "interaction": inter,
                "kind": m["kind"],
                "raw_feature_1": m["raw_feature_1"],
                "raw_feature_2": m["raw_feature_2"],
                "spearman_r": float(m["spearman_r"]),
                "coef": float(c),
                "abs_coef": float(abs(c)),
                "odds_ratio": float(math.exp(c)),
                "direction": "increases_attrition" if c > 0 else "decreases_attrition",
            }
        )

    df = pd.DataFrame(rows).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df.head(top_n)


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    return missing
