import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


DATA_PATH = Path(__file__).parent / "data" / "Cleaned_dataset" / "WA_Fn-UseC_-HR-Employee-Attrition_capped.csv"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def _safe_category_options(series: pd.Series) -> list[str]:
    values = (
        series.dropna()
        .astype(str)
        .replace({"": None})
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(values)


def main() -> None:
    st.set_page_config(
        page_title="IBM HR Attrition Insights",
        page_icon="📊",
        layout="wide",
    )

    st.title("IBM HR Employee Attrition — Insights Dashboard")
    st.caption("Interactive exploration of the cleaned IBM HR dataset.")

    with st.spinner("Loading dataset..."):
        df = load_data()

    if "Attrition" not in df.columns:
        st.error("Expected column 'Attrition' not found in dataset.")
        st.stop()

    # Normalize Attrition to Yes/No strings
    attrition = df["Attrition"].astype(str).str.strip().str.title()
    df = df.copy()
    df["Attrition"] = attrition.where(attrition.isin(["Yes", "No"]), attrition)

    st.sidebar.header("Filters")

    department_col = "Department" if "Department" in df.columns else None
    job_role_col = "JobRole" if "JobRole" in df.columns else None
    gender_col = "Gender" if "Gender" in df.columns else None

    filtered = df

    if department_col:
        dept_options = _safe_category_options(df[department_col])
        selected_depts = st.sidebar.multiselect("Department", dept_options, default=dept_options)
        if selected_depts:
            filtered = filtered[filtered[department_col].astype(str).isin(selected_depts)]

    if job_role_col:
        role_options = _safe_category_options(df[job_role_col])
        selected_roles = st.sidebar.multiselect("Job Role", role_options, default=role_options)
        if selected_roles:
            filtered = filtered[filtered[job_role_col].astype(str).isin(selected_roles)]

    if gender_col:
        gender_options = _safe_category_options(df[gender_col])
        selected_genders = st.sidebar.multiselect("Gender", gender_options, default=gender_options)
        if selected_genders:
            filtered = filtered[filtered[gender_col].astype(str).isin(selected_genders)]

    if "Age" in df.columns:
        age_min = int(pd.to_numeric(df["Age"], errors="coerce").min())
        age_max = int(pd.to_numeric(df["Age"], errors="coerce").max())
        selected_age = st.sidebar.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max))
        filtered = filtered[pd.to_numeric(filtered["Age"], errors="coerce").between(selected_age[0], selected_age[1])]

    # KPIs
    left, mid, right, far_right = st.columns(4)

    employee_count = int(len(filtered))
    attrition_rate = float((filtered["Attrition"].astype(str).str.title() == "Yes").mean()) if employee_count else 0.0

    with left:
        st.metric("Employees", f"{employee_count:,}")
    with mid:
        st.metric("Attrition Rate", f"{attrition_rate * 100:.1f}%")
    with right:
        if "MonthlyIncome" in filtered.columns and employee_count:
            avg_income = pd.to_numeric(filtered["MonthlyIncome"], errors="coerce").mean()
            st.metric("Avg Monthly Income", f"{avg_income:,.0f}")
        else:
            st.metric("Avg Monthly Income", "—")
    with far_right:
        if "YearsAtCompany" in filtered.columns and employee_count:
            avg_tenure = pd.to_numeric(filtered["YearsAtCompany"], errors="coerce").mean()
            st.metric("Avg Years at Company", f"{avg_tenure:.1f}")
        else:
            st.metric("Avg Years at Company", "—")

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition by Department")
        if department_col and employee_count:
            plot_df = (
                filtered.assign(_attr=(filtered["Attrition"].astype(str).str.title() == "Yes").astype(int))
                .groupby(department_col, as_index=False)["_attr"]
                .mean()
                .sort_values("_attr", ascending=False)
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=plot_df, x="_attr", y=department_col, ax=ax, palette="Blues_r")
            ax.set_xlabel("Attrition rate")
            ax.set_ylabel("")
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x * 100:.0f}%")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Department column not available in this dataset.")

    with col2:
        st.subheader("Attrition Drivers Snapshot")
        driver_cols = [c for c in ["OverTime", "BusinessTravel", "JobSatisfaction", "WorkLifeBalance"] if c in filtered.columns]
        if employee_count and driver_cols:
            driver = st.selectbox("Choose a driver", driver_cols)
            tmp = filtered.copy()
            tmp["_attr"] = (tmp["Attrition"].astype(str).str.title() == "Yes").astype(int)
            plot_df = (
                tmp.groupby(driver, as_index=False)["_attr"]
                .mean()
                .sort_values("_attr", ascending=False)
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=plot_df, x="_attr", y=driver, ax=ax, palette="Reds_r")
            ax.set_xlabel("Attrition rate")
            ax.set_ylabel("")
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x * 100:.0f}%")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("No driver columns found to chart.")

    st.subheader("Filtered Data")
    st.dataframe(filtered, use_container_width=True, height=380)


if __name__ == "__main__":
    main()
