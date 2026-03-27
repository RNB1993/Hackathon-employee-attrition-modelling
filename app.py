import streamlit as st

st.set_page_config(
    page_title="IBM HR Attrition Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("IBM HR Analytics — Attrition Dashboard")

st.markdown(
    """
This is a multi-page Streamlit dashboard built from the EDA, statistical tests, and feature engineering work.

Use the left sidebar (or the Pages menu) to navigate:
- **EDA (Plotly)**: univariate + bivariate interactive charts (color/hue, facet rows/cols)
- **Stats**: quick hypothesis tests and association metrics
- **Feature Engineering**: correlations + engineered interactions overview
- **Prediction**: load the saved model pipelines and predict attrition probability
"""
)

st.info(
    "If you deployed with the existing `Procfile`, it runs `streamlit run app.py`. "
    "The pages are in the `pages/` folder."
)
