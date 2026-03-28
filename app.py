
import streamlit as st

from dashboard_utils import audience_selector, configure_plotly_theme, render_audience_markdown

st.set_page_config(
    page_title="IBM HR Attrition Dashboard",
    page_icon="📊",
    layout="wide",
)

configure_plotly_theme()

st.title("IBM HR Analytics — Attrition Dashboard")

audience = audience_selector()

render_audience_markdown(
    {
        "Non-technical": """
Welcome. This dashboard helps explore patterns related to employee attrition and gives a **risk estimate**.

Use the left sidebar to navigate:
- **EDA**: interactive charts
- **Stats**: simple comparisons between groups
- **Feature Engineering**: how extra features were created
- **Prediction**: score an employee or upload a file and export results
""",
        "Semi-technical": """
Multi-page Streamlit dashboard for IBM HR attrition analytics.

- **EDA** (Plotly): univariate + bivariate interactive charts
- **Stats**: hypothesis tests + association metrics
- **Feature Engineering**: correlations + engineered interaction overview
- **Prediction**: score a row, key in edits, run what-if plots, or batch-score uploads
""",
        "Technical": """
Streamlit front-end on top of pandas + plotly + sklearn pipelines.

Pages include EDA, statistical tests, feature engineering (interaction mapping), and prediction with scenario analysis + exports.
""",
    },
    audience=audience,
)

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
