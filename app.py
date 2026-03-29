import streamlit as st

import dashboard_utils as du


AUDIENCE_MD = {
    "Non-technical": """
This dashboard is designed for quick exploration and clear takeaways.

Use the **Prediction** page to see attrition probability for example employee rows, and the **EDA/Stats** pages for intuitive patterns.
""".strip(),
    "Semi-technical": """
This view includes some statistical context (tests, distributions) and model-driven interaction rankings.

Use **Stats** and **Feature Engineering** to cross-check patterns and correlations.
""".strip(),
    "Technical": """
This dashboard exposes model-related artifacts (feature-space importances) and supports exporting Plotly HTML reports.

Use **Feature Engineering** and **Prediction** for deeper investigation.
""".strip(),
}


def main() -> None:
    du.set_page_config(page_title="IBM HR Attrition Dashboard", layout="wide")

    theme_mode = du.theme_selector()
    audience = du.audience_selector()
    du.apply_app_theme(theme_mode)
    du.configure_plotly_theme(theme_mode)

    st.title("IBM HR Analytics — Attrition Dashboard")
    du.render_audience_markdown(audience, AUDIENCE_MD)

    with st.expander("Deployment notes", expanded=True):
        st.markdown(
            "If you deployed with the existing `Procfile`, it runs `streamlit run app.py`.\n"
            "The pages are in the `pages/` folder."
        )


if __name__ == "__main__":
    main()
