
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

EVALUATION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluation"
    / "evaluation_results.json"
)

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"


# ==================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 3. HELPER FUNCTIONS
# ==================================================

def check_api_connection():

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )

        if response.status_code == 200:
            return response.json()

    except requests.RequestException:
        return None

    return None


def load_customer_data():

    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    return None


def load_evaluation_results():

    if EVALUATION_PATH.exists():

        with open(EVALUATION_PATH, "r") as file:
            return json.load(file)

    return {}


def get_dashboard_metrics():

    df = load_customer_data()

    if df is not None and "Churn" in df.columns:

        total_customers = len(df)

        churn_rate = (
            df["Churn"].astype(str).str.strip().eq("Yes").mean()
            * 100
        )

    else:
        total_customers = 0
        churn_rate = 0

    model_count = len(
        list(MODEL_DIR.glob("*.joblib"))
    ) if MODEL_DIR.exists() else 0

    evaluation = load_evaluation_results()

    available_scores = [
        values["accuracy"]
        for name, values in evaluation.items()
        if isinstance(values, dict) and "accuracy" in values
    ]

    best_accuracy = (
        max(available_scores) * 100
        if available_scores else 0
    )

    return (
        total_customers,
        churn_rate,
        model_count,
        best_accuracy
    )


# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:

    st.title("📡 Telecom MLOps")
    st.caption("Customer Churn Prediction")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Predict Churn",
            "Customer Insights",
            "Model Performance",
            "ML Pipeline",
            "About Project"
        ]
    )

    st.divider()

    st.subheader("Project Stack")

    st.write("🐍 Python")
    st.write("🧠 Scikit-learn")
    st.write("📊 MLflow")
    st.write("📦 DVC")
    st.write("⚡ FastAPI")
    st.write("🎨 Streamlit")

    st.divider()

    api_status = check_api_connection()

    if api_status and api_status.get("model_loaded"):

        st.success("API Connected")

    else:

        st.error("API Disconnected")

    st.caption(API_URL)


# ==================================================
# 5. DASHBOARD HEADER
# ==================================================

st.title("📡 Telecom Customer Churn Predictor")

st.caption(
    "Predict whether a customer is likely to leave "
    "using a trained Machine Learning model."
)

st.divider()


# ==================================================
# 6. DASHBOARD METRICS
# ==================================================

total_customers, churn_rate, model_count, best_accuracy = (
    get_dashboard_metrics()
)

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

metric2.metric(
    "Churn Rate",
    f"{churn_rate:.1f}%"
)

metric3.metric(
    "Models Trained",
    model_count
)

metric4.metric(
    "Best Model Accuracy",
    f"{best_accuracy:.2f}%"
)

st.divider()


# ==================================================
# 7. PREDICT CHURN PAGE
# ==================================================

if page == "Predict Churn":

    left, right = st.columns(
        [1.35, 1],
        gap="large"
    )

    with left:

        st.subheader("👤 Customer Details")

        st.write(
            "Enter customer information to predict churn."
        )

        with st.form("customer_prediction_form"):

            # Personal information
            st.markdown("#### Personal Information")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                gender = st.selectbox(
                    "Gender",
                    ["Female", "Male"]
                )

            with c2:
                senior_citizen = st.selectbox(
                    "Senior Citizen",
                    [0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No"
                )

            with c3:
                partner = st.selectbox(
                    "Partner",
                    ["Yes", "No"]
                )

            with c4:
                dependents = st.selectbox(
                    "Dependents",
                    ["Yes", "No"]
                )

            st.divider()

            # Service information
            st.markdown("#### Service Information")

            c1, c2, c3 = st.columns(3)

            with c1:
                tenure = st.number_input(
                    "Tenure (months)",
                    min_value=0,
                    max_value=100,
                    value=12
                )

            with c2:
                phone_service = st.selectbox(
                    "Phone Service",
                    ["Yes", "No"]
                )

            with c3:
                multiple_lines = st.selectbox(
                    "Multiple Lines",
                    ["No", "Yes", "No phone service"]
                )

            c1, c2, c3 = st.columns(3)

            with c1:
                internet_service = st.selectbox(
                    "Internet Service",
                    ["Fiber optic", "DSL", "No"]
                )

            with c2:
                online_security = st.selectbox(
                    "Online Security",
                    ["Yes", "No", "No internet service"]
                )

            with c3:
                online_backup = st.selectbox(
                    "Online Backup",
                    ["Yes", "No", "No internet service"]
                )

            c1, c2, c3 = st.columns(3)

            with c1:
                device_protection = st.selectbox(
                    "Device Protection",
                    ["Yes", "No", "No internet service"]
                )

            with c2:
                tech_support = st.selectbox(
                    "Tech Support",
                    ["Yes", "No", "No internet service"]
                )

            with c3:
                streaming_tv = st.selectbox(
                    "Streaming TV",
                    ["Yes", "No", "No internet service"]
                )

            c1, c2 = st.columns(2)

            with c1:
                streaming_movies = st.selectbox(
                    "Streaming Movies",
                    ["Yes", "No", "No internet service"]
                )

            with c2:
                contract = st.selectbox(
                    "Contract",
                    [
                        "Month-to-month",
                        "One year",
                        "Two year"
                    ]
                )

            st.divider()

            # Billing information
            st.markdown("#### Billing Information")

            c1, c2, c3 = st.columns(3)

            with c1:
                paperless_billing = st.selectbox(
                    "Paperless Billing",
                    ["Yes", "No"]
                )

            with c2:
                payment_method = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)"
                    ]
                )

            with c3:
                monthly_charges = st.number_input(
                    "Monthly Charges (₹)",
                    min_value=0.0,
                    value=70.50,
                    step=1.0
                )

            total_charges = st.number_input(
                "Total Charges (₹)",
                min_value=0.0,
                value=850.0,
                step=10.0
            )

            predict_button = st.form_submit_button(
                "🚀 Predict Churn",
                use_container_width=True
            )

        # Submit customer information to FastAPI
        if predict_button:

            customer_data = {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }

            try:

                with st.spinner("Analyzing customer information..."):

                    response = requests.post(
                        f"{API_URL}/predict",
                        json=customer_data,
                        timeout=30
                    )

                if response.status_code == 200:

                    st.session_state["prediction_result"] = (
                        response.json()
                    )

                else:

                    st.session_state.pop(
                        "prediction_result", None
                    )

                    st.error(
                        f"Prediction failed: {response.text}"
                    )

            except requests.RequestException as error:

                st.error(
                    f"Cannot connect to FastAPI: {error}"
                )

    # Prediction result panel
    with right:

        st.subheader("🎯 Prediction Result")

        result = st.session_state.get("prediction_result")

        if result:

            probability = result["churn_probability"]
            percentage = result["churn_percentage"]

            if result["prediction"] == 1:

                st.error("⚠️ Customer Likely to Churn")

                st.write(
                    "The model predicts that this customer "
                    "has a higher likelihood of leaving."
                )

            else:

                st.success("✅ Customer Likely to Stay")

                st.write(
                    "The model predicts that this customer "
                    "is likely to remain."
                )

            st.markdown("### Churn Probability")

            st.title(f"{percentage:.2f}%")

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            r1, r2 = st.columns(2)

            r1.metric(
                "Prediction",
                result["prediction"]
            )

            r2.metric(
                "Probability",
                f"{probability:.4f}"
            )

            st.divider()

            st.markdown("#### 💡 What does this mean?")

            if result["prediction"] == 1:

                st.info(
                    "Consider proactive retention strategies, "
                    "such as suitable plans, offers and customer support."
                )

            else:

                st.info(
                    "The model estimates a lower likelihood of churn "
                    "for this customer."
                )

            st.caption(
                "This is a machine-learning estimate, not a guarantee."
            )

        else:

            st.info(
                "Enter customer details and click Predict Churn "
                "to see the prediction here."
            )

            st.markdown("### Prediction output")

            st.write("Your prediction will appear here.")

            st.progress(0)


# ==================================================
# 8. CUSTOMER INSIGHTS PAGE
# ==================================================

elif page == "Customer Insights":

    st.subheader("📊 Customer Insights")

    df = load_customer_data()

    if df is not None:

        c1, c2 = st.columns(2)

        with c1:

            st.markdown("#### Churn Distribution")

            churn_counts = (
                df["Churn"]
                .value_counts()
                .rename_axis("Customer Status")
                .reset_index(name="Customers")
            )

            st.bar_chart(
                churn_counts,
                x="Customer Status",
                y="Customers"
            )

        with c2:

            st.markdown("#### Contract Distribution")

            contract_counts = (
                df["Contract"]
                .value_counts()
                .rename_axis("Contract")
                .reset_index(name="Customers")
            )

            st.bar_chart(
                contract_counts,
                x="Contract",
                y="Customers"
            )

        st.markdown("#### Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

    else:

        st.warning("Raw customer dataset was not found.")


# ==================================================
# 9. MODEL PERFORMANCE PAGE
# ==================================================

elif page == "Model Performance":

    st.subheader("🏆 Model Performance")

    results = load_evaluation_results()

    model_results = {
        name: values
        for name, values in results.items()
        if isinstance(values, dict) and "accuracy" in values
    }

    if model_results:

        performance_df = pd.DataFrame(
            model_results
        ).T

        st.markdown("#### Model Comparison")

        st.dataframe(
            performance_df.style.format("{:.4f}"),
            use_container_width=True
        )

        st.markdown("#### Accuracy Comparison")

        accuracy_df = performance_df[
            ["accuracy"]
        ]

        st.bar_chart(accuracy_df)

        st.markdown("#### F1-score Comparison")

        f1_df = performance_df[
            ["f1_score"]
        ]

        st.bar_chart(f1_df)

        if "best_model" in results:

            st.success(
                f"Best model according to F1-score: "
                f"{results['best_model']}"
            )

    else:

        st.warning(
            "Evaluation results not found. "
            "Run the model evaluation script first."
        )


# ==================================================
# 10. ML PIPELINE PAGE
# ==================================================

elif page == "ML Pipeline":

    st.subheader("⚙️ Machine Learning Pipeline")

    pipeline_steps = [
        ("1", "Data Collection", "Customer churn dataset"),
        ("2", "EDA", "Exploratory data analysis"),
        ("3", "Preprocessing", "Data cleaning and train/test split"),
        ("4", "Feature Engineering", "Encoding, imputation and scaling"),
        ("5", "Model Training", "Logistic Regression, Decision Tree and Random Forest"),
        ("6", "Model Evaluation", "Accuracy, precision, recall, F1 and ROC-AUC"),
        ("7", "MLflow", "Experiment tracking"),
        ("8", "DVC", "Dataset and pipeline versioning"),
        ("9", "FastAPI", "Prediction backend"),
        ("10", "Streamlit", "User interface")
    ]

    for number, title, description in pipeline_steps:

        with st.container(border=True):

            st.markdown(
                f"**{number}. {title}**"
            )

            st.caption(description)


# ==================================================
# 11. ABOUT PROJECT PAGE
# ==================================================

elif page == "About Project":

    st.subheader("ℹ️ About This Project")

    st.write(
        """
        This is an end-to-end Customer Churn Prediction
        MLOps project.

        The application predicts whether a telecom customer
        is likely to leave the company using machine learning.
        """
    )

    st.markdown("#### Technologies Used")

    technologies = [
        "Python",
        "Pandas",
        "Scikit-learn",
        "MLflow",
        "DVC",
        "FastAPI",
        "Streamlit"
    ]

    for technology in technologies:

        st.write(f"• {technology}")

    st.markdown("#### Project Objective")

    st.info(
        "Help telecom businesses identify customers at risk "
        "of leaving so they can plan suitable retention strategies."
    )

    st.caption(
        "Customer Churn Prediction | End-to-End MLOps Project"
    )
