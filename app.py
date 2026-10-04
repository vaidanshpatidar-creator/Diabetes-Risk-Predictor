import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Load model & encoder ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("diabetes_model.joblib")
    le    = joblib.load("label_encoder.joblib")
    return model, le

model, le = load_artifacts()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")
st.markdown(
    "Enter your clinical measurements below and click **Predict** to see "
    "whether the model classifies you as **Positive** or **Negative** for diabetes."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Patient Measurements")

col1, col2 = st.columns(2)

with col1:
    hba1c = st.number_input(
        "HbA1c (%)",
        min_value=3.0,
        max_value=20.0,
        value=5.5,
        step=0.1,
        help="Glycated haemoglobin — average blood glucose over ~3 months. Normal < 5.7 %",
    )
    fasting_bg = st.number_input(
        "Fasting Blood Glucose (mg/dL)",
        min_value=50.0,
        max_value=400.0,
        value=100.0,
        step=0.5,
        help="Blood glucose after at least 8 hours of fasting. Normal < 100 mg/dL",
    )
    postprandial_bg = st.number_input(
        "Postprandial Blood Glucose (mg/dL)",
        min_value=50.0,
        max_value=500.0,
        value=120.0,
        step=0.5,
        help="Blood glucose 2 hours after a meal. Normal < 140 mg/dL",
    )

with col2:
    random_bg = st.number_input(
        "Random Blood Glucose (mg/dL)",
        min_value=50.0,
        max_value=500.0,
        value=140.0,
        step=0.5,
        help="Blood glucose measured at any random time. Normal < 200 mg/dL",
    )
    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=10.0,
        max_value=60.0,
        value=24.0,
        step=0.1,
        help="Body Mass Index. Normal range: 18.5 – 24.9",
    )

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("Predict", type="primary", use_container_width=True):
    features = pd.DataFrame(
        [[hba1c, fasting_bg, postprandial_bg, random_bg, bmi]],
        columns=[
            "HbA1c",
            "Fasting_Blood_Glucose",
            "Postprandial_Blood_Glucose",
            "Random_Blood_Glucose",
            "BMI",
        ],
    )

    pred_encoded   = model.predict(features)[0]
    pred_proba     = model.predict_proba(features)[0]
    pred_label     = le.inverse_transform([pred_encoded])[0]

    # Class order from LabelEncoder: 0 → Negative, 1 → Positive
    class_labels   = le.classes_          # e.g. ['Negative', 'Positive']
    confidence     = pred_proba[pred_encoded] * 100

    st.subheader("Prediction Result")

    if pred_label == "Positive":
        st.error(f"### 🔴 Diabetes Status: **Positive**")
        st.warning(
            "The model predicts a **positive** result for diabetes. "
            "Please consult a healthcare professional for a proper diagnosis."
        )
    else:
        st.success(f"### 🟢 Diabetes Status: **Negative**")
        st.info(
            "The model predicts a **negative** result for diabetes. "
            "Continue maintaining a healthy lifestyle."
        )

    st.markdown(f"**Model confidence:** {confidence:.1f}%")

    # Probability breakdown
    st.subheader("Prediction Probabilities")
    proba_df = pd.DataFrame(
        {"Status": class_labels, "Probability (%)": (pred_proba * 100).round(2)}
    )
    st.bar_chart(proba_df.set_index("Status"))

    # Input summary
    with st.expander("View submitted values"):
        st.dataframe(features.T.rename(columns={0: "Value"}), use_container_width=True)

# ── Feature importance sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About this model")
    st.markdown(
        """
**Algorithm:** Random Forest Classifier  
**Estimators:** 300 trees  
**Max depth:** 3  
**Training accuracy:** ~84 % 

**Features used:**
| Feature | Importance |
|---|---|
| Fasting Blood Glucose | 38.3 % |
| Postprandial Blood Glucose | 25.9 % |
| HbA1c | 24.3 % |
| Random Blood Glucose | 10.8 % |
| BMI | 0.7 % |

*This tool is for informational purposes only and is not a substitute for professional medical advice.*
        """
    )
