import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
import features  # noqa: F401  (joblib needs this module to rebuild the pipeline)

DATA_PATH = ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models/churn_pipeline.joblib"
FIG_DIR = ROOT / "reports/figures"

LOW, HIGH = 0.30, 0.60   # same risk bands as in Step 6

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


# ---------- Loading (cached) ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    data = features.clean_data(pd.read_csv(DATA_PATH))
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
    return data


model = load_model()
df = load_data()

# ---------- Header and sidebar ----------
st.title("📉 Customer Churn Predictor")
st.caption("Estimate how likely a telecom customer is to leave, and what to do about it.")

with st.sidebar:
    st.header("About")
    st.write("Model trained on the IBM Telco Customer Churn dataset (7,043 customers).")
    st.write("The score is a **risk score**, not an exact probability, "
             "because class weights were used during training.")
    st.write("Patterns shown are **associations**, not proven causes.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Business insights", "🧪 Model performance"])

# ================= TAB 1: PREDICT =================
with tab1:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Customer")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with c2:
        st.subheader("Contract & billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, step=1.0)
        total = monthly * tenure
        st.caption(f"Total charges are estimated as monthly x tenure = ${total:,.0f}")

    with c3:
        st.subheader("Services")
        phone = st.selectbox("Phone service", ["Yes", "No"])
        if phone == "No":
            multiple = "No phone service"
        else:
            multiple = st.selectbox("Multiple lines", ["No", "Yes"])

        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        addon_names = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                       "TechSupport", "StreamingTV", "StreamingMovies"]
        if internet == "No":
            addons = {a: "No internet service" for a in addon_names}
            st.info("No internet: add-on services are not available.")
        else:
            with st.expander("Add-on services"):
                addons = {a: st.selectbox(a, ["No", "Yes"]) for a in addon_names}

    if st.button("Predict churn risk", type="primary"):
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple, "InternetService": internet, **addons,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": total,
        }])
        score = float(model.predict_proba(row)[0, 1])

        st.divider()
        left, right = st.columns([1, 2])
        with left:
            st.metric("Churn risk score", f"{score:.0%}")
            st.progress(score)
        with right:
            if score >= HIGH:
                st.error("🔴 **High risk.** Contact this customer soon.")
            elif score >= LOW:
                st.warning("🟠 **Medium risk.** Keep monitoring this customer.")
            else:
                st.success("🟢 **Low risk.** No action needed now.")
            st.caption(f"Average churn rate in the dataset: {df['Churn'].mean():.1%}")

        st.subheader("Suggested actions")
        actions = []
        if contract == "Month-to-month":
            actions.append("Offer an incentive to move to a 1-year or 2-year contract.")
        if payment == "Electronic check":
            actions.append("Encourage automatic payment (bank transfer or credit card).")
        if internet != "No" and addons.get("TechSupport") == "No":
            actions.append("Offer a trial of tech support or online security.")
        if tenure < 12:
            actions.append("Run an onboarding check-in call: the first year is the danger zone.")
        if internet == "Fiber optic" and contract == "Month-to-month":
            actions.append("High-risk segment (fiber + month-to-month): consider a loyalty offer.")
        if not actions:
            actions.append("No specific action. Keep the current service level.")
        for a in actions:
            st.write("- " + a)
        st.caption("These actions are hypotheses based on observed patterns, not proven fixes.")

# ================= TAB 2: INSIGHTS =================
with tab2:
    k1, k2, k3 = st.columns(3)
    k1.metric("Customers", f"{len(df):,}")
    k2.metric("Overall churn rate", f"{df['Churn'].mean():.1%}")
    k3.metric("Average tenure", f"{df['tenure'].mean():.0f} months")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Churn rate by contract")
        st.bar_chart(df.groupby("Contract")["Churn"].mean())
        st.subheader("Churn rate by payment method")
        st.bar_chart(df.groupby("PaymentMethod")["Churn"].mean())
    with col_b:
        st.subheader("Churn rate by internet service")
        st.bar_chart(df.groupby("InternetService")["Churn"].mean())
        st.subheader("Churn rate by tech support")
        st.bar_chart(df.groupby("TechSupport")["Churn"].mean())

    st.info("Month-to-month contracts, short tenure, fiber optic and electronic check "
            "are associated with higher churn. These are associations, not proof of cause.")

# ================= TAB 3: MODEL PERFORMANCE =================
with tab3:
    comp = ROOT / "reports/model_comparison.csv"
    if comp.exists():
        st.subheader("Model comparison (5-fold cross-validation, training data)")
        st.dataframe(pd.read_csv(comp))

    left, right = st.columns(2)
    for col, name in [(left, "confusion_matrix.png"), (right, "roc_curve.png")]:
        if (FIG_DIR / name).exists():
            col.image(str(FIG_DIR / name))

    for name in ["feature_importance.png", "threshold_business_value.png"]:
        if (FIG_DIR / name).exists():
            st.image(str(FIG_DIR / name))

    st.caption("The test set was used only once, for the final evaluation.")