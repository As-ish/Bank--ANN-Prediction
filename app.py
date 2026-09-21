import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Bank Deposit AI",
    page_icon="🏦",
    layout="wide"
)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "bank_ann_model.pkl"
PREPROCESSOR_PATH = BASE / "bank_preprocessor.pkl"
DATA_PATH = BASE / "bank.csv"
METRICS_PATH = BASE / "metrics.json"

# ============================================================
# SIMPLE MODERN STYLE
# ============================================================
st.markdown("""
<style>
.main {background:#f7f9fc;}
.block-container {padding-top:2rem; max-width:1200px;}
.hero {
    background: linear-gradient(135deg,#10233f,#1f4e79);
    padding:32px;
    border-radius:22px;
    color:white;
    margin-bottom:22px;
}
.hero h1 {font-size:42px; margin:0 0 8px 0;}
.hero p {color:#d8e4f0; margin:0; font-size:16px;}
.card {
    background:white;
    padding:20px;
    border-radius:16px;
    border:1px solid #e5eaf0;
    box-shadow:0 5px 18px rgba(20,40,70,.06);
}
.result {
    background:#10233f;
    color:white;
    padding:28px;
    border-radius:18px;
    text-align:center;
}
.result h2 {font-size:36px; margin:5px 0;}
.small {color:#6b7788; font-size:13px;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ASSETS
# ============================================================
@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "Model files missing. Run train_bank_ann_model.py first."
        )
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

try:
    model, preprocessor = load_assets()
except Exception as e:
    st.error(str(e))
    st.stop()

# ============================================================
# DATA / METRICS
# ============================================================
df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🏦 Bank Deposit AI")
    st.caption("Artificial Neural Network")
    page = st.radio(
        "Navigation",
        ["Overview", "Prediction", "Dataset", "Model Info"]
    )

# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    st.markdown("""
    <div class="hero">
        <h1>Bank Deposit Prediction</h1>
        <p>ANN-based classification system that predicts whether a customer
        is likely to subscribe to a bank term deposit.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Dataset Rows", f"{len(df):,}" if df is not None else "—")
    with c2:
        st.metric("Input Features", "16")
    with c3:
        st.metric("ANN Layers", "64 → 32 → 16")

    st.markdown("### Project Workflow")
    st.markdown("""
    **Bank Dataset → Preprocessing → One-Hot Encoding → StandardScaler → ANN → Prediction**

    The target variable is **deposit**:
    - `yes` = customer subscribes
    - `no` = customer does not subscribe
    """)

# ============================================================
# PREDICTION
# ============================================================
elif page == "Prediction":
    st.markdown("""
    <div class="hero">
        <h1>Customer Prediction</h1>
        <p>Enter customer information and let the trained ANN predict the deposit outcome.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.subheader("Customer Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            job = st.selectbox("Job", [
                "admin.", "blue-collar", "entrepreneur", "housemaid",
                "management", "retired", "self-employed", "services",
                "student", "technician", "unemployed", "unknown"
            ])
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
            education = st.selectbox(
                "Education", ["secondary", "tertiary", "primary", "unknown"]
            )
            default = st.selectbox("Credit Default", ["no", "yes"])

        with c2:
            balance = st.number_input("Account Balance", value=1000, step=100)
            housing = st.selectbox("Housing Loan", ["no", "yes"])
            loan = st.selectbox("Personal Loan", ["no", "yes"])
            contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"])
            day = st.number_input("Last Contact Day", 1, 31, 15)

        with c3:
            month = st.selectbox(
                "Last Contact Month",
                ["jan","feb","mar","apr","may","jun",
                 "jul","aug","sep","oct","nov","dec"]
            )
            duration = st.number_input("Call Duration (seconds)", 0, 5000, 300)
            campaign = st.number_input("Campaign Contacts", 1, 50, 2)
            pdays = st.number_input("Days Since Previous Contact", -1, 1000, -1)
            previous = st.number_input("Previous Contacts", 0, 100, 0)
            poutcome = st.selectbox(
                "Previous Campaign Outcome",
                ["unknown", "failure", "other", "success"]
            )

        submitted = st.form_submit_button(
            "🤖 Predict Deposit", use_container_width=True
        )

    if submitted:
        input_df = pd.DataFrame([{
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }])

        X_input = preprocessor.transform(input_df)
        prediction = int(model.predict(X_input)[0])
        probability = float(model.predict_proba(X_input)[0][1])

        if prediction == 1:
            label = "YES — Likely to Subscribe"
        else:
            label = "NO — Unlikely to Subscribe"

        st.markdown(f"""
        <div class="result">
            <div>ANN PREDICTION</div>
            <h2>{label}</h2>
            <div>Probability of subscription: <b>{probability*100:.2f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# DATASET
# ============================================================
elif page == "Dataset":
    st.title("📊 Dataset")
    if df is not None:
        st.write("First 20 rows:")
        st.dataframe(df.head(20), use_container_width=True)

        st.write("### Dataset Shape")
        st.write(f"Rows: **{df.shape[0]:,}** | Columns: **{df.shape[1]}**")

        st.write("### Target Distribution")
        st.bar_chart(df["deposit"].value_counts())

# ============================================================
# MODEL INFO
# ============================================================
else:
    st.title("🧠 ANN Model Information")
    st.markdown("""
    ### Architecture

    **Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Output(2 classes)**

    ### Training
    - Algorithm: Artificial Neural Network / MLP
    - Optimizer: Adam
    - Activation: ReLU
    - Batch size: 32
    - Maximum iterations: 300
    - Early stopping: enabled
    - Train/Test split: 70% / 30%
    - Random state: 123
    - Numeric features: StandardScaler
    - Categorical features: One-Hot Encoding
    """)

    if METRICS_PATH.exists():
        import json
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
        with c2:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

    st.info(
        "This project is for educational/demo purposes. "
        "Predictions should not be treated as financial advice."
    )
