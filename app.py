# ============================================================
# STEP 1: Import Libraries
# ============================================================
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score
)
import streamlit as st
from ucimlrepo import fetch_ucirepo

# ============================================================
# STEP 2: Page Configuration
# ============================================================
st.set_page_config(
    page_title="🫀 Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# ============================================================
# STEP 3: Load Model, Scaler, Feature Names
# ============================================================
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# ============================================================
# STEP 4: Load Dataset for EDA Section
# ============================================================
@st.cache_data
def load_data():
    heart   = fetch_ucirepo(id=45)
    X_raw   = heart.data.features
    y_raw   = heart.data.targets
    df      = X_raw.copy()
    df["target"] = (y_raw.values.flatten() > 0).astype(int)
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df

# ============================================================
# STEP 5: App Title
# ============================================================
st.title("🫀 Heart Disease Prediction App")
st.write("Predicts whether a patient is **at risk of heart disease** using a **Decision Tree** model trained on the UCI Heart Disease dataset.")
st.write("---")

# ============================================================
# STEP 6: Sidebar — Patient Input
# ============================================================
st.sidebar.header("🧾 Enter Patient Data")

age      = st.sidebar.slider("Age",                        20, 80, 50)
sex      = st.sidebar.selectbox("Sex",                     options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
cp       = st.sidebar.selectbox("Chest Pain Type (cp)",    options=[0, 1, 2, 3], format_func=lambda x: f"Type {x}")
trestbps = st.sidebar.slider("Resting Blood Pressure",     80, 200, 120)
chol     = st.sidebar.slider("Serum Cholesterol (mg/dl)",  100, 600, 240)
fbs      = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
restecg  = st.sidebar.selectbox("Resting ECG Results",     options=[0, 1, 2])
thalach  = st.sidebar.slider("Max Heart Rate Achieved",    60, 220, 150)
exang    = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
oldpeak  = st.sidebar.slider("ST Depression (oldpeak)",    0.0, 7.0, 1.0, step=0.1)
slope    = st.sidebar.selectbox("Slope of Peak ST Segment",options=[0, 1, 2])
ca       = st.sidebar.selectbox("No. of Major Vessels (ca)", options=[0, 1, 2, 3])
thal     = st.sidebar.selectbox("Thal",                    options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x])

# ============================================================
# STEP 7: Patient Summary Cards
# ============================================================
st.subheader("📋 Patient Summary")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Age",          age)
col2.metric("Sex",          "Male" if sex == 1 else "Female")
col3.metric("Chest Pain",   f"Type {cp}")
col4.metric("Blood Pressure", trestbps)
col5.metric("Cholesterol",  chol)

col6, col7, col8, col9, col10 = st.columns(5)
col6.metric("Max Heart Rate", thalach)
col7.metric("ST Depression",  oldpeak)
col8.metric("Vessels (ca)",   ca)
col9.metric("Exercise Angina", "Yes" if exang else "No")
col10.metric("Thal",          {1:"Normal",2:"Fixed",3:"Reversible"}[thal])

st.write("---")

# ============================================================
# STEP 8: Prediction
# ============================================================
input_data = np.array([[
    age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak, slope, ca, thal
]])

input_scaled = scaler.transform(input_data)

if st.button("🔍 Predict Heart Disease Risk"):

    prediction    = model.predict(input_scaled)[0]
    probability   = model.predict_proba(input_scaled)[0]

    risk_pct    = round(probability[1] * 100, 2)
    no_risk_pct = round(probability[0] * 100, 2)

    st.subheader("🩺 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **HIGH RISK** — This patient is likely to have **Heart Disease**.")
    else:
        st.success(f"✅ **LOW RISK** — This patient is **NOT likely** to have Heart Disease.")

    col_a, col_b = st.columns(2)
    col_a.metric("❤️ Disease Risk",    f"{risk_pct}%")
    col_b.metric("💚 No Disease Risk", f"{no_risk_pct}%")

    st.progress(int(risk_pct), text=f"Risk Level: {risk_pct}%")

st.write("---")

# ============================================================
# STEP 9: EDA Section
# ============================================================
st.subheader("📊 Exploratory Data Analysis")

with st.spinner("Loading dataset for EDA..."):
    df = load_data()

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Class Distribution",
    "🗺️ Correlation Heatmap",
    "📉 ROC Curve & Confusion Matrix",
    "🔍 Feature Importance"
])

# --- Tab 1: Class Distribution + Age KDE ---
with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["target"].value_counts().plot(
        kind="bar", ax=axes[0],
        color=["steelblue", "tomato"], edgecolor="black"
    )
    axes[0].set_title("Class Distribution")
    axes[0].set_xticklabels(["No Disease", "Disease"], rotation=0)
    axes[0].set_ylabel("Count")

    df.groupby("target")["age"].plot(kind="kde", ax=axes[1], legend=True)
    axes[1].set_title("Age Distribution by Target")
    axes[1].set_xlabel("Age")
    axes[1].legend(["No Disease", "Disease"])
    plt.tight_layout()
    st.pyplot(fig)

# --- Tab 2: Correlation Heatmap ---
with tab2:
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax2)
    ax2.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig2)

# --- Tab 3: ROC Curve + Confusion Matrix ---
with tab3:
    from sklearn.model_selection import train_test_split
    X_all = df.drop("target", axis=1)
    y_all = df["target"]
    _, X_test_e, _, y_test_e = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_test_scaled_e = scaler.transform(X_test_e)
    y_pred_e        = model.predict(X_test_scaled_e)
    y_prob_e        = model.predict_proba(X_test_scaled_e)[:, 1]

    fpr, tpr, _ = roc_curve(y_test_e, y_prob_e)
    auc_score   = roc_auc_score(y_test_e, y_prob_e)
    acc         = accuracy_score(y_test_e, y_pred_e)
    cm          = confusion_matrix(y_test_e, y_pred_e)

    st.metric("Model Accuracy", f"{acc*100:.2f}%")
    st.metric("ROC-AUC Score",  f"{auc_score:.4f}")

    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
    axes3[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc_score:.2f}")
    axes3[0].plot([0,1],[0,1], color="gray", linestyle="--")
    axes3[0].set_title("ROC Curve")
    axes3[0].set_xlabel("False Positive Rate")
    axes3[0].set_ylabel("True Positive Rate")
    axes3[0].legend(loc="lower right")
    axes3[0].grid(True, alpha=0.3)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=axes3[1], colorbar=False, cmap="Blues")
    axes3[1].set_title("Confusion Matrix")
    plt.tight_layout()
    st.pyplot(fig3)

# --- Tab 4: Feature Importance ---
with tab4:
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.barh(feat_df["Feature"], feat_df["Importance"],
             color="steelblue", edgecolor="black")
    ax4.set_title("Feature Importance — Decision Tree")
    ax4.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig4)

    st.write("**Top 3 most important features:**")
    top3 = feat_df.sort_values("Importance", ascending=False).head(3)
    for i, row in top3.iterrows():
        st.write(f"- **{row['Feature']}** → `{row['Importance']:.4f}`")

# ============================================================
# STEP 10: Footer
# ============================================================
st.write("---")
st.caption("⚕️ This app is for educational purposes only. Always consult a medical professional for diagnosis.")
