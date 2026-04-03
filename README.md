# 🫀 Heart Disease Prediction App

A machine learning web application built with **Streamlit** that predicts whether a patient is **at risk of heart disease** using the **UCI Heart Disease Dataset** and a **Decision Tree Classifier**.

---

## 🚀 App Features

- 🤖 **Decision Tree** classifier with tuned hyperparameters
- 📥 **Auto-downloads** dataset via `ucimlrepo` — no Kaggle account needed
- 🧾 **Patient input form** with 13 medical features in the sidebar
- 📊 **4-tab EDA dashboard:**
  - Class Distribution & Age KDE
  - Correlation Heatmap
  - ROC Curve & Confusion Matrix
  - Feature Importance Chart
- 🩺 **Risk prediction** with confidence percentage
- 📈 **ROC-AUC score** and model accuracy displayed live

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── train_model.py          # Train model, run EDA, save .pkl files
├── app.py                  # Streamlit web application
├── model.pkl               # Saved Decision Tree model (generated)
├── scaler.pkl              # Saved StandardScaler (generated)
├── feature_names.pkl       # Saved feature names list (generated)
├── eda_plot.png            # EDA charts (generated)
├── correlation_heatmap.png # Heatmap (generated)
├── evaluation_plot.png     # ROC + Confusion Matrix (generated)
├── feature_importance.png  # Feature importance chart (generated)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📊 Dataset

- **Name:** Heart Disease UCI Dataset
- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) (fetched via `ucimlrepo`)
- **Samples:** 303 patients
- **Features:** 13 medical attributes
- **Target:** `0` = No Disease, `1` = Disease (binarized from original 0–4 scale)

### Features Explained

| Feature | Description |
|---|---|
| age | Age in years |
| sex | Sex (1 = Male, 0 = Female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = True) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (1 = Yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy (0–3) |
| thal | Thalassemia type (1 = Normal, 2 = Fixed, 3 = Reversible) |

---

## 🤖 Model

- **Algorithm:** Decision Tree Classifier
- **Max Depth:** 5
- **Min Samples Split:** 10
- **Preprocessing:** StandardScaler
- **Train/Test Split:** 80% / 20% (stratified)
- **Evaluation Metrics:** Accuracy, ROC-AUC, Confusion Matrix

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train_model.py
```

Expected output:
```
📥 Downloading Heart Disease UCI Dataset...
✅ Dataset downloaded! Shape: (303, 13)
🌳 Training Decision Tree model...
✅ Model trained!

📊 Model Evaluation:
   Accuracy : 81.97%
   ROC-AUC  : 0.8854
✅ model.pkl, scaler.pkl, and feature_names.pkl saved!
```

### 4. Run the Streamlit App

```bash
python -m streamlit run app.py
```

Open browser at: `http://localhost:8501`

---

## 📦 Requirements

```
numpy
pandas
scikit-learn
streamlit
matplotlib
seaborn
ucimlrepo
```

---

## 🛠️ How It Works

```
UCI Repository API (ucimlrepo)
        ↓
Raw Patient Data (303 rows, 13 features)
        ↓
Data Cleaning (missing values → median fill)
        ↓
Feature Engineering + EDA
        ↓
StandardScaler (normalization)
        ↓
Decision Tree Classifier
        ↓
Prediction + ROC-AUC + Confusion Matrix + Feature Importance
```

---

## 🐙 Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit - Heart Disease Prediction App"
git remote add origin https://github.com/YOUR_USERNAME/heart-disease-prediction.git
git branch -M main
git push -u origin main
```

---

## ⚠️ Disclaimer

> This application is for **educational purposes only**.
> It is **not** a substitute for professional medical advice or diagnosis.
> Always consult a qualified healthcare provider.

---

## 👤 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
