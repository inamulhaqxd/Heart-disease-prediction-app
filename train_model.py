# ============================================================
# STEP 1: Import Libraries
# ============================================================
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

# ============================================================
# STEP 2: Download Heart Disease Dataset (UCI Repository)
# No Kaggle account needed — fetched automatically!
# ============================================================
print("📥 Downloading Heart Disease UCI Dataset...")
heart = fetch_ucirepo(id=45)   # id=45 is the Heart Disease dataset

X_raw = heart.data.features
y_raw = heart.data.targets

print("✅ Dataset downloaded!")
print(f"   Features shape : {X_raw.shape}")
print(f"   Target  shape  : {y_raw.shape}")
print(f"\nFeature columns:\n{list(X_raw.columns)}")
print(f"\nSample data:\n{X_raw.head()}")

# ============================================================
# STEP 3: Data Cleaning
# Target: 0 = No Disease, 1-4 = Disease → binarize to 0 / 1
# ============================================================
df = X_raw.copy()
df["target"] = (y_raw.values.flatten() > 0).astype(int)

print(f"\n🔍 Missing values per column:\n{df.isnull().sum()}")

# Fill missing numerical values with median
for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

print(f"\nAfter cleaning — shape: {df.shape}")
print(f"Class distribution:\n{df['target'].value_counts()}")

# ============================================================
# STEP 4: Exploratory Data Analysis (EDA)
# ============================================================
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# --- Plot 1: Class Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df["target"].value_counts().plot(
    kind="bar", ax=axes[0],
    color=["steelblue", "tomato"],
    edgecolor="black"
)
axes[0].set_title("Class Distribution", fontsize=13)
axes[0].set_xticklabels(["No Disease (0)", "Disease (1)"], rotation=0)
axes[0].set_ylabel("Count")

# --- Plot 2: Age Distribution by Target ---
df.groupby("target")["age"].plot(
    kind="kde", ax=axes[1], legend=True
)
axes[1].set_title("Age Distribution by Target", fontsize=13)
axes[1].set_xlabel("Age")
axes[1].legend(["No Disease", "Disease"])

plt.tight_layout()
plt.savefig("eda_plot.png", dpi=150)
plt.show()
print("📊 EDA plot saved as eda_plot.png")

# --- Plot 3: Correlation Heatmap ---
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()
print("🗺️  Correlation heatmap saved as correlation_heatmap.png")

# ============================================================
# STEP 5: Define Features (X) and Target (y)
# ============================================================
X = df.drop("target", axis=1)
y = df["target"]

# ============================================================
# STEP 6: Train-Test Split (80% / 20%)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ============================================================
# STEP 7: Feature Scaling
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ============================================================
# STEP 8: Train Decision Tree Classifier
# ============================================================
print("\n🌳 Training Decision Tree model...")

model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
model.fit(X_train_scaled, y_train)
print("✅ Model trained!")

# ============================================================
# STEP 9: Evaluate — Accuracy + Classification Report
# ============================================================
y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_pred_prob)

print(f"\n📊 Model Evaluation:")
print(f"   Accuracy  : {accuracy * 100:.2f}%")
print(f"   ROC-AUC   : {roc_auc:.4f}")
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# ============================================================
# STEP 10: Plot — ROC Curve + Confusion Matrix
# ============================================================
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ROC Curve
axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--")
axes[0].set_title("ROC Curve", fontsize=13)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Confusion Matrix", fontsize=13)

plt.tight_layout()
plt.savefig("evaluation_plot.png", dpi=150)
plt.show()
print("📈 Evaluation plot saved as evaluation_plot.png")

# ============================================================
# STEP 11: Feature Importance
# ============================================================
importances = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature":    list(X.columns),
    "Importance": importances
}).sort_values("Importance", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feat_df["Feature"], feat_df["Importance"], color="steelblue", edgecolor="black")
plt.title("Feature Importance — Decision Tree", fontsize=13)
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("🔍 Feature importance plot saved as feature_importance.png")

# ============================================================
# STEP 12: Save Model, Scaler, and Feature Names
# ============================================================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("\n✅ model.pkl, scaler.pkl, and feature_names.pkl saved successfully!")
