import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import joblib

# ---------------- CONFIG ----------------
FEATURES_CSV = "outputs/engineered_features.csv"
RANDOM_STATE = 42
# --------------------------------------

# Load data
df = pd.read_csv(FEATURES_CSV)

# Features
drop_cols = ['target', 'Date', 'HomeTeam', 'AwayTeam']
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df['target']   # 0=Away, 1=Draw, 2=Home

# Time-based split (same as baseline to maintain consistency)
split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Original Training Class Distribution:\n{y_train.value_counts()}")

# Apply SMOTE to the training set only
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"SMOTE Training Class Distribution:\n{y_train_smote.value_counts()}")

# Train XGBoost on the SMOTE dataset
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=RANDOM_STATE,
    use_label_encoder=False
)

model.fit(X_train_smote, y_train_smote)

# Evaluation
y_pred = model.predict(X_test)

print("\n===== SMOTE MODEL PERFORMANCE =====")
print("Overall accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Save confusion matrix plot
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("SMOTE Model Confusion Matrix")
plt.colorbar()
labels = ['Away(0)','Draw(1)','Home(2)']
plt.xticks(np.arange(len(labels)), labels, rotation=45)
plt.yticks(np.arange(len(labels)), labels)
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label'); plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig("outputs/smote_confusion_matrix.png", dpi=300)
print("Saved outputs/smote_confusion_matrix.png")
