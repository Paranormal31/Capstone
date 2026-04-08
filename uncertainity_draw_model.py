# uncertainty_draw_model.py
"""
Draw prediction using uncertainty-based post-processing.
If model confidence is low, predict Draw.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# ---------------- CONFIG ----------------
FEATURES_CSV = "outputs/engineered_features.csv"
RANDOM_STATE = 42
CONFIDENCE_THRESHOLD = 0.48  # <-- we will tune this
# --------------------------------------

# Load data
df = pd.read_csv(FEATURES_CSV)

# Features (same as baseline)
drop_cols = ['target', 'Date', 'HomeTeam', 'AwayTeam']
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df['target']   # 0=Away, 1=Draw, 2=Home

# Time-based split (same as before)
split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train standard 3-class XGBoost
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=RANDOM_STATE,
    use_label_encoder=False
)

model.fit(X_train, y_train)

# Get predicted probabilities
probs = model.predict_proba(X_test)

# Uncertainty-based prediction
final_preds = []

for p in probs:
    max_prob = np.max(p)
    if max_prob < CONFIDENCE_THRESHOLD:
        final_preds.append(1)  # Draw
    else:
        final_preds.append(np.argmax(p))

final_preds = np.array(final_preds)

# Evaluation
print("\n===== UNCERTAINTY-BASED DRAW MODEL =====")
print("Confidence threshold:", CONFIDENCE_THRESHOLD)
print("Overall accuracy:", accuracy_score(y_test, final_preds))
print(classification_report(y_test, final_preds, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, final_preds))