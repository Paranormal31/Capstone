# two_stage_model.py
"""
Two-stage football match prediction:
Stage 1: Draw vs Not-Draw
Stage 2: Home vs Away (only if Not-Draw)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
FEATURES_CSV = "outputs/engineered_features.csv"
RANDOM_STATE = 42
# ---------------------------------------

# Load engineered features
df = pd.read_csv(FEATURES_CSV)

# Target encoding
# Original: 0=Away, 1=Draw, 2=Home
df['is_draw'] = (df['target'] == 1).astype(int)

# Features (reuse all numeric except targets)
drop_cols = ['target', 'is_draw', 'Date', 'HomeTeam', 'AwayTeam']
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y_draw = df['is_draw']
y_result = df['target']

# ---------------- ADD CLOSENESS FEATURES ----------------
# These help detect draws (balanced matches)

X['elo_diff'] = (df['home_elo'] - df['away_elo']).abs()

if 'odds_imp_home' in df.columns and 'odds_imp_away' in df.columns:
    X['odds_diff'] = (df['odds_imp_home'] - df['odds_imp_away']).abs()

X['form_diff'] = (df['home_pts_avg'] - df['away_pts_avg']).abs()
X['gf_diff'] = (df['home_gf_avg'] - df['away_gf_avg']).abs()

# Train-test split (same temporal logic as before)
split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_draw_train, y_draw_test = y_draw.iloc[:split_idx], y_draw.iloc[split_idx:]
y_result_train, y_result_test = y_result.iloc[:split_idx], y_result.iloc[split_idx:]

# ---------------- STAGE 1: Draw vs Not-Draw ----------------
draw_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    max_depth=4,
    learning_rate=0.05,
    n_estimators=300,
    random_state=RANDOM_STATE
)

draw_model.fit(X_train, y_draw_train)

draw_probs = draw_model.predict_proba(X_test)[:, 1]

DRAW_THRESHOLD = 0.35   # start here
draw_pred = (draw_probs >= DRAW_THRESHOLD).astype(int)

print("\n===== STAGE 1: DRAW DETECTION =====")
print(classification_report(y_draw_test, draw_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_draw_test, draw_pred))

# ---------------- STAGE 2: Home vs Away (Non-Draw Only) ----------------
# Filter non-draw matches
mask_train = y_result_train != 1
mask_test = y_result_test != 1

X_train_hw = X_train[mask_train]
X_test_hw = X_test[mask_test]

y_train_hw = y_result_train[mask_train]
y_test_hw = y_result_test[mask_test]

# Convert target: 2->1 (Home), 0->0 (Away)
y_train_hw = (y_train_hw == 2).astype(int)
y_test_hw = (y_test_hw == 2).astype(int)

home_away_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    max_depth=4,
    learning_rate=0.05,
    n_estimators=300,
    random_state=RANDOM_STATE
)

home_away_model.fit(X_train_hw, y_train_hw)

hw_pred = home_away_model.predict(X_test_hw)
print("\n===== STAGE 2: HOME vs AWAY =====")
print(classification_report(y_test_hw, hw_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test_hw, hw_pred))

# ---------------- COMBINED PREDICTION ----------------
final_preds = []
hw_idx = 0

for i in range(len(X_test)):
    if draw_pred[i] == 1:
        # Stage 1 predicts Draw
        final_preds.append(1)
    else:
        # Stage 1 predicts Not-Draw → use Stage 2 ONLY if actual was non-draw
        if y_result_test.iloc[i] != 1:
            final_preds.append(2 if hw_pred[hw_idx] == 1 else 0)
            hw_idx += 1
        else:
            # Fallback: if mismatch occurs, default to Home (safe choice)
            final_preds.append(2)

final_preds = np.array(final_preds)

final_preds = np.array(final_preds)

print("\n===== FINAL TWO-STAGE MODEL PERFORMANCE =====")
print("Overall accuracy:", accuracy_score(y_result_test, final_preds))
print(classification_report(y_result_test, final_preds, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_result_test, final_preds))

cm = confusion_matrix(y_result_test, final_preds)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Two-Stage Model Confusion Matrix")
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
plt.savefig("outputs/two_stage_confusion_matrix.png")
print("Saved confusion matrix plot to outputs/two_stage_confusion_matrix.png")

# Save models
joblib.dump(draw_model, "outputs/draw_detector_model.joblib")
joblib.dump(home_away_model, "outputs/home_away_model.joblib")
print("\nTwo-stage models saved to outputs/")
