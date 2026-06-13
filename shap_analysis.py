import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "outputs/improved_match_model_full.joblib"
FEATURES_CSV = "outputs/engineered_features.csv"
# --------------------------------------

# Load data
df = pd.read_csv(FEATURES_CSV)

# Load model and feature names
mdata = joblib.load(MODEL_PATH)
model = mdata['model']
model_features = mdata['features']

# Filter data to the features used by the model
X = df[model_features]

# We will analyze SHAP values on a subset (e.g., the last season) to keep it manageable and fast
X_sample = X.iloc[-380:] # Last 380 matches (roughly one full season)

# Create a SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# For a multi-class model (0: Away, 1: Draw, 2: Home), shap_values is a list of arrays.
# We will focus on class 2 (Home Win) for the summary plot as it is the most common outcome.
class_index = 2 

if isinstance(shap_values, list):
    shap_values_class = shap_values[class_index]
else:
    # In some versions of SHAP/XGBoost, it returns a 3D array (samples, features, classes)
    if len(shap_values.shape) == 3:
        shap_values_class = shap_values[:, :, class_index]
    else:
        shap_values_class = shap_values

# 1. SHAP Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_class, X_sample, show=False)
plt.title(f"SHAP Summary Plot (Impact on 'Home Win' Probability)")
plt.tight_layout()
plt.savefig("outputs/shap_summary_home_win.png", dpi=300)
print("Saved outputs/shap_summary_home_win.png")
plt.clf()

# 2. SHAP Dependence Plot for the most important feature (e.g., odds_imp_home)
if 'odds_imp_home' in model_features:
    plt.figure(figsize=(8, 6))
    shap.dependence_plot('odds_imp_home', shap_values_class, X_sample, show=False)
    plt.title("SHAP Dependence Plot: Home Odds vs 'Home Win' Impact")
    plt.tight_layout()
    plt.savefig("outputs/shap_dependence_odds.png", dpi=300)
    print("Saved outputs/shap_dependence_odds.png")
    plt.clf()

if 'home_elo' in model_features:
    plt.figure(figsize=(8, 6))
    shap.dependence_plot('home_elo', shap_values_class, X_sample, show=False)
    plt.title("SHAP Dependence Plot: Home ELO vs 'Home Win' Impact")
    plt.tight_layout()
    plt.savefig("outputs/shap_dependence_elo.png", dpi=300)
    print("Saved outputs/shap_dependence_elo.png")
    plt.clf()

print("SHAP analysis complete.")
