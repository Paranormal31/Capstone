import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
import json

# ---------------- CONFIG ----------------
FEATURES_CSV = "outputs/engineered_features.csv"
RANDOM_STATE = 42
# --------------------------------------

# Load data
df = pd.read_csv(FEATURES_CSV)

drop_cols = ['target', 'Date', 'HomeTeam', 'AwayTeam']
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df['target']   # 0=Away, 1=Draw, 2=Home

# Since this is time-series data (matches over seasons), we should ideally use TimeSeriesSplit
# instead of standard random K-Fold to prevent data leakage from future to past.
tscv = TimeSeriesSplit(n_splits=3)

# Define the base model
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=RANDOM_STATE,
    use_label_encoder=False
)

# Define the hyperparameter grid
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}

print(f"Starting Grid Search with {tscv.n_splits}-fold Time Series Cross Validation...")
print(f"Search Space: {param_grid}")

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy', # We could optimize for 'neg_log_loss' but accuracy is more interpretable for the report
    cv=tscv,
    n_jobs=-1, # Use all available cores
    verbose=1
)

# Run the search
grid_search.fit(X, y)

print("\n===== GRID SEARCH RESULTS =====")
print("Best Parameters Found:")
print(json.dumps(grid_search.best_params_, indent=4))
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# Extract results for plotting
results = pd.DataFrame(grid_search.cv_results_)

# Plot the effect of n_estimators on performance for the best learning_rate and max_depth
best_lr = grid_search.best_params_['learning_rate']
best_depth = grid_search.best_params_['max_depth']

subset = results[(results['param_learning_rate'] == best_lr) & (results['param_max_depth'] == best_depth)]
subset = subset.sort_values('param_n_estimators')

plt.figure(figsize=(8, 6))
plt.plot(subset['param_n_estimators'], subset['mean_test_score'], marker='o', linestyle='-', color='b')
plt.title(f"Accuracy vs. Number of Trees (LR={best_lr}, Depth={best_depth})")
plt.xlabel("Number of Estimators (Trees)")
plt.ylabel("Cross-Validation Accuracy")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/hyperparameter_tuning_results.png", dpi=300)
print("Saved outputs/hyperparameter_tuning_results.png")

# Save the best parameters to a text file
with open("outputs/best_hyperparameters.txt", "w") as f:
    f.write(f"Best Parameters: {grid_search.best_params_}\n")
    f.write(f"Best CV Accuracy: {grid_search.best_score_:.4f}\n")
