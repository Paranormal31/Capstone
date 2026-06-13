import pandas as pd
import joblib

# ---------------- CONFIG ----------------
MODEL_PATH = "outputs/improved_match_model_full.joblib"
FEATURES_CSV = "outputs/engineered_features.csv"
TEAM = "Arsenal"
# --------------------------------------

# Load data and model
df = pd.read_csv(FEATURES_CSV)
mdata = joblib.load(MODEL_PATH)
model = mdata['model']
model_features = mdata['features']

# Filter last season's matches for the team
df['Date'] = pd.to_datetime(df['Date'])
# Let's say last season is the last 38 matches the team played
team_df = df[(df['HomeTeam'] == TEAM) | (df['AwayTeam'] == TEAM)].sort_values('Date').tail(38)

X_team = team_df[model_features]
y_team = team_df['target']

probs = model.predict_proba(X_team)
preds = model.predict(X_team)

label_map = {0: "Away", 1: "Draw", 2: "Home"}

print(f"===== CASE STUDY: {TEAM} (Last 38 Matches) =====")

correct = 0
results = []
for i in range(len(team_df)):
    row = team_df.iloc[i]
    act = label_map[row['target']]
    prd = label_map[preds[i]]
    prob = probs[i]
    
    if act == prd:
        correct += 1
    
    match_str = f"{row['HomeTeam']} vs {row['AwayTeam']}"
    res_str = f"{match_str:30} | True: {act:4} | Pred: {prd:4} | Probs: A:{prob[0]:.2f} D:{prob[1]:.2f} H:{prob[2]:.2f}"
    results.append(res_str)

print(f"Accuracy for {TEAM}: {correct}/{len(team_df)} ({correct/len(team_df):.2f})")
print("\nRecent 10 Matches Analysis:")
for r in results[-10:]:
    print(r)

# Write to text file
with open("outputs/case_study_arsenal.txt", "w") as f:
    f.write(f"Accuracy for {TEAM}: {correct}/{len(team_df)} ({correct/len(team_df):.2f})\n\n")
    for r in results:
        f.write(r + "\n")
print("Saved outputs/case_study_arsenal.txt")
