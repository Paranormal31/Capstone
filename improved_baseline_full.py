"""
improved_baseline_full.py
Complete script: feature engineering (dynamic stats), ELO, odds -> model -> outputs.
Run inside project folder (with .venv activated).
"""

import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
DATA_GLOB = r"Premiere League CSV\*.csv"
ROLLING_N = 5
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "improved_match_model_full.joblib")
FEATURES_CSV = os.path.join(OUTPUT_DIR, "engineered_features.csv")
PLOT_FEAT_IMPORTANCE = os.path.join(OUTPUT_DIR, "feature_importance.png")
PLOT_CONF_MATRIX = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
USE_XGBOOST = True
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ----------------------------

# ---------- Helpers ----------
def safe_read_csv(file):
    df = pd.read_csv(file, encoding='latin1')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        if df['Date'].isna().any():
            mask = df['Date'].isna()
            df.loc[mask, 'Date'] = pd.to_datetime(df.loc[mask, 'Date'].astype(str), errors='coerce')
    return df

def find_odds_triplet(cols):
    candidates = []
    for c in cols:
        if c.endswith('H'):
            prefix = c[:-1]
            h = prefix + 'H'; d = prefix + 'D'; a = prefix + 'A'
            if h in cols and d in cols and a in cols:
                candidates.append((h,d,a))
    for t in candidates:
        if t[0].startswith('B365'):
            return t
    return candidates[0] if candidates else None

def odds_to_implied_probs(h, d, a):
    try:
        inv = np.array([1.0/float(h), 1.0/float(d), 1.0/float(a)])
    except Exception:
        return [np.nan, np.nan, np.nan]
    s = inv.sum()
    if s == 0:
        return [np.nan, np.nan, np.nan]
    return (inv / s).tolist()

def elo_update(rating_a, rating_b, score_a, k=20):
    qa = 10 ** (rating_a / 400.0)
    qb = 10 ** (rating_b / 400.0)
    ea = qa / (qa + qb)
    eb = qb / (qa + qb)
    ra = rating_a + k * (score_a - ea)
    rb = rating_b + k * ((1 - score_a) - eb)
    return ra, rb

# ---------- Load & merge ----------
files = sorted(glob.glob(DATA_GLOB))
if not files:
    raise FileNotFoundError(f"No CSV files found for pattern: {DATA_GLOB}")

dfs = [safe_read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
print("Loaded rows:", df.shape[0], "files:", len(files))

# Keep core cols
possible_cols = list(df.columns)
cols_keep = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR']

# Detect odds triplet (B365 etc.)
odds_triplet = find_odds_triplet(possible_cols)
if odds_triplet:
    cols_keep += list(odds_triplet)

# Detect additional match-stat columns (dynamic)
common_stat_candidates = {
    'shots': [('HS','AS')],
    'shots_on_target': [('HST','AST')],
    'corners': [('HC','AC')],
    'fouls': [('HF','AF')],
    'yellow_cards': [('HY','AY')],
    'red_cards': [('HR','AR')],
    'possession': [('HP','AP'),('HomePoss','AwayPoss'),('PossH','PossA'),('HPoss','APoss')]
}
available_stats = {}
for stat, candidates in common_stat_candidates.items():
    for hc, ac in candidates:
        if hc in possible_cols and ac in possible_cols:
            available_stats[stat] = (hc, ac)
            cols_keep += [hc, ac]
            break

# Filter df to columns we will read (safe)
cols_keep = [c for c in cols_keep if c in df.columns]
df = df[cols_keep].copy()
print("Columns kept:", df.columns.tolist())

# Sort chronologically
df = df.sort_values('Date', ascending=True).reset_index(drop=True)

# ---------- Feature engineering: rolling stats, ELO, odds ----------
team_history = defaultdict(lambda: deque(maxlen=ROLLING_N))         # stores (gf,ga,pts)
team_stat_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=ROLLING_N)))
team_elo = defaultdict(lambda: 1500.0)

rows = []
for idx, r in df.iterrows():
    home = r['HomeTeam']; away = r['AwayTeam']

    # basic rolling for goals/points
    def get_basic(team):
        h = team_history[team]
        if not h:
            return {'gf_avg':0.0,'ga_avg':0.0,'pts_avg':0.0,'matches':0}
        arr = np.array(h, dtype=float)
        return {'gf_avg':arr[:,0].mean(),'ga_avg':arr[:,1].mean(),'pts_avg':arr[:,2].mean(),'matches':len(h)}

    home_stats = get_basic(home); away_stats = get_basic(away)
    home_elo = team_elo[home]; away_elo = team_elo[away]

    feat = {
        'Date': r['Date'],
        'HomeTeam': home,
        'AwayTeam': away,
        'home_gf_avg': home_stats['gf_avg'],
        'home_ga_avg': home_stats['ga_avg'],
        'home_pts_avg': home_stats['pts_avg'],
        'home_recent_matches': home_stats['matches'],
        'away_gf_avg': away_stats['gf_avg'],
        'away_ga_avg': away_stats['ga_avg'],
        'away_pts_avg': away_stats['pts_avg'],
        'away_recent_matches': away_stats['matches'],
        'home_elo': home_elo,
        'away_elo': away_elo,
        'home_advantage': 1
    }

    # dynamic stat rolling features
    for stat, (hc, ac) in available_stats.items():
        def get_team_stat_roll(team, stat_name):
            q = team_stat_history[team][stat_name]
            if not q:
                return {'avg':0.0,'matches':0}
            arr = np.array(q, dtype=float)
            return {'avg': arr.mean(), 'matches': len(q)}
        h_roll = get_team_stat_roll(home, stat)
        a_roll = get_team_stat_roll(away, stat)
        feat[f'home_{stat}_avg'] = h_roll['avg']
        feat[f'away_{stat}_avg'] = a_roll['avg']
        feat[f'{stat}_diff'] = h_roll['avg'] - a_roll['avg']
        feat[f'home_{stat}_matches'] = h_roll['matches']
        feat[f'away_{stat}_matches'] = a_roll['matches']

    # odds implied probabilities
    if odds_triplet:
        hcol,dcol,acol = odds_triplet
        hp, dp, ap = odds_to_implied_probs(r.get(hcol), r.get(dcol), r.get(acol))
        feat['odds_imp_home'] = hp
        feat['odds_imp_draw'] = dp
        feat['odds_imp_away'] = ap

    # target
    ftr = r.get('FTR', np.nan)
    feat['target'] = {'H':2,'D':1,'A':0}.get(ftr, np.nan)

    rows.append(feat)

    # AFTER computing pre-match features -> update histories using actual match stats
    if pd.notnull(r.get('FTHG')) and pd.notnull(r.get('FTAG')):
        fthg = float(r['FTHG']); ftag = float(r['FTAG'])
        if fthg > ftag:
            home_pts, away_pts, score_a = 3,0,1.0
        elif fthg == ftag:
            home_pts, away_pts, score_a = 1,1,0.5
        else:
            home_pts, away_pts, score_a = 0,3,0.0

        team_history[home].append((fthg, ftag, home_pts))
        team_history[away].append((ftag, fthg, away_pts))

        # update ELO
        new_home_elo, new_away_elo = elo_update(team_elo[home], team_elo[away], score_a, k=20)
        team_elo[home] = new_home_elo
        team_elo[away] = new_away_elo

        # update stat histories
        for stat, (hc, ac) in available_stats.items():
            try:
                h_val = float(r.get(hc)) if pd.notnull(r.get(hc)) else np.nan
                a_val = float(r.get(ac)) if pd.notnull(r.get(ac)) else np.nan
            except Exception:
                h_val, a_val = np.nan, np.nan
            if not np.isnan(h_val):
                team_stat_history[home][stat].append(h_val)
            if not np.isnan(a_val):
                team_stat_history[away][stat].append(a_val)

# Build features DataFrame
feat_df = pd.DataFrame(rows).dropna(subset=['target']).reset_index(drop=True)

# Impute numeric NAs with median
for c in feat_df.columns:
    if feat_df[c].isnull().any() and feat_df[c].dtype.kind in 'fiu':
        feat_df[c] = feat_df[c].fillna(feat_df[c].median())

feat_df.to_csv(FEATURES_CSV, index=False)
print("Saved engineered features to", FEATURES_CSV)

# ---------- Prepare model matrix ----------
model_features = [
    'home_gf_avg','home_ga_avg','home_pts_avg',
    'away_gf_avg','away_ga_avg','away_pts_avg',
    'home_recent_matches','away_recent_matches',
    'home_elo','away_elo','home_advantage'
]
# add dynamic stat features
for stat in available_stats.keys():
    model_features += [f'home_{stat}_avg', f'away_{stat}_avg', f'{stat}_diff']
# add odds if present
if 'odds_imp_home' in feat_df.columns:
    model_features += ['odds_imp_home','odds_imp_draw','odds_imp_away']

model_features = [c for c in model_features if c in feat_df.columns]
X = feat_df[model_features].fillna(0)
y = feat_df['target'].astype(int)

print("X shape, y shape:", X.shape, y.shape)
print("Class distribution:\n", y.value_counts())

# Time split
split_idx = int(0.8 * len(feat_df))
X_train = X.iloc[:split_idx,:]; X_test = X.iloc[split_idx:,:]
y_train = y.iloc[:split_idx]; y_test = y.iloc[split_idx:]

# ---------- Model train (try XGBoost else RandomForest) ----------
model = None

# Define class weights (penalize Draws more)
class_weight_map = {0: 1.0, 1: 2.5, 2: 1.0}  # Away, Draw, Home
sample_weights = y_train.map(class_weight_map)

if USE_XGBOOST:
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            eval_metric='mlogloss',
            random_state=RANDOM_STATE,
            use_label_encoder=False
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        print("Using XGBoost with draw-weighted loss.")
    except Exception as e:
        print("XGBoost import failed, falling back:", e)

if model is None:
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print("Using RandomForest with class_weight='balanced'.")

# compute sample weights for multiclass balanced training
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
cw_map = {c:w for c,w in zip(classes, class_weights)}
sample_weight = y_train.map(cw_map).values

# fit
model.fit(X_train, y_train, sample_weight=sample_weight if hasattr(model, "fit") else None)

# ---------- Evaluation ----------
y_pred = model.predict(X_test)
# ==========================
# ERROR ANALYSIS (STEP 3)
# ==========================

# Create results dataframe
results_df = X_test.copy()
results_df['actual'] = y_test.values
results_df['predicted'] = y_pred

# Label mapping for readability
label_map = {0: 'Away', 1: 'Draw', 2: 'Home'}
results_df['actual_label'] = results_df['actual'].map(label_map)
results_df['predicted_label'] = results_df['predicted'].map(label_map)

# Identify misclassified matches
errors_df = results_df[results_df['actual'] != results_df['predicted']]

print("\n===== ERROR ANALYSIS =====")
print("Total test matches:", len(results_df))
print("Misclassified matches:", len(errors_df))

print("\nErrors by ACTUAL class:")
print(errors_df['actual_label'].value_counts())

print("\nHow DRAWS are misclassified:")
draw_errors = errors_df[errors_df['actual_label'] == 'Draw']
print(draw_errors['predicted_label'].value_counts())

y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)
if y_prob is not None:
    try:
        print("Log loss:", log_loss(y_test, y_prob))
    except Exception:
        pass

# ---------- Feature importance plot ----------
try:
    feat_names = X_train.columns
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(8,6))
        plt.barh(feat_names[idx], importances[idx])
        plt.title("Feature importances")
        plt.tight_layout()
        plt.savefig(PLOT_FEAT_IMPORTANCE)
        print("Saved feature importance plot to", PLOT_FEAT_IMPORTANCE)
except Exception as e:
    print("Feature importance plotting failed:", e)

# ---------- Confusion matrix plot ----------
try:
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
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
    plt.savefig(PLOT_CONF_MATRIX)
    print("Saved confusion matrix plot to", PLOT_CONF_MATRIX)
except Exception as e:
    print("Confusion matrix plotting failed:", e)

# Save model + feature list
joblib.dump({'model': model, 'features': model_features}, MODEL_PATH)
print("Model saved to", MODEL_PATH)

# Naive baseline
naive_pred = np.full_like(y_test.values, 2)
print("Naive (always-home) accuracy:", accuracy_score(y_test, naive_pred))
