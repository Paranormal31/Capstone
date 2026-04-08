# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Football Match Predictor", layout="wide")

# ---------- CONFIG - adjust if your paths differ ----------
MODEL_PATH = os.path.join("outputs", "improved_match_model_full.joblib")
FEATURES_CSV = os.path.join("outputs", "engineered_features.csv")
# ---------------------------------------------------------

st.title("⚽ AI Football Match Predictor")
st.markdown(
    """
    Predict match outcome probabilities (Home / Draw / Away) using the trained model.
    The app uses the latest engineered team stats from `engineered_features.csv`.
    """
)

# ---------- Load model and features ----------
@st.cache_data(show_spinner=True)
def load_model_and_data(model_path, features_csv):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run the training script first.")
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Features CSV not found at {features_csv}. Run the feature engineering script first.")
    data = pd.read_csv(features_csv, parse_dates=['Date'])
    mdata = joblib.load(model_path)
    model = mdata['model'] if isinstance(mdata, dict) and 'model' in mdata else mdata
    features = mdata['features'] if isinstance(mdata, dict) and 'features' in mdata else None
    return model, features, data

try:
    model, model_features, feat_df = load_model_and_data(MODEL_PATH, FEATURES_CSV)
except Exception as e:
    st.error(str(e))
    st.stop()

# ---------- Prepare team list and lookup of last-known stats ----------
teams = sorted(set(feat_df['HomeTeam']).union(set(feat_df['AwayTeam'])))
st.sidebar.header("Match selection")
home_team = st.sidebar.selectbox("Home team", teams, index=0)
away_team = st.sidebar.selectbox("Away team", [t for t in teams if t != home_team], index=1 if len(teams)>1 else 0)

# Optional date input (we will not recompute ELO on a future date; we use latest known stats)
match_date = st.sidebar.date_input("Match date (for reference)", value=pd.Timestamp.now().date())

# Helper: get last pre-match row for a team (most recent row where team was home or away)
@st.cache_data
def get_last_team_stats(df, team):
    # Find most recent row where team appears as HomeTeam or AwayTeam and return its feature snapshot
    mask = (df['HomeTeam'] == team) | (df['AwayTeam'] == team)
    team_rows = df[mask].sort_values('Date')
    if team_rows.empty:
        return None
    # last row's pre-match features are present as columns home_*/away_* etc.
    return team_rows.iloc[-1]

home_last = get_last_team_stats(feat_df, home_team)
away_last = get_last_team_stats(feat_df, away_team)

if home_last is None:
    st.error(f"No historical data found for home team: {home_team}")
    st.stop()
if away_last is None:
    st.error(f"No historical data found for away team: {away_team}")
    st.stop()

# ---------- Build feature vector for prediction ----------
st.header(f"Predicting: {home_team} (Home)  vs  {away_team} (Away)")

# We will build a feature vector by taking the 'home' stats from the last home row for home_team
# and the 'away' stats from last away row for away_team, and ELOs if present
def build_feature_vector(home_last_row, away_last_row, feature_list):
    vec = {}
    for f in feature_list:
        # If f is a home_* feature and present in home_last_row, use it
        if f in home_last_row.index and f.startswith('home_'):
            vec[f] = home_last_row.get(f, 0)
        # If f is away_* and present in away_last_row, use it
        elif f in away_last_row.index and f.startswith('away_'):
            vec[f] = away_last_row.get(f, 0)
        # special features like home_advantage/home_elo/away_elo
        elif f in home_last_row.index:
            vec[f] = home_last_row.get(f, 0)
        elif f in away_last_row.index:
            vec[f] = away_last_row.get(f, 0)
        else:
            vec[f] = 0.0
    return pd.DataFrame([vec])

X_input = build_feature_vector(home_last, away_last, model_features)

st.subheader("Input feature snapshot (used for prediction)")
st.dataframe(X_input.T.rename(columns={0: 'value'}))

# ---------- Prediction ----------
try:
    probs = model.predict_proba(X_input)[0] if hasattr(model, "predict_proba") else None
    pred_class = model.predict(X_input)[0]
    label_map = {0: "Away win", 1: "Draw", 2: "Home win"}
    if probs is not None:
        prob_map = {"Away": probs[0], "Draw": probs[1], "Home": probs[2]}
        st.metric("Predicted outcome", label_map.get(pred_class, str(pred_class)))
        st.write("Predicted probabilities:")
        st.progress(0)
        # Show probabilities nicely
        st.write(pd.DataFrame({
            'Outcome': ['Away','Draw','Home'],
            'Probability': [probs[0], probs[1], probs[2]]
        }).set_index('Outcome').style.format("{:.3f}"))
    else:
        st.write("Model does not provide probabilities. Predicted class:", label_map.get(pred_class, str(pred_class)))
except Exception as e:
    st.error("Error running prediction: " + str(e))
    st.stop()

# ---------- Explainability: show top contributing features (simple) ----------
st.subheader("Top input features (values)")

# show top N features by absolute value (quick heuristic)
top_n = st.slider("Number of features to display", 5, min(3, len(model_features)), min(10, len(model_features)))
fv = X_input.iloc[0].abs().sort_values(ascending=False).head(top_n)
orig = X_input.iloc[0][fv.index]
df_top = pd.DataFrame({'feature': fv.index, 'abs_value': fv.values, 'value': orig.values}).set_index('feature')
st.table(df_top)

st.markdown("---")
st.write("Model features used:", model_features)
st.write("Last known data dates — Home team:", home_last['Date'], "  Away team:", away_last['Date'])
st.caption("Note: For new/unseen teams or future dates the app uses the latest available engineered stats from the dataset.")
