import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Load data
FEATURES_CSV = "outputs/engineered_features.csv"
try:
    df = pd.read_csv(FEATURES_CSV)
except Exception as e:
    print(f"Error loading {FEATURES_CSV}: {e}")
    exit(1)

# 1. Class Distribution Bar Chart
plt.figure(figsize=(8, 6))
class_counts = df['target'].value_counts().sort_index()
labels = ['Away Win (0)', 'Draw (1)', 'Home Win (2)']
sns.barplot(x=labels, y=class_counts.values, palette='viridis')
plt.title('Distribution of Match Outcomes (10 Seasons)')
plt.ylabel('Number of Matches')
plt.xlabel('Match Outcome')
# Add counts on top
for index, value in enumerate(class_counts.values):
    plt.text(index, value + 20, str(value), ha='center', va='bottom')
plt.tight_layout()
plt.savefig("outputs/eda_class_distribution.png", dpi=300)
print("Saved outputs/eda_class_distribution.png")

# 2. Correlation Heatmap
plt.figure(figsize=(10, 8))
# Select key features
cols_to_plot = [
    'target', 'home_elo', 'away_elo', 
    'home_pts_avg', 'away_pts_avg',
    'home_gf_avg', 'away_gf_avg',
    'odds_imp_home', 'odds_imp_draw', 'odds_imp_away'
]
# Filter to existing columns
cols_to_plot = [c for c in cols_to_plot if c in df.columns]
corr_matrix = df[cols_to_plot].corr()

# Mask upper triangle
import numpy as np
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, cbar_kws={'shrink': .8})
plt.title('Correlation Heatmap of Key Features vs Target')
plt.tight_layout()
plt.savefig("outputs/eda_correlation_heatmap.png", dpi=300)
print("Saved outputs/eda_correlation_heatmap.png")

# 3. ELO Progression for a few top teams
# Since we only have pre-match engineered features, we can track a team's ELO whenever they play
plt.figure(figsize=(10, 6))
top_teams = ['Arsenal', 'Man City', 'Liverpool'] # Examples
for team in top_teams:
    # Filter rows where the team played
    team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
    if not team_matches.empty:
        # Convert date
        team_matches['Date'] = pd.to_datetime(team_matches['Date'])
        team_matches = team_matches.sort_values('Date')
        
        # Get ELO for that match (home_elo if home, away_elo if away)
        elos = []
        for _, row in team_matches.iterrows():
            if row['HomeTeam'] == team:
                elos.append(row['home_elo'])
            else:
                elos.append(row['away_elo'])
        
        plt.plot(team_matches['Date'], elos, label=team, alpha=0.7, linewidth=2)

plt.title('ELO Rating Progression Over Time')
plt.ylabel('ELO Rating')
plt.xlabel('Date')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/eda_elo_progression.png", dpi=300)
print("Saved outputs/eda_elo_progression.png")

print("EDA plots generation complete.")
