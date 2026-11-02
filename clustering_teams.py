"""
Clustering team-seasons by GoalsScored / GoalsConceded per game.
Top-5 leagues, per-match data -> aggregation per team-season -> K-Means + Hierarchical.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering


# ----------------------------------------------------------------------
# 1. Load & filter to the top-5 leagues
# ----------------------------------------------------------------------
df = pd.read_csv('data/Matches.csv')

top_leagues = ['E0', 'SP1', 'D1', 'I1', 'F1']
df_top = df[df['Division'].isin(top_leagues)].copy()


# ----------------------------------------------------------------------
# 2. Derive the season from MatchDate (football season: August -> May)
# ----------------------------------------------------------------------
df_top['MatchDate'] = pd.to_datetime(df_top['MatchDate'])

def get_season(date):
    year = date.year
    month = date.month
    if month >= 8:
        start = year          # August onwards -> season starts this year
    else:
        start = year - 1      # Jan-Jul -> season started last year
    return f"{start}-{start + 1}"

df_top["Season"] = df_top["MatchDate"].apply(get_season)


# ----------------------------------------------------------------------
# 3. Aggregate per team-season (two-pass: home view + away view)
# ----------------------------------------------------------------------
# Home view: the home team scored FTHome and conceded FTAway
df_home = df_top[["HomeTeam", "Season", "FTHome", "FTAway"]].rename(
    columns={"HomeTeam": "Team", "FTHome": "GoalsScored", "FTAway": "GoalsConceded"}
)

# Away view: the away team scored FTAway and conceded FTHome (reversed)
df_away = df_top[["AwayTeam", "Season", "FTHome", "FTAway"]].rename(
    columns={"AwayTeam": "Team", "FTHome": "GoalsConceded", "FTAway": "GoalsScored"}
)

team_matches = pd.concat([df_home, df_away], ignore_index=True)

# Per-game averages per team-season + number of games
team_season = team_matches.groupby(["Team", "Season"]).agg(
    GoalsScored=("GoalsScored", "mean"),
    GoalsConceded=("GoalsConceded", "mean"),
    Games=("GoalsScored", "count"),
)

# Drop incomplete seasons (too few games -> unreliable average)
team_season = team_season[team_season["Games"] >= 20]


# ----------------------------------------------------------------------
# 4. Features + scaling
# ----------------------------------------------------------------------
X = team_season[["GoalsScored", "GoalsConceded"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ----------------------------------------------------------------------
# 5. K-Means: elbow method to choose k
# ----------------------------------------------------------------------
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# ----------------------------------------------------------------------
# 6. K-Means with k=3 + visualisation
# ----------------------------------------------------------------------
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 1], s=10, c='red',   label='Cluster 1')
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 1], s=10, c='blue',  label='Cluster 2')
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 1], s=10, c='green', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('K-Means Clusters')
plt.xlabel('Goals Scored (scaled)')
plt.ylabel('Goals Conceded (scaled)')
plt.legend()
plt.show()


# ----------------------------------------------------------------------
# 7. Hierarchical clustering: dendrogram + agglomerative
# ----------------------------------------------------------------------
sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Teams-season')
plt.ylabel('Euclidean Distances')
plt.show()

hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X_scaled)

# (AgglomerativeClustering has no centroids -> none are plotted here)
plt.scatter(X_scaled[y_hc == 0, 0], X_scaled[y_hc == 0, 1], s=10, c='red',   label='Cluster 1')
plt.scatter(X_scaled[y_hc == 1, 0], X_scaled[y_hc == 1, 1], s=10, c='blue',  label='Cluster 2')
plt.scatter(X_scaled[y_hc == 2, 0], X_scaled[y_hc == 2, 1], s=10, c='green', label='Cluster 3')
plt.title('Hierarchical Clusters')
plt.xlabel('Goals Scored (scaled)')
plt.ylabel('Goals Conceded (scaled)')
plt.legend()
plt.show()


# ----------------------------------------------------------------------
# 8. Interpretation: cluster profiles + real teams
# ----------------------------------------------------------------------
# Keep both labelings side by side (K-Means + Hierarchical)
team_season["Cluster_kmeans"] = y_kmeans
team_season["Cluster_hc"] = y_hc
team_season = team_season.reset_index()

# Average profile per cluster (based on K-Means)
print(team_season.groupby("Cluster_kmeans")[["GoalsScored", "GoalsConceded"]].mean())

# Compare the two methods: how much K-Means and Hierarchical agree.
# (Cluster numbers are arbitrary within each method -> look for one dominant
#  cell per row, not necessarily on the diagonal.)
print(pd.crosstab(team_season["Cluster_kmeans"], team_season["Cluster_hc"]))

# Top team-seasons from the "strong teams" cluster (Cluster 1)
print(
    team_season[team_season["Cluster_kmeans"] == 1]
    .sort_values("GoalsScored", ascending=False)
    [["Team", "Season", "GoalsScored", "GoalsConceded"]]
    .head(15)
)
