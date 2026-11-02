# Football ML Portfolio: Regression, Classification, Clustering

Three small machine learning projects on the same football dataset, built with scikit-learn. Each script is self-contained and compares several models with the reasoning explained in the comments.

## Data

**Club Football Match Data (2000-2025)** by Adam Gábor, licensed MIT.
Match results and stats come from [Football-Data.co.uk](https://www.football-data.co.uk/), Elo ratings from [ClubElo](https://www.clubelo.com/).

Dataset: https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025

The data file is not included in this repo. To run the scripts:

1. Download the dataset from Kaggle.
2. Place `Matches.csv` in the `data/` folder.

All projects use the top-5 European leagues only (`E0`, `SP1`, `D1`, `I1`, `F1`).

## Projects

### 1. Regression: `regression_odds.py`
Predict the home-win odds (`OddHome`) from team strength (`EloDiff = HomeElo - AwayElo`).

| Model | Test R² |
|---|---|
| Multiple Poly (deg 3, + form) | 0.808 |
| Polynomial (deg 3) | 0.805 |
| SVR (rbf) | 0.801 |
| Random Forest | 0.683 |
| Simple Linear | 0.575 |
| Decision Tree | 0.549 |

`EloDiff` carries almost all the signal. Adding form features barely moves the score, and the ~0.81 ceiling reflects the randomness of football. Tree models overfit (train R² ~0.95+).

### 2. Classification: `classification_homewin.py`
Predict whether the home team wins (`HomeWin`) from `EloDiff`, against a 54.4% majority-class baseline.

| Model | Test accuracy |
|---|---|
| Logistic Regression | 0.638 |
| Kernel SVM (rbf) | 0.637 |
| Naive Bayes | 0.637 |
| Linear SVM | 0.635 |
| K-NN | 0.595 |
| Decision Tree | 0.572 |
| Random Forest | 0.572 |

Four very different models land on the same ~0.637, so the ceiling comes from the single feature, not the algorithm. The most flexible models do worst (overfitting).

### 3. Clustering: `clustering_teams.py`
Group team-seasons by goals scored and conceded per game, using K-Means and Hierarchical (Ward) clustering with k=3 (elbow method + dendrogram).

The clusters read as strong teams (high scored, low conceded), mid/defensive teams and weak teams. Top-cluster team-seasons include Barcelona, Real Madrid, Bayern and PSG.

## Setup

```bash
pip install -r requirements.txt
python regression_odds.py
python classification_homewin.py
python clustering_teams.py
```

## Repo structure

```
.
├── data/              # put Matches.csv here (not tracked)
├── regression_odds.py
├── classification_homewin.py
├── clustering_teams.py
├── requirements.txt
└── README.md
```
