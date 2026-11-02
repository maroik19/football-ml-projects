"""
Binary classification — predict whether the home team wins (HomeWin: 1/0).
Dataset: Club Football Match Data 2000-2025 (Kaggle, adamgbor), top-5 leagues.
Single feature: EloDiff = HomeElo - AwayElo.

Goal: compare 7 classifiers against a majority-class baseline and see which
model family fits the shape of the data.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------
df = pd.read_csv('data/Matches.csv')
df['EloDiff'] = df['HomeElo'] - df['AwayElo']

top_leagues = ['E0', 'SP1', 'D1', 'I1', 'F1']
df_top = df[df['Division'].isin(top_leagues)].copy()
df_top['HomeWin'] = (df_top['FTResult'] == 'H').astype(int)

# Drop rows with missing Elo (40 old matches) before modelling.
df_top = df_top.dropna(subset=['EloDiff'])

# Baseline = majority class. Any model must beat this to be worth anything.
print('Class balance:')
print(df_top['HomeWin'].value_counts(normalize=True))   # 0: 0.544, 1: 0.456 -> baseline 54.4%

X = df_top[['EloDiff']]   # double brackets keep X 2D
y = df_top['HomeWin']

# stratify=y keeps the 54/46 split identical in train and test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ----------------------------------------------------------------------
# Helper — one place for all metrics
# ----------------------------------------------------------------------
def evaluate(name, model, X_tr, X_te):
    """Fit, predict, and print the standard classification metrics."""
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    print(f'\n--- {name} ---')
    print(confusion_matrix(y_test, y_pred))
    print(f'accuracy : {accuracy_score(y_test, y_pred):.4f}')
    print(f'precision: {precision_score(y_test, y_pred):.4f}')
    print(f'recall   : {recall_score(y_test, y_pred):.4f}')
    return model


# Distance/weight-based models need scaling. Scaler is fit on train only
# (fit on test would leak future information into training).
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# ----------------------------------------------------------------------
# Models that NEED scaling -> run on X_*_sc
# ----------------------------------------------------------------------
evaluate('Logistic Regression',
         LogisticRegression(random_state=0), X_train_sc, X_test_sc)

evaluate('K-NN (k=5)',
         KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2), X_train_sc, X_test_sc)

evaluate('Linear SVM',
         SVC(kernel='linear', random_state=0), X_train_sc, X_test_sc)

evaluate('Kernel SVM (rbf)',
         SVC(kernel='rbf', random_state=0), X_train_sc, X_test_sc)


# ----------------------------------------------------------------------
# Models that do NOT need scaling -> run on raw X_train / X_test
# (Naive Bayes uses per-class distributions; trees split on thresholds.
#  Neither is affected by feature scale.)
# ----------------------------------------------------------------------
evaluate('Naive Bayes',
         GaussianNB(), X_train, X_test)

dt = evaluate('Decision Tree',
              DecisionTreeClassifier(criterion='entropy', random_state=0), X_train, X_test)
# Compare train vs test to expose overfitting: big gap = memorised noise.
print(f'  train acc: {accuracy_score(y_train, dt.predict(X_train)):.4f}   '
      f'test acc: {accuracy_score(y_test, dt.predict(X_test)):.4f}')

rf = evaluate('Random Forest (100)',
              RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
              X_train, X_test)
print(f'  train acc: {accuracy_score(y_train, rf.predict(X_train)):.4f}   '
      f'test acc: {accuracy_score(y_test, rf.predict(X_test)):.4f}')


# ======================================================================
# RESULTS
# ======================================================================
# Model              Test Acc  Precision  Recall   Note
# Logistic Reg.       0.638     0.625      0.516    best; clean linear fit
# Kernel SVM (rbf)    0.637     0.638      0.475    curve = straight -> linear
# Naive Bayes         0.637     0.627      0.508    same ceiling
# Linear SVM          0.635     0.630      0.485    ~ Logistic
# K-NN                0.595     0.561      0.521    locality = noise here
# Decision Tree       0.572     0.536      0.470    overfit (train 0.884)
# Random Forest       0.572     0.532      0.527    overfit; 1 feature disables
#                                                   its feature-randomness
# baseline (always 0) 0.544                         majority class
#
# Takeaways
# 1. The problem is linear: the four strongest models all land on ~0.637.
#    When very different algorithms agree, the ceiling is real, not a
#    modelling choice.
# 2. Flexibility is punished: KNN and the trees, the most flexible models,
#    did worst — on a smooth monotonic relationship, flexibility just fits
#    noise.
# 3. The ceiling comes from the feature, not the algorithm. EloDiff alone
#    can't separate a home win from a draw. Beating ~0.637 needs more
#    features (recent form), not a fancier model.
# ======================================================================