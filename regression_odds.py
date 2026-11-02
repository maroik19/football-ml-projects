"""
Predicting home win odds from team strength (Elo) — regression study.

Dataset: Club Football Match Data 2000-2025 (Kaggle, adamgbor).
Filtered to the top-5 European leagues, using pre-match features only
(no in-match stats — those would be data leakage).

Target: OddHome (average market home-win odds).
Main feature: EloDiff = HomeElo - AwayElo (strength gap drives the price).

Models compared: Simple Linear, Polynomial, SVR, Decision Tree,
Random Forest, plus a multiple-feature version adding recent form.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# ----------------------------------------------------------------------
# Helper: plot a fitted model as a smooth curve over the data cloud.
# predict_fn takes a DataFrame of EloDiff values and returns predictions.
# ----------------------------------------------------------------------
def plot_fit(X, y, predict_fn, title):
    grid = pd.DataFrame(
        np.arange(X['EloDiff'].min(), X['EloDiff'].max(), 1),
        columns=['EloDiff']
    )
    plt.scatter(X, y, s=2, alpha=0.3, color='red')
    plt.plot(grid, predict_fn(grid), color='blue')
    plt.xlabel('EloDiff (HomeElo - AwayElo)')
    plt.ylabel('OddHome')
    plt.title(title)
    plt.show()


# ======================================================================
# 1. LOAD & CLEAN
# ======================================================================
df = pd.read_csv('data/Matches.csv')
df['EloDiff'] = df['HomeElo'] - df['AwayElo']

# Elo is only tracked for ~top European teams, so ~38% of rows have no
# Elo. Restricting to the top-5 leagues removes that structural gap
# instead of dropping two-thirds of the data blindly.
top_leagues = ['E0', 'SP1', 'D1', 'I1', 'F1']
df_top = df[df['Division'].isin(top_leagues)]

# Only a handful of rows still miss Elo/odds (newly promoted teams) — drop them.
df_clean = df_top.dropna(subset=['EloDiff', 'OddHome'])
print(f"Rows after cleaning: {len(df_clean)}")

# The relationship is a curve, not a line — steep on the left (weak home
# side, odds shoot up), flattening toward 1.0 on the right (odds floor).
plt.scatter(df_clean['EloDiff'], df_clean['OddHome'], s=2, alpha=0.3)
plt.xlabel('EloDiff (HomeElo - AwayElo)')
plt.ylabel('OddHome')
plt.title('Home odds vs Elo difference')
plt.show()


# ======================================================================
# 2. SINGLE-FEATURE MODELS  (EloDiff -> OddHome)
# ======================================================================
X = df_clean[['EloDiff']]
y = df_clean['OddHome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scores = {}   # collect test R2 per model for the final table


# --- Simple Linear -----------------------------------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
scores['Simple Linear'] = r2_score(y_test, y_pred_lin)
plot_fit(X, y, lin_reg.predict, 'Simple Linear')


# --- Polynomial (degree sweep 2/3/4) -----------------------------------
# A straight line can't bend into this curve; polynomial features let it.
poly_results = []
for d in [2, 3, 4]:
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    poly_results.append({
        'degree': d,
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    })
poly_table = pd.DataFrame(poly_results)
print("\nPolynomial degree sweep:\n", poly_table)

# deg2 underfits, deg4 adds nothing over deg3 -> pick degree 3 (simplest
# that captures the curve). This is the single-feature winner.
poly_reg = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
poly_reg.fit(X_train, y_train)
scores['Polynomial (deg3)'] = r2_score(y_test, poly_reg.predict(X_test))
plot_fit(X, y, poly_reg.predict, 'Polynomial degree 3')


# --- SVR (needs feature scaling — it works on distances) ---------------
# Scale X and y; fit only on train (scaler must not see the test set).
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_sc = sc_X.fit_transform(X_train)
y_train_sc = sc_y.fit_transform(y_train.values.reshape(-1, 1))

svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_train_sc, y_train_sc.ravel())

# Predict: scale input in, run model, inverse-scale the output back to odds.
def svr_predict(df_in):
    scaled = svr_reg.predict(sc_X.transform(df_in)).reshape(-1, 1)
    return sc_y.inverse_transform(scaled)

y_pred_svr = svr_predict(X_test)
scores['SVR (rbf)'] = r2_score(y_test, y_pred_svr)
plot_fit(X, y, svr_predict, 'SVR (rbf)')


# --- Decision Tree (no scaling needed) ---------------------------------
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
scores['Decision Tree'] = r2_score(y_test, tree_reg.predict(X_test))
# Overfitting check: train R2 near-perfect while test R2 is much lower.
print(f"\nDecision Tree  train R2: {r2_score(y_train, tree_reg.predict(X_train)):.3f}"
      f"  test R2: {scores['Decision Tree']:.3f}")
plot_fit(X, y, tree_reg.predict, 'Decision Tree (note the step pattern)')


# --- Random Forest (averages many trees -> less overfitting) -----------
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(X_train, y_train)
scores['Random Forest'] = r2_score(y_test, forest_reg.predict(X_test))
print(f"Random Forest  train R2: {r2_score(y_train, forest_reg.predict(X_train)):.3f}"
      f"  test R2: {scores['Random Forest']:.3f}")
plot_fit(X, y, forest_reg.predict, 'Random Forest')


# ======================================================================
# 3. MULTIPLE FEATURES  (add recent form)
# ======================================================================
feature_cols = ['EloDiff', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']
mult_df = df_top[feature_cols + ['OddHome']].dropna()

X_m = mult_df[feature_cols]
y_m = mult_df['OddHome']
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_m, y_m, test_size=0.2, random_state=42
)

# Multiple linear — form barely helps over EloDiff alone, because Elo
# already absorbs recent form (it updates after every result). The Form
# coefficients also fight each other (multicollinearity: Form3 vs Form5
# say almost the same thing).
mult_lin = LinearRegression()
mult_lin.fit(X_train_m, y_train_m)
scores['Multiple Linear'] = r2_score(y_test_m, mult_lin.predict(X_test_m))
print("\nMultiple-linear coefficients:")
print(pd.Series(mult_lin.coef_, index=feature_cols))

# Multiple polynomial — curvature + all features together.
mult_poly_results = []
for d in [2, 3, 4]:
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(X_train_m, y_train_m)
    y_pred = model.predict(X_test_m)
    mult_poly_results.append({
        'degree': d,
        'r2': r2_score(y_test_m, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test_m, y_pred)),
    })
print("\nMultiple polynomial degree sweep:\n", pd.DataFrame(mult_poly_results))
# Best multiple-poly ~0.808 — essentially tied with the single-feature
# poly. Extra features don't move the ceiling: the remaining ~19% is
# football's inherent randomness that Elo/form can't capture.
scores['Multiple Poly (deg3)'] = pd.DataFrame(mult_poly_results).set_index('degree').loc[3, 'r2']


# ======================================================================
# 4. FINAL COMPARISON
# ======================================================================
summary = pd.DataFrame(
    sorted(scores.items(), key=lambda kv: kv[1], reverse=True),
    columns=['Model', 'Test R2']
)
print("\n===== Model comparison (test R2) =====")
print(summary.to_string(index=False))