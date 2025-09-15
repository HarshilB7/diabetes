# Core libs
import numpy as np
import pandas as pd
import joblib


# Modeling
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# XGBoost
from xgboost import XGBClassifier # type: ignore

from sklearn.model_selection import cross_val_score, StratifiedKFold

# Cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Load local CSV
df = pd.read_csv("cleve_heart.csv")

# Feature and target columns provided
X_cols = ['Age', 'sex', 'chest pain type', 'Trestbps', 'cholesteral', 'fasting blood sugar',
          'resting ecg', 'max heart rate', 'exercise induced angina', 'oldpeak', 'slope',
          'number of vessels colored', 'thal']
y_col = 'healthy'

# df = df[df['race'] != '?']
df.replace('?', np.nan, inplace=True)


df.replace('sick', 1, inplace=True)

df.replace('buff', 0, inplace=True)
# df['health'] = (df[y_col] == 'buff').astype(int)

# Select features and target
X_raw = df[X_cols].copy()
y = df['healthy'].values

# Identify categorical vs numeric columns
categorical_cols = ['sex', 'chest pain type', 'Trestbps', 'resting ecg', 
                    'fasting blood sugar', 'exercise induced angina', 
                    'slope', 'thal']

numeric_cols = ['cholesteral', 'max heart rate', 
                'oldpeak', 'number of vessels colored']

# Preprocess: impute and encode categoricals; impute numeric
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", numeric_transformer, numeric_cols)
    ],
    remainder="drop"
)


rf = RandomForestClassifier(
    n_estimators=400, n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
    eval_metric="logloss", n_jobs=-1, random_state=42, tree_method="hist"
)

meta_learner = LogisticRegression(max_iter=500, solver="lbfgs")

stack = StackingClassifier(
    estimators=[("rf", rf), ("xgb", xgb)],
    final_estimator=meta_learner,
    stack_method="predict_proba",
    passthrough=False,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)

# End-to-end pipeline with preprocessing
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("stack", stack)
])

# Optional calibration wrapper
clf = CalibratedClassifierCV(estimator=model, method="isotonic", cv=3)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

print("start training...\n")
# Fit
clf.fit(X_train, y_train)

print("training completed...\n")
joblib.dump(clf, "heart_stack_calibrated2.pkl")

# Predict probabilities for positive class
y_proba = clf.predict_proba(X_test)[:, 1]

# Choose an operating threshold; 0.5 default, but consider tuning by PR curve or cost
threshold = 0.5
y_pred = (y_proba >= threshold).astype(int)

# Evaluation metrics
auroc = roc_auc_score(y_test, y_proba)
auprc = average_precision_score(y_test, y_proba)

# Confusion matrix at chosen threshold
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calibration curve data (for plotting later)
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")

# ROC curve points
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)

# Precision-Recall curve points
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)

# Print summary
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print("Confusion Matrix (threshold=0.5):")
print(cm)
print(f"TPR (Recall): {tp / (tp + fn + 1e-12):.4f}")
print(f"FPR: {fp / (fp + tn + 1e-12):.4f}")

# print(clf)



# Cross-validated AUROC
cv_scores_auroc = cross_val_score(
    clf, X_raw, y, scoring="roc_auc", cv=cv_strategy, n_jobs=-1
)

# Cross-validated AUPRC
cv_scores_auprc = cross_val_score(
    clf, X_raw, y, scoring="average_precision", cv=cv_strategy, n_jobs=-1
)

print(f"Cross-validated AUROC: {cv_scores_auroc.mean():.4f} ± {cv_scores_auroc.std():.4f}")
print(f"Cross-validated AUPRC: {cv_scores_auprc.mean():.4f} ± {cv_scores_auprc.std():.4f}")
print(f"All AUROC scores: {cv_scores_auroc}")
print(f"All AUPRC scores: {cv_scores_auprc}")

# start training...

# training completed...

# AUROC: 0.6243
# AUPRC: 0.1999
# Confusion Matrix (threshold=0.5):
# [[17659     6]
#  [ 2224    10]]
# TPR (Recall): 0.0045
# FPR: 0.0003
