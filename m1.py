# Core libs
import numpy as np
import pandas as pd

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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# XGBoost
from xgboost import XGBClassifier

# Example: X, y ready as numpy arrays or pandas DataFrame/Series
# X: features aggregated over 30–180 days per patient
# y: binary label: 1 if deterioration within 90 days else 0

X=['race', 'gender', 'age', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
   'number_emergency', 'number_inpatient','max_glu_serum', 
   'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 
   'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
   'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
   'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed']

y = ['readmitted']

# Split train/test with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Optional: scale only for linear meta-learner stability (RF/XGB don’t require scaling).
# We’ll place scaler inside the meta-learner pipeline only.
meta_learner = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=False)),
    ("lr", LogisticRegression(max_iter=500, solver="lbfgs"))
])

# Base learners
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42,
    class_weight=None
)

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42,
    tree_method="hist"
)

# Stacking configuration: use out-of-fold predictions for meta features via cv
stack = StackingClassifier(
    estimators=[("rf", rf), ("xgb", xgb)],
    final_estimator=meta_learner,
    stack_method="predict_proba", 
    passthrough=False, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)

# Optional probability calibration on top of the stack
calibrated_stack = CalibratedClassifierCV(
    base_estimator=stack,
    method="sigmoid", 
    cv=3
)

# Train
calibrated_stack.fit(X_train, y_train)

# Predict probabilities for positive class
y_proba = calibrated_stack.predict_proba(X_test)[:, 1]

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

# If needed, return artifacts for downstream plots/dashboards
evaluation_artifacts = {
    "auroc": auroc,
    "auprc": auprc,
    "confusion_matrix": cm,
    "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
    "pr_curve": {"precision": precision, "recall": recall, "thresholds": pr_thresholds},
    "calibration_curve": {"prob_true": prob_true, "prob_pred": prob_pred},
}

