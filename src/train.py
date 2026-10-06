from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, RandomizedSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

from features import clean_data, build_pipeline

ROOT = Path(__file__).resolve().parent.parent
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)

# 1. Load and clean
df = clean_data(pd.read_csv(ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
y = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn", "customerID"])

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Models
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                             scale_pos_weight=neg / pos, eval_metric="logloss",
                             random_state=42),
}

# 4. Compare with 5-fold stratified CV (training data only)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
rows = []
for name, model in models.items():
    scores = cross_validate(build_pipeline(model), X_train, y_train, cv=cv, scoring=metrics)
    rows.append({"model": name, **{m: scores[f"test_{m}"].mean() for m in metrics}})
results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).round(3)
print(results.to_string(index=False))
results.to_csv(ROOT / "reports/model_comparison.csv", index=False)

# 5. Tune XGBoost
param_dist = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [3, 4, 5, 6],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.7, 0.85, 1.0],
}
search = RandomizedSearchCV(build_pipeline(models["XGBoost"]), param_dist, n_iter=20,
                            scoring="roc_auc", cv=cv, random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
best = search.best_estimator_
print("Best params:", search.best_params_)
print("Best CV ROC-AUC:", round(search.best_score_, 3))

# 6. Final evaluation on the untouched test set (once)
proba = best.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
print("\nTEST ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
print(classification_report(y_test, pred, target_names=["Stay", "Churn"]))

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stay", "Churn"])
plt.title("Confusion matrix (test set)")
plt.savefig(ROOT / "reports/figures/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

RocCurveDisplay.from_predictions(y_test, proba)
plt.title("ROC curve (test set)")
plt.savefig(ROOT / "reports/figures/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# 7. Feature importance (permutation: shuffle one column, measure the score drop)
imp = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                             n_repeats=5, random_state=42, n_jobs=-1)
(pd.Series(imp.importances_mean, index=X_test.columns)
   .sort_values().tail(10)
   .plot(kind="barh", figsize=(8, 5), title="Top 10 features (permutation importance)"))
plt.xlabel("Drop in ROC-AUC when shuffled")
plt.savefig(ROOT / "reports/figures/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# 8. Save the full pipeline
joblib.dump(best, ROOT / "models/churn_pipeline.joblib")
print("\nModel saved to models/churn_pipeline.joblib")