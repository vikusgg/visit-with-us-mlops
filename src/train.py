import pandas as pd
import joblib
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

# -----------------------------
# 1️⃣ Load Data from Hugging Face
# -----------------------------

dataset = load_dataset("vikusg/visit-with-us-wellness-data1")

train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

X_train = train_df.drop("ProdTaken", axis=1)
y_train = train_df["ProdTaken"]

X_test = test_df.drop("ProdTaken", axis=1)
y_test = test_df["ProdTaken"]

# -----------------------------
# 2️⃣ Define Model & Parameters
# -----------------------------

model = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# -----------------------------
# 3️⃣ Hyperparameter Tuning
# -----------------------------

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)

# -----------------------------
# 4️⃣ Model Evaluation
# -----------------------------

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

print("Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)

# -----------------------------
# 5️⃣ Save Model
# -----------------------------

joblib.dump(best_model, "wellness_model.pkl")

print("Model saved successfully.")
