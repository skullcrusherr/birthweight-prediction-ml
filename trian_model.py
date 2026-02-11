import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False


# ----------------------------
# Configuration
# ----------------------------
MODEL_DIR = "model"
DATA_PATH = "birth_weight.csv"

FEATURES = [
    "Length", "Headcirc", "Gestation", "smoker", "mage", "mnocig",
    "mheight", "mppwt", "fage", "fedyrs", "fnocig", "fheight", "mage35"
]
TARGET = "Birthweight"


def _safe_model_size_kb(path: str) -> float:
    try:
        return os.path.getsize(path) / 1024.0
    except Exception:
        return float("nan")


def _complexity(model, name: str):
    """Return a simple, presentation-friendly complexity proxy."""
    try:
        if name == "Decision Tree":
            return int(model.get_n_leaves())
        if name == "Linear Regression":
            return int(len(model.coef_))
        if name == "Random Forest":
            return int(getattr(model, "n_estimators", 0))
        if name == "XGBoost":
            return int(getattr(model, "n_estimators", 0))
        if name == "Semantic KNN (Cosine)":
            return int(getattr(model, "n_neighbors", 0))
    except Exception:
        pass
    return None


def train_models(show_plots: bool = True):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load dataset
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset file '{DATA_PATH}' not found in the current directory")
        sys.exit(1)

    # Prepare X/y
    y = data[TARGET].dropna()
    X = data[FEATURES]

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    X_imputed = X_imputed[: len(y)]  # align with y if any rows dropped in y

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        # "Semantic" model for tabular data: cosine-similarity KNN over scaled feature vectors.
        # Interpretable story for presentation: "predict by averaging the most similar pregnancy profiles".
        "Semantic KNN (Cosine)": KNeighborsRegressor(
            n_neighbors=7, metric="cosine", algorithm="brute", n_jobs=-1
        ),
    }

    if _HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
    else:
        print("Warning: xgboost is not available in this environment. Skipping XGBoost.")

    results = []

    # Train/eval
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        results.append(
            {
                "Model": name,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
            }
        )

    results_df = pd.DataFrame(results).sort_values("RMSE")
    print("\nModel Performance (lower RMSE is better):")
    print(results_df.to_string(index=False))

    # Save preprocessing
    with open(os.path.join(MODEL_DIR, "imputer.pkl"), "wb") as f:
        pickle.dump(imputer, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Save models
    save_map = {
        "Decision Tree": "dt_model.pkl",
        "Linear Regression": "lr_model.pkl",
        "Random Forest": "rf_model.pkl",
        "Semantic KNN (Cosine)": "semantic_knn_model.pkl",
        "XGBoost": "xgb_model.pkl",
    }
    for name, filename in save_map.items():
        if name in models:
            with open(os.path.join(MODEL_DIR, filename), "wb") as f:
                pickle.dump(models[name], f)

    print(f"\nSaved: imputer.pkl, scaler.pkl, and {len([k for k in save_map if k in models])} model(s) in '{MODEL_DIR}/'")

    if show_plots:
        # 1) Size + Complexity
        labels = results_df["Model"].tolist()
        sizes = []
        complexities = []
        for label in labels:
            fname = save_map.get(label)
            sizes.append(_safe_model_size_kb(os.path.join(MODEL_DIR, fname)) if fname else float('nan'))
            complexities.append(_complexity(models[label], label) or 0)

        x = np.arange(len(labels))
        width = 0.35

        fig1, ax1 = plt.subplots(figsize=(11, 6))
        ax1.bar(x - width / 2, sizes, width, label="Model Size (KB)")
        ax1.bar(x + width / 2, complexities, width, label="Complexity (proxy)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=20, ha="right")
        ax1.set_ylabel("KB / proxy units")
        ax1.set_title("Model Size & Complexity Comparison")
        ax1.legend()
        plt.tight_layout()
        plt.show()

        # 2) Error metrics
        fig2, ax2 = plt.subplots(figsize=(11, 6))
        ax2.bar(labels, results_df["RMSE"].tolist())
        ax2.set_xticklabels(labels, rotation=20, ha="right")
        ax2.set_ylabel("RMSE (Birthweight units)")
        ax2.set_title("RMSE by Model (lower is better)")
        plt.tight_layout()
        plt.show()

    return results_df


if __name__ == "__main__":
    train_models(show_plots=True)
