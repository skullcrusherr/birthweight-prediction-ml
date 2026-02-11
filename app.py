# app.py
import os
import sys
import json
import sqlite3
import hashlib
import base64
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# Performance + stability tweaks (prevents CPU thread stampede on Windows)
# ---------------------------------------------------------------------
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

# ---------------------------------------------------------------------
# Paths
# Your folder layout (per screenshot):
#   updated_models_project/code/app.py
#   updated_models_project/code/model/*.pkl
# ---------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
TRAIN_SCRIPT = os.path.join(APP_DIR, "trian_model.py")
DB_PATH = os.path.join(APP_DIR, "app.db")

# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Birthweight Prediction", layout="wide")

# ---------------------------------------------------------------------
# SQLite helpers (Login + Prediction History)
# ---------------------------------------------------------------------
def get_db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            salt TEXT NOT NULL,
            pw_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            prediction REAL NOT NULL,
            inputs_json TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    con.commit()
    return con

def _hash_password(password: str, salt_bytes: bytes) -> str:
    # PBKDF2-HMAC-SHA256
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 200_000)
    return base64.b64encode(dk).decode("utf-8")

def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    con = get_db()
    salt = os.urandom(16)
    salt_b64 = base64.b64encode(salt).decode("utf-8")
    pw_hash = _hash_password(password, salt)
    try:
        con.execute(
            "INSERT INTO users (username, salt, pw_hash, created_at) VALUES (?, ?, ?, ?)",
            (username, salt_b64, pw_hash, datetime.now().isoformat(timespec="seconds")),
        )
        con.commit()
        return True, "Account created. Please log in."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    finally:
        con.close()

def login_user(username: str, password: str) -> tuple[bool, str, dict | None]:
    username = username.strip()
    con = get_db()
    cur = con.execute("SELECT id, username, salt, pw_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    con.close()

    if not row:
        return False, "User not found.", None

    user_id, uname, salt_b64, pw_hash = row
    salt = base64.b64decode(salt_b64.encode("utf-8"))
    check_hash = _hash_password(password, salt)

    if check_hash != pw_hash:
        return False, "Invalid password.", None

    return True, "Logged in.", {"id": user_id, "username": uname}

def save_prediction(user_id: int, model_name: str, prediction: float, inputs: dict):
    con = get_db()
    con.execute(
        "INSERT INTO predictions (user_id, created_at, model_name, prediction, inputs_json) VALUES (?, ?, ?, ?, ?)",
        (
            user_id,
            datetime.now().isoformat(timespec="seconds"),
            model_name,
            float(prediction),
            json.dumps(inputs, ensure_ascii=False),
        ),
    )
    con.commit()
    con.close()

def load_history(user_id: int, limit: int = 200) -> pd.DataFrame:
    con = get_db()
    cur = con.execute(
        """
        SELECT created_at, model_name, prediction, inputs_json
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cur.fetchall()
    con.close()

    if not rows:
        return pd.DataFrame(columns=["created_at", "model_name", "prediction", "inputs_json"])

    df = pd.DataFrame(rows, columns=["created_at", "model_name", "prediction", "inputs_json"])
    return df

# ---------------------------------------------------------------------
# Model loading + preprocessing
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    required = [
        "imputer.pkl",
        "scaler.pkl",
        "lr_model.pkl",
        "dt_model.pkl",
        "rf_model.pkl",
        "xgb_model.pkl",
        "semantic_knn_model.pkl",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        raise FileNotFoundError(
            "Missing required model files in 'code/model/'.\n"
            f"Missing: {missing}\n\n"
            "Train once:\n"
            "  python trian_model.py"
        )

    imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    models = {
        "Linear Regression": joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl")),
        "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "dt_model.pkl")),
        "Random Forest": joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl")),
        "XGBoost": joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl")),
        "Semantic KNN (Cosine)": joblib.load(os.path.join(MODEL_DIR, "semantic_knn_model.pkl")),
    }
    return imputer, scaler, models

def preprocess_X(df_X: pd.DataFrame, imputer, scaler) -> np.ndarray:
    # Force same columns and order used in training
    if hasattr(imputer, "feature_names_in_"):
        train_features = list(imputer.feature_names_in_)
        df_X = df_X.reindex(columns=train_features)
    else:
        df_X = df_X.select_dtypes(include=[np.number])

    X_imp = imputer.transform(df_X)
    X_scaled = scaler.transform(X_imp)
    return X_scaled

def run_training_and_reload():
    with st.spinner("Training models..."):
        try:
            import subprocess
            subprocess.run([sys.executable, TRAIN_SCRIPT], check=True)
        except Exception as e:
            st.error("Training failed. Check terminal output.")
            st.exception(e)
            return False

    load_artifacts.clear()
    st.success("Training complete. Models reloaded ✅")
    return True

# ---------------------------------------------------------------------
# Dataset utilities (optional, for evaluation charts)
# ---------------------------------------------------------------------
def _find_default_dataset() -> str | None:
    patterns = [
        os.path.join(APP_DIR, "*.csv"),
        os.path.join(APP_DIR, "*.xlsx"),
        os.path.join(APP_DIR, "*.xls"),
        os.path.join(APP_DIR, "..", "*.csv"),
        os.path.join(APP_DIR, "..", "*.xlsx"),
        os.path.join(APP_DIR, "..", "*.xls"),
        os.path.join(APP_DIR, "..", "data", "*.csv"),
        os.path.join(APP_DIR, "..", "data", "*.xlsx"),
        os.path.join(APP_DIR, "..", "data", "*.xls"),
    ]
    import glob
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda f: os.path.getmtime(f), reverse=True)
    return candidates[0]

def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path, engine="openpyxl")
    if path.lower().endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type")

@st.cache_data
def evaluate_on_test(df: pd.DataFrame):
    """
    Evaluates loaded models (no retraining) on a simple 80/20 split.
    Target is fixed to Birthweight.
    """
    if "Birthweight" not in df.columns:
        raise ValueError("Dataset must contain 'Birthweight' column for evaluation.")

    imputer, scaler, models = load_artifacts()
    numeric_df = df.select_dtypes(include=[np.number]).dropna(subset=["Birthweight"])
    X = numeric_df.drop(columns=["Birthweight"])
    y = numeric_df["Birthweight"]

    # Align to training columns
    if hasattr(imputer, "feature_names_in_"):
        X = X.reindex(columns=list(imputer.feature_names_in_))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test_scaled = preprocess_X(X_test, imputer, scaler)

    rows = []
    for name, model in models.items():
        pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        rows.append({"Model": name, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2})

    return pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "Predict"

# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.title("Navigation")

if st.session_state.user is None:
    st.session_state.page = st.sidebar.radio("Go to", ["Login", "Register"], index=0)
else:
    st.sidebar.markdown(f"**Logged in as:** `{st.session_state.user['username']}`")
    st.session_state.page = st.sidebar.radio("Go to", ["Predict", "History", "Model Performance", "Settings"], index=0)
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

# ---------------------------------------------------------------------
# Login / Register pages
# ---------------------------------------------------------------------
st.title("Birthweight Prediction App")

if st.session_state.user is None:
    if st.session_state.page == "Login":
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
        if submit:
            ok, msg, user = login_user(username, password)
            if ok:
                st.session_state.user = user
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    elif st.session_state.page == "Register":
        st.subheader("Register")
        with st.form("register_form"):
            username = st.text_input("Choose a username")
            password = st.text_input("Choose a password", type="password")
            password2 = st.text_input("Confirm password", type="password")
            submit = st.form_submit_button("Create account")
        if submit:
            if password != password2:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user(username, password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    st.stop()

# ---------------------------------------------------------------------
# Logged-in pages
# ---------------------------------------------------------------------
# Load models (once)
try:
    imputer, scaler, models = load_artifacts()
except Exception as e:
    st.error("Could not load model artifacts from code/model.")
    st.exception(e)
    st.stop()

# Feature list from training artifacts
if hasattr(imputer, "feature_names_in_"):
    FEATURE_COLS = list(imputer.feature_names_in_)
else:
    FEATURE_COLS = []  # fallback later

# -----------------------
# Settings (Train button)
# -----------------------
if st.session_state.page == "Settings":
    st.subheader("Settings")
    st.write("This project uses saved `.pkl` models. It will not retrain unless you click the button below.")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Train / Refresh Models", type="primary"):
            ok = run_training_and_reload()
            if ok:
                st.rerun()

    with col2:
        st.caption("Tip (Windows): if your PC hangs, run before Streamlit:")
        st.code(
            "set OPENBLAS_NUM_THREADS=4\n"
            "set OMP_NUM_THREADS=4\n"
            "set MKL_NUM_THREADS=4\n"
            "set NUMEXPR_NUM_THREADS=4\n"
            "streamlit run app.py",
            language="bat",
        )

    st.write("Model files in:", f"`{MODEL_DIR}`")
    st.stop()

# -----------------------
# Predict page
# -----------------------
if st.session_state.page == "Predict":
    st.subheader("Predict Birthweight")
    st.caption("Target is fixed to **Birthweight**. Provide values for all input features used during training.")

    left, right = st.columns([1.2, 1])

    with left:
        selected_model_name = st.selectbox("Choose model", list(models.keys()), index=0)

        # Build input form. Use loose defaults if we don't have dataset stats.
        with st.form("predict_form"):
            inputs = {}

            # If FEATURE_COLS is empty, let user upload dataset to infer columns
            if not FEATURE_COLS:
                st.warning("Could not read training feature names from the imputer. Upload the dataset to infer columns.")
                uploaded = st.file_uploader("Upload CSV/Excel (to infer feature columns)", type=["csv", "xlsx", "xls"])
                if uploaded is not None:
                    if uploaded.name.lower().endswith(".csv"):
                        df = pd.read_csv(uploaded)
                    elif uploaded.name.lower().endswith(".xlsx"):
                        df = pd.read_excel(uploaded, engine="openpyxl")
                    else:
                        df = pd.read_excel(uploaded)
                    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Birthweight"]
                    FEATURE_COLS = numeric_cols
                else:
                    st.stop()

            for col in FEATURE_COLS:
                # Heuristic: binary-ish gets a selectbox, else number_input
                default_val = 0.0
                if col.lower() in {"smoker", "lowbwt", "mage35"}:
                    inputs[col] = st.selectbox(col, [0, 1], index=0)
                else:
                    inputs[col] = st.number_input(col, value=float(default_val))

            submitted = st.form_submit_button("Predict")

        if submitted:
            model = models[selected_model_name]

            X_user = pd.DataFrame([inputs], columns=FEATURE_COLS)
            X_scaled = preprocess_X(X_user, imputer, scaler)
            pred = float(model.predict(X_scaled)[0])

            st.success(f"Predicted Birthweight ({selected_model_name}): **{pred:.4f}**")
            save_prediction(st.session_state.user["id"], selected_model_name, pred, inputs)

            st.info("Saved to History ✅")

    with right:
        st.subheader("Quick Guide (Presentation)")
        st.write(
            "- **Linear Regression**: baseline, fast.\n"
            "- **Decision Tree**: interpretable splits.\n"
            "- **Random Forest**: ensemble of trees, stable.\n"
            "- **XGBoost**: boosting, often strongest.\n"
            "- **Semantic KNN (Cosine)**: predicts using most similar records (similarity-based)."
        )
        st.caption("Use the History page to show saved predictions per user.")

    st.stop()

# -----------------------
# History page
# -----------------------
if st.session_state.page == "History":
    st.subheader("Prediction History")
    st.caption("Shows predictions made by the logged-in user.")

    limit = st.slider("How many recent predictions to show?", 10, 200, 50, 10)
    hist = load_history(st.session_state.user["id"], limit=limit)

    if hist.empty:
        st.info("No predictions yet. Go to Predict and make one.")
        st.stop()

    # Expand inputs_json into columns for viewing
    inputs_expanded = hist["inputs_json"].apply(lambda s: json.loads(s))
    inputs_df = pd.json_normalize(inputs_expanded)
    view_df = pd.concat([hist.drop(columns=["inputs_json"]), inputs_df], axis=1)

    st.dataframe(view_df, use_container_width=True, hide_index=True)

    # Download CSV
    csv_bytes = view_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download history as CSV", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv")

    st.stop()

# -----------------------
# Model Performance page (optional evaluation on dataset)
# -----------------------
if st.session_state.page == "Model Performance":
    st.subheader("Model Performance (Evaluation)")
    st.caption("This evaluates the saved models on a dataset split (no retraining). Target is Birthweight.")

    default_path = _find_default_dataset()
    use_upload = st.toggle("Upload dataset instead of auto-detect", value=(default_path is None))

    df = None
    try:
        if use_upload:
            uploaded = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
            if uploaded is not None:
                if uploaded.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded)
                elif uploaded.name.lower().endswith(".xlsx"):
                    df = pd.read_excel(uploaded, engine="openpyxl")
                else:
                    df = pd.read_excel(uploaded)
        else:
            if default_path is not None:
                df = load_data(default_path)
                st.caption(f"Loaded: `{default_path}`")
    except Exception as e:
        st.error("Failed to load dataset.")
        st.exception(e)

    if df is None:
        st.info("Upload a dataset (or place one in the folder) to see evaluation charts.")
        st.stop()

    try:
        results = evaluate_on_test(df)
    except Exception as e:
        st.error("Evaluation failed.")
        st.exception(e)
        st.stop()

    st.dataframe(results, use_container_width=True, hide_index=True)

    st.subheader("RMSE Comparison")
    fig, ax = plt.subplots()
    ax.bar(range(len(results)), results["RMSE"].values)
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_xlabel("Model")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results["Model"].tolist(), rotation=20, ha="right")
    st.pyplot(fig)

    best = results.iloc[0]
    st.success(f"Best on RMSE: **{best['Model']}** (RMSE={best['RMSE']:.4f})")
