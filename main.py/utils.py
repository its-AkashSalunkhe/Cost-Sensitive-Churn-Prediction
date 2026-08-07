"""
utils.py
--------
All data-loading, preprocessing, modeling and business-cost logic for the
Customer Churn Prediction dashboard. This mirrors the exact pipeline built
in main.ipynb so that the app's numbers match the notebook 1:1.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco_customer_churn.xlsx")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

BINARY_COLS = [
    'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
    'Multiple Lines', 'Online Security', 'Online Backup', 'Device Protection',
    'Tech Support', 'Streaming TV', 'Streaming Movies', 'Paperless Billing',
    'Churn Label'
]

COLS_REPLACE_NO = [
    'Multiple Lines', 'Online Security', 'Online Backup', 'Device Protection',
    'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract'
]

DUMMY_COLS = ['Internet Service', 'Contract', 'Payment Method']

RAW_INPUT_COLUMNS = [
    'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Tenure Months',
    'Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security',
    'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
    'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method',
    'Monthly Charges', 'CLTV'
]


# --------------------------------------------------------------------------
# Artifact loading (the objects saved at the end of main.ipynb)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(os.path.join(ARTIFACT_DIR, "churn_model.pkl"))
    model_columns = joblib.load(os.path.join(ARTIFACT_DIR, "model_columns.pkl"))
    best_threshold = joblib.load(os.path.join(ARTIFACT_DIR, "best_threshold.pkl"))
    label_encoders = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoders.pkl"))
    return model, model_columns, best_threshold, label_encoders


# --------------------------------------------------------------------------
# Raw data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data():
    data = pd.read_excel(DATA_PATH)
    return data


# --------------------------------------------------------------------------
# Full cleaning + encoding pipeline — mirrors the notebook exactly.
# Uses the SAVED label encoders (fit at training time) so encoding is
# guaranteed to be identical to what the deployed model expects.
# --------------------------------------------------------------------------
def clean_raw(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors='coerce').fillna(0)
    if 'CustomerID' in data.columns:
        data = data.drop('CustomerID', axis=1)

    for col in COLS_REPLACE_NO:
        data[col] = data[col].replace({'No internet service': 'No', 'No phone service': 'No'})

    drop_cols = ['Country', 'State', 'City', 'Lat Long', 'Churn Reason',
                 'Count', 'Zip Code', 'Latitude', 'Longitude', 'Churn Value',
                 'Churn Score', 'Total Charges']
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])
    return data


@st.cache_data(show_spinner=False)
def preprocess_full_dataset(_label_encoders, model_columns):
    """Clean + encode the full raw dataset (for EDA and for rebuilding the
    exact train/test split used when the artifacts were created)."""
    raw = load_raw_data()
    data = clean_raw(raw)

    for col in BINARY_COLS:
        data[col] = _label_encoders[col].transform(data[col])

    data = pd.get_dummies(data, columns=DUMMY_COLS, drop_first=True)
    bool_cols = data.select_dtypes(include='bool').columns
    data[bool_cols] = data[bool_cols].astype(int)

    y = data['Churn Label']
    x = data.drop('Churn Label', axis=1)

    # align columns exactly to what the model expects (add any missing
    # dummy columns as 0, drop extras, and fix order)
    for col in model_columns:
        if col not in x.columns:
            x[col] = 0
    x = x[model_columns]

    return raw, data, x, y


@st.cache_data(show_spinner=False)
def get_train_test_split(x: pd.DataFrame, y: pd.Series):
    """Reproduces the EXACT split used in main.ipynb (random_state=42,
    test_size=0.2, stratify=y) so test-set metrics match the notebook."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    return x_train, x_test, y_train, y_test


# --------------------------------------------------------------------------
# Business cost functions — identical to calculate_business_cost_v2 in the
# notebook: FN = lose full CLTV, FP = wasted retention offer sized at
# fp_pct of that customer's CLTV.
# --------------------------------------------------------------------------
def calculate_business_cost(y_true, y_pred, cltv_values, fp_pct=0.10):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cltv_values = np.array(cltv_values)

    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_cost = cltv_values[fn_mask].sum()

    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_cost_total = (cltv_values[fp_mask] * fp_pct).sum()

    total_cost = fn_cost + fp_cost_total
    return total_cost, fn_cost, fp_cost_total


def threshold_sweep(model, x_test, y_test, cltv_values, thresholds, fp_pct=0.10):
    y_proba = model.predict_proba(x_test)[:, 1]
    rows = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        cost, fn_c, fp_c = calculate_business_cost(y_test, y_pred_t, cltv_values, fp_pct)
        pct_flagged = y_pred_t.mean() * 100
        rows.append({
            'threshold': t, 'total_cost': cost, 'fn_cost': fn_c,
            'fp_cost': fp_c, 'pct_flagged': pct_flagged
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Train all 4 models (for the Model Comparison tab). Cached so this only
# runs once per session. Hyperparameters match the notebook exactly.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_all_models(x_train, y_train):
    models = {}

    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr.fit(x_train, y_train)
    models['Logistic Regression'] = lr

    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf.fit(x_train, y_train)
    models['Random Forest'] = rf

    if XGB_AVAILABLE:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42,
                             eval_metric='logloss')
        xgb.fit(x_train, y_train)
        models['XGBoost'] = xgb

    if LGBM_AVAILABLE:
        lgbm = LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)
        lgbm.fit(x_train, y_train)
        models['LightGBM'] = lgbm

    return models


def evaluate_model_at_threshold(model, x_test, y_test, cltv_test, threshold):
    y_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    total_cost, fn_cost, fp_cost = calculate_business_cost(y_test, y_pred, cltv_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'total_cost': total_cost,
        'fn_cost': fn_cost,
        'fp_cost': fp_cost,
        'pct_flagged': y_pred.mean() * 100,
    }
    return y_pred, y_proba, cm, metrics


def find_optimal_threshold(model, x_test, y_test, cltv_test,
                            thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)
    df = threshold_sweep(model, x_test, y_test, cltv_test, thresholds)
    best_row = df.loc[df['total_cost'].idxmin()]
    return best_row['threshold'], df


# --------------------------------------------------------------------------
# Single-customer prediction — takes raw-form-style input dict and runs it
# through the same encoding pipeline as training, using the SAVED encoders.
# --------------------------------------------------------------------------
def preprocess_single_customer(input_dict: dict, label_encoders: dict, model_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([input_dict])

    # apply the "No internet/phone service" -> "No" collapse, same as training
    for col in COLS_REPLACE_NO:
        if col in df.columns:
            df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

    for col in BINARY_COLS:
        if col == 'Churn Label':
            continue
        if col in df.columns:
            le = label_encoders[col]
            df[col] = le.transform(df[col])

    df = pd.get_dummies(df, columns=[c for c in DUMMY_COLS if c in df.columns], drop_first=True)

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]
    return df


def risk_segment(probability: float) -> str:
    if probability >= 0.60:
        return "🔴 Critical Risk"
    elif probability >= 0.30:
        return "🟠 High Risk"
    elif probability >= 0.15:
        return "🟡 Moderate Risk"
    else:
        return "🟢 Low Risk"
