import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

# Columns that are extra services (used to count add-ons)
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]

# Columns the model will treat as numbers
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges",
           "num_addons", "is_autopay", "avg_charge_per_month"]

# Columns the model will treat as categories
CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
               "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
               "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
               "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group"]


def clean_data(df):
    """Fix the data problems found in the EDA."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def add_features(df):
    """Create new columns. This step is inside the pipeline, so the app uses it too."""
    df = df.copy()
    # How many extra services the customer has (0 to 6)
    df["num_addons"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    # 1 if payment is automatic (bank transfer or credit card), else 0
    df["is_autopay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    # Average monthly spending; clip(lower=1) avoids dividing by zero when tenure = 0
    df["avg_charge_per_month"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    # Loyalty stage of the customer
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                                labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    return df


def build_pipeline(model):
    """Feature engineering -> scaling and encoding -> model, in one object."""
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", preprocess),
        ("model", model),
    ])