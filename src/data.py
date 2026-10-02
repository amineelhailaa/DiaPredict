"""Data loading and basic cleaning helpers."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection.tests.test_split import test_train_test_split

from src.validation import (
    validate_age,
    validate_blood_pressure,
    validate_bmi,
    validate_diabetes_pedigree_function,
    validate_glucose,
    validate_insulin,
    validate_pregnancies,
    validate_skin_thickness,
)

VALIDATORS = {
    "Pregnancies": validate_pregnancies,
    "Glucose": validate_glucose,
    "BloodPressure": validate_blood_pressure,
    "SkinThickness": validate_skin_thickness,
    "Insulin": validate_insulin,
    "BMI": validate_bmi,
    "DiabetesPedigreeFunction": validate_diabetes_pedigree_function,
    "Age": validate_age,
}


def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def invalid_values_to_na(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with values rejected by their validators replaced by NA."""
    cleaned = data.copy()
    for column, validator in VALIDATORS.items():
        if column not in cleaned:
            continue
        valid = cleaned[column].map(validator)
        cleaned.loc[~valid, column] = pd.NA
    return cleaned


def outliering(series: pd.Series) -> pd.Series:
    """Return a boolean mask identifying outliers with the 1.5 IQR rule."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (series < lower_bound) | (series > upper_bound)


def outliers_for_all_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return an outlier mask for every numeric column."""
    numeric_data = data.select_dtypes(include="number")
    return numeric_data.apply(outliering)


def stringify_column(series: pd.Series) -> pd.Series:
    """Convert a column to clean strings while preserving missing values."""
    missing_values = ["NaN", "nan", "None", "null", ""]
    return series.astype("string").str.strip().replace(missing_values, pd.NA)


def numeric_column(series: pd.Series) -> pd.Series:
    """Convert a column to numbers, replacing conversion errors with NA."""
    return pd.to_numeric(series, errors="coerce")


def clean_data(df):
    df = df.drop(columns=["Unnamed: 0"])
    return invalid_values_to_na(df)

def splitData(df,size, random_state):
    return (
        train_test_split(
    df,
    test_size=size,
    random_state=random_state,
    )
    )

