"""Simple checks for individual DiaPredict values.

Each function returns True for a valid value and False for an invalid value.
Missing values need preprocessing; an unusual value is not a diagnosis.
Zero-as-missing applies to this dataset, not every clinical laboratory.

Feature meanings assume the Pima schema; confirm the local data's provenance:
https://www.openml.org/search?type=data&id=37
Diabetes thresholds are not data-validity limits:
https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis
"""

from math import isfinite
from numbers import Real

import pandas as pd


def validate_number(value):
    """Common check: require a present, finite number; reject strings and booleans."""
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return False
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    return isfinite(value)


def validate_nonnegative(value):
    """Counts, concentrations, and these physical measurements cannot be negative."""
    return validate_number(value) and value >= 0


def validate_zero_as_missing(value):
    """Shared rule for measurements where zero represents missing data here."""
    return validate_nonnegative(value) and value != 0


def validate_pregnancies(value):
    """Pregnancy count must be a nonnegative integer; zero pregnancies is valid."""
    return validate_nonnegative(value) and value % 1 == 0


def validate_glucose(value):
    """2-hour glucose (mg/dL): zero is missing; high glucose may be real data."""
    return validate_zero_as_missing(value)


def validate_blood_pressure(value):
    """Diastolic pressure (mmHg): zero is missing; do not apply systolic limits."""
    return validate_zero_as_missing(value)


def validate_skin_thickness(value):
    """Triceps skinfold (mm): negative thickness is invalid; zero is missing."""
    return validate_zero_as_missing(value)


def validate_insulin(value):
    """2-hour insulin (microU/mL): zero is missing here; normal ranges vary by assay."""
    return validate_zero_as_missing(value)


def validate_bmi(value):
    """BMI (kg/m²) must be positive; obesity is not a data-validation error."""
    return validate_zero_as_missing(value)


def validate_diabetes_pedigree_function(value):
    """Nonnegative family-history score, not a probability; values above 1 are valid."""
    return validate_nonnegative(value)


def validate_age(value):
    """Age must be a positive whole number of completed years."""
    return validate_nonnegative(value) and value != 0 and value % 1 == 0
