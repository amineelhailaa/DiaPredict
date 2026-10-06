

from math import isfinite
from numbers import Real

import pandas as pd


def validate_number(value):
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return False
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    return isfinite(value)


def validate_nonnegative(value):
    return validate_number(value) and value >= 0


def validate_zero_as_missing(value):
    return validate_nonnegative(value) and value != 0


def validate_pregnancies(value):
    return validate_nonnegative(value) and value % 1 == 0


def validate_glucose(value):
    return validate_zero_as_missing(value)


def validate_blood_pressure(value):
    return validate_zero_as_missing(value)


def validate_skin_thickness(value):
    return validate_zero_as_missing(value)


def validate_insulin(value):
    return validate_zero_as_missing(value)


def validate_bmi(value):
    return validate_zero_as_missing(value)


def validate_diabetes_pedigree_function(value):
    return validate_nonnegative(value)


def validate_age(value):
    return validate_nonnegative(value) and value != 0 and value % 1 == 0
