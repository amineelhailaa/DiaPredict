from pydantic import BaseModel, Field, field_validator


class PredictionCMD(BaseModel):
    Pregnancies: int = Field(
        ...,
        ge=0,
        le=20,
        description="Number of pregnancies. Must be an integer between 0 and 20.",
    )

    Glucose: float | None = Field(
        ...,
        gt=0,
        le=600,
        description="Plasma glucose concentration in mg/dL. Must be between 1 and 600, or null for imputation.",
    )

    BloodPressure: float | None = Field(
        ...,
        gt=0,
        le=200,
        description="Diastolic blood pressure in mmHg. Must be between 1 and 200, or null for imputation.",
    )

    SkinThickness: float | None = Field(
        ...,
        gt=0,
        le=100,
        description="Triceps skin fold thickness in mm. Must be between 1 and 100, or null for imputation.",
    )

    Insulin: float | None = Field(
        ...,
        gt=0,
        le=1000,
        description="2-hour serum insulin in microU/mL. Must be between 1 and 1000, or null for imputation.",
    )

    BMI: float | None = Field(
        ...,
        ge=10,
        le=80,
        description="Body mass index in kg/m². Must be between 10 and 80, or null for imputation.",
    )

    DiabetesPedigreeFunction: float = Field(
        ...,
        ge=0,
        le=3,
        description="Diabetes pedigree function. Must be between 0 and 3.",
    )

    Age: int = Field(
        ...,
        ge=21,
        le=100,
        description="Age in years. Must be an integer between 21 and 100.",
    )

    @field_validator(
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    )
    @classmethod
    def reject_non_finite_values(cls, value):
        if value is None:
            return value

        if isinstance(value, float) and value != value:
            raise ValueError("Value must not be NaN.")

        if value in (float("inf"), float("-inf")):
            raise ValueError("Value must be finite.")

        return value