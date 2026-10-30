"""
Machine learning prediction pipeline for NutriCheck.

This module is the single authoritative source for:
  * the feature schema expected by the saved malnutrition model,
  * input validation,
  * feature assembly and scaling,
  * model prediction and label decoding.

The same logic is used by the Flask backend (app.py) for every assessment
request. Frontend code must never perform its own ML classification.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILENAME = "malnutrition_model.joblib"
ENHANCED_MODEL_FILENAME = "enhanced_malnutrition_classifier.joblib"
SCALER_FILENAME = "feature_scaler.joblib"
LABEL_ENCODER_FILENAME = "label_encoder.joblib"

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
# The saved StandardScaler was fitted on exactly these five features, in this
# order, with these names. The RandomForest model was trained on the same five
# scaled features. Any deviation (renamed columns, different order, extra
# engineered columns, imputed or fabricated values) breaks the contract with
# the saved artifacts.
FEATURES = ["age_months", "weight_kg", "height_cm", "muac_cm", "bmi"]

# Plausible measurement ranges used for input validation. These are sanity
# checks only - they are not medical reference standards.
VALIDATION_RULES = {
    "age_years": {"min": 1, "max": 120},
    "weight_kg": {"min": 2, "max": 300},
    "height_cm": {"min": 30, "max": 250},
    "muac_cm": {"min": 8, "max": 60},
    "bmi": {"min": 8, "max": 80},
}

# Display label for each encoded class. The saved LabelEncoder stores classes
# in lowercase: ['moderate', 'normal', 'severe'].
CLASS_DISPLAY_LABELS = {
    "normal": "Normal",
    "moderate": "Moderate malnutrition",
    "severe": "Severe malnutrition",
}

# BMI categories (adult screening reference, shown separately from the ML
# prediction - they are never treated as malnutrition diagnoses).
BMI_CATEGORIES = [
    (18.5, "underweight"),
    (25.0, "normal"),
    (30.0, "overweight"),
    (float("inf"), "obese"),
]


@dataclass
class ValidationError:
    field: str
    message: str


@dataclass
class PredictionResult:
    nutritional_status: str          # e.g. "Moderate malnutrition"
    nutritional_status_key: str      # e.g. "moderate"
    bmi: float
    bmi_category: str                # e.g. "normal"
    model_type: str
    features_used: list[str]
    classes: list[str]


class MLPipeline:
    """Loads the saved artifacts and exposes one validated prediction path."""

    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.enhanced_classifier = None
        self.loaded = False
        self.load_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> bool:
        """Load model, scaler, label encoder and (optionally) enhanced wrapper.

        Returns True when the base prediction path is available. Missing
        enhanced wrapper is not fatal - the base model is authoritative.
        """
        try:
            self.model = joblib.load(os.path.join(BASE_DIR, MODEL_FILENAME))
            self.scaler = joblib.load(os.path.join(BASE_DIR, SCALER_FILENAME))
            self.label_encoder = joblib.load(os.path.join(BASE_DIR, LABEL_ENCODER_FILENAME))

            enhanced_path = os.path.join(BASE_DIR, ENHANCED_MODEL_FILENAME)
            if os.path.exists(enhanced_path):
                self.enhanced_classifier = joblib.load(enhanced_path)
                logger.info(
                    "Enhanced classifier loaded (base model remains authoritative)"
                )

            self._validate_artifacts()
            self.loaded = True
            logger.info("ML pipeline loaded successfully")
            return True
        except Exception as exc:  # pragma: no cover - startup safeguard
            self.loaded = False
            self.load_error = str(exc)
            logger.exception("Failed to load ML pipeline: %s", exc)
            return False

    def _validate_artifacts(self) -> None:
        if self.model is None or self.scaler is None or self.label_encoder is None:
            raise RuntimeError("Model, scaler or label encoder failed to load")

        scaler_features = list(getattr(self.scaler, "feature_names_in_", []))
        if scaler_features != FEATURES:
            raise RuntimeError(
                f"Scaler feature contract mismatch: expected {FEATURES}, "
                f"got {scaler_features}"
            )

        if getattr(self.model, "n_features_in_", None) != len(FEATURES):
            raise RuntimeError(
                f"Model expects {self.model.n_features_in_} features, "
                f"pipeline provides {len(FEATURES)}"
            )

        if len(getattr(self.label_encoder, "classes_", [])) == 0:
            raise RuntimeError("Label encoder has no classes")

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @classmethod
    def validate_assessment_input(cls, data: dict) -> list[ValidationError]:
        """Validate raw assessment input from the frontend.

        Never fabricates missing measurements. Returns a list of errors;
        an empty list means the input is valid.
        """
        errors: list[ValidationError] = []

        # ---- Required scalar fields ----------------------------------- #
        for field, rule in [
            ("age", VALIDATION_RULES["age_years"]),
            ("weight_kg", VALIDATION_RULES["weight_kg"]),
            ("height_cm", VALIDATION_RULES["height_cm"]),
            ("muac_cm", VALIDATION_RULES["muac_cm"]),
        ]:
            if field not in data or data[field] is None or data[field] == "":
                errors.append(ValidationError(field, f"{field} is required"))
                continue
            try:
                value = float(data[field])
            except (TypeError, ValueError):
                errors.append(ValidationError(field, f"{field} must be a number"))
                continue
            if not np.isfinite(value):
                errors.append(ValidationError(field, f"{field} must be a finite number"))
            elif value < rule["min"] or value > rule["max"]:
                errors.append(
                    ValidationError(
                        field,
                        f"{field} must be between {rule['min']} and {rule['max']}",
                    )
                )
            data[field] = value  # normalize to float

        # ---- Gender --------------------------------------------------- #
        gender = data.get("gender")
        if not gender or str(gender).strip().lower() not in {"male", "female", "other"}:
            errors.append(ValidationError("gender", "Select a valid gender"))
        else:
            data["gender"] = str(gender).strip().lower()

        # ---- Symptoms ------------------------------------------------- #
        symptoms = data.get("symptoms", [])
        if not isinstance(symptoms, list):
            errors.append(ValidationError("symptoms", "Symptoms must be a list"))
        else:
            clean = [str(s).strip().lower() for s in symptoms if str(s).strip()]
            data["symptoms"] = clean

        # ---- Optional text fields ------------------------------------- #
        for field in ("name", "activity", "diet", "life_stage"):
            value = data.get(field)
            data[field] = str(value).strip() if value is not None else ""

        return errors

    # ------------------------------------------------------------------ #
    # Feature assembly (the one reusable preprocessing path)
    # ------------------------------------------------------------------ #
    def build_feature_frame(self, data: dict) -> pd.DataFrame:
        """Build the single-row feature frame exactly matching the scaler.

        BMI is derived server-side from weight and height - it is never
        accepted as user input, which removes a fabrication vector.
        """
        age_years = float(data["age"])
        weight_kg = float(data["weight_kg"])
        height_cm = float(data["height_cm"])
        muac_cm = float(data["muac_cm"])

        age_months = age_years * 12
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m * height_m)

        bmi = round(float(bmi), 4)

        frame = pd.DataFrame(
            [
                {
                    "age_months": age_months,
                    "weight_kg": weight_kg,
                    "height_cm": height_cm,
                    "muac_cm": muac_cm,
                    "bmi": bmi,
                }
            ],
            columns=FEATURES,
        )
        return frame

    def predict(self, data: dict) -> PredictionResult:
        """Run the full prediction pipeline on validated input data."""
        if not self.loaded:
            raise RuntimeError(f"ML pipeline is not loaded: {self.load_error}")

        frame = self.build_feature_frame(data)
        scaled = self.scaler.transform(frame[FEATURES])

        # The base RandomForest is the authoritative estimator. The enhanced
        # wrapper wraps the same estimator and only adds a heuristic,
        # uncalibrated confidence score, which we deliberately do not expose
        # as a probability to users.
        encoded = int(self.model.predict(scaled)[0])
        classes = list(self.label_encoder.classes_)
        key = classes[encoded]
        display = CLASS_DISPLAY_LABELS.get(key, key.title())

        bmi = float(frame.iloc[0]["bmi"])
        bmi_category = next(
            label for threshold, label in BMI_CATEGORIES if bmi < threshold
        )

        return PredictionResult(
            nutritional_status=display,
            nutritional_status_key=key,
            bmi=round(bmi, 1),
            bmi_category=bmi_category,
            model_type=type(self.model).__name__,
            features_used=list(FEATURES),
            classes=classes,
        )

    def health_status(self) -> dict:
        """Lightweight status report for the /api/health endpoint."""
        return {
            "model_loaded": self.loaded,
            "model_type": type(self.model).__name__ if self.model else None,
            "features": list(FEATURES),
            "n_features": len(FEATURES),
            "classes": list(self.label_encoder.classes_) if self.label_encoder else None,
            "enhanced_classifier_available": self.enhanced_classifier is not None,
            "load_error": self.load_error,
        }


# Singleton used by the Flask application.
pipeline = MLPipeline()
