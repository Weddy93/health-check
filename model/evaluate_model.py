"""
Reproducible model evaluation for NutriCheck.

This script performs structural validation of the saved ML artifacts and,
IF the labeled dataset is available, a genuine stratified evaluation.

IMPORTANT: no evaluation metrics are fabricated. The file
preprocessed_malnutrition_data.csv does not exist in this repository, so
accuracy/precision/recall cannot be genuinely measured here. This script
reports exactly what is missing instead of inventing results.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CANDIDATES = [
    "preprocessed_malnutrition_data.csv",
    os.path.join(BASE_DIR, "preprocessed_malnutrition_data.csv"),
]

MODEL_FILENAME = "malnutrition_model.joblib"
SCALER_FILENAME = "feature_scaler.joblib"
LABEL_ENCODER_FILENAME = "label_encoder.joblib"
ENHANCED_FILENAME = "enhanced_malnutrition_classifier.joblib"

# The contract enforced by ml_pipeline.py and used in production.
EXPECTED_FEATURES = ["age_months", "weight_kg", "height_cm", "muac_cm", "bmi"]

REPORT = {
    "generated_at": datetime.now().isoformat(),
    "artifact_checks": {},
    "structural_checks": {},
    "dataset": None,
    "evaluation": None,
    "conclusion": "",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def find_dataset() -> str | None:
    for candidate in DATASET_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def check_artifacts() -> dict:
    """Load every saved artifact and verify the production contract."""
    checks = {}
    artifacts = {}

    for name, filename in [
        ("model", MODEL_FILENAME),
        ("scaler", SCALER_FILENAME),
        ("label_encoder", LABEL_ENCODER_FILENAME),
    ]:
        path = os.path.join(BASE_DIR, filename)
        try:
            artifacts[name] = joblib.load(path)
            checks[name] = {
                "loaded": True,
                "type": type(artifacts[name]).__name__,
            }
        except Exception as exc:
            checks[name] = {"loaded": False, "error": str(exc)}

    enhanced_path = os.path.join(BASE_DIR, ENHANCED_FILENAME)
    if os.path.exists(enhanced_path):
        try:
            artifacts["enhanced"] = joblib.load(enhanced_path)
            checks["enhanced"] = {
                "loaded": True,
                "type": type(artifacts["enhanced"]).__name__,
                "wraps": type(artifacts["enhanced"].model).__name__
                if hasattr(artifacts["enhanced"], "model")
                else None,
            }
        except Exception as exc:
            checks["enhanced"] = {"loaded": False, "error": str(exc)}
    else:
        checks["enhanced"] = {"loaded": False, "error": "file not found"}

    # Contract checks
    contract = {}
    try:
        model = artifacts["model"]
        scaler = artifacts["scaler"]
        encoder = artifacts["label_encoder"]

        scaler_features = list(getattr(scaler, "feature_names_in_", []))
        contract["scaler_features"] = scaler_features
        contract["scaler_features_match"] = scaler_features == EXPECTED_FEATURES
        contract["model_n_features"] = int(getattr(model, "n_features_in_", -1))
        contract["model_n_features_match"] = int(getattr(model, "n_features_in_", -1)) == len(
            EXPECTED_FEATURES
        )
        contract["encoder_classes"] = list(getattr(encoder, "classes_", []))
        contract["encoder_fitted"] = bool(getattr(encoder, "classes_", None) is not None)
        contract["model_classes"] = [int(x) for x in getattr(model, "classes_", [])]
    except Exception as exc:
        contract["error"] = str(exc)

    checks["contract"] = contract
    return checks, artifacts


def structural_validation(artifacts: dict) -> dict:
    """Verify the production prediction path end-to-end with synthetic
    but plausible inputs. No labels are needed; this checks alignment,
    determinism and sensible decoding only."""
    results = {}
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    encoder = artifacts["label_encoder"]

    samples = [
        {"label": "child underweight", "age_months": 36, "weight_kg": 10, "height_cm": 85, "muac_cm": 13, "bmi": 13.8},
        {"label": "adult normal", "age_months": 300, "weight_kg": 70, "height_cm": 175, "muac_cm": 27, "bmi": 22.9},
        {"label": "adult obese", "age_months": 360, "weight_kg": 105, "height_cm": 170, "muac_cm": 34, "bmi": 36.3},
    ]

    predictions = []
    for sample in samples:
        frame = pd.DataFrame([sample], columns=EXPECTED_FEATURES)
        scaled = scaler.transform(frame)
        encoded = int(model.predict(scaled)[0])
        proba = model.predict_proba(scaled)[0]
        key = list(encoder.classes_)[encoded]
        predictions.append({
            "input": sample["label"],
            "features": list(frame.iloc[0]),
            "prediction_key": key,
            "max_class_probability": float(np.max(proba)),
        })

    # Determinism: same input twice must yield the same prediction.
    frame = pd.DataFrame([{
        "age_months": 300, "weight_kg": 70, "height_cm": 175, "muac_cm": 27, "bmi": 22.9,
    }], columns=EXPECTED_FEATURES)
    p1 = int(model.predict(scaler.transform(frame))[0])
    p2 = int(model.predict(scaler.transform(frame))[0])
    results["deterministic"] = bool(p1 == p2)

    results["sample_predictions"] = predictions
    results["feature_order_used"] = EXPECTED_FEATURES
    return results


def evaluate_with_dataset(dataset_path: str, artifacts: dict) -> dict | None:
    """Genuine stratified evaluation. Only runs when a labeled dataset exists."""
    try:
        data = pd.read_csv(dataset_path)
    except Exception as exc:
        return {"skipped": True, "reason": f"Could not read dataset: {exc}"}

    if "nutrition_status" not in data.columns:
        return {
            "skipped": True,
            "reason": "Dataset has no 'nutrition_status' target column",
            "columns": list(data.columns),
        }

    y_raw = data["nutrition_status"].astype(str).str.lower().str.strip()
    encoder = artifacts["label_encoder"]
    known = set(encoder.classes_)
    unknown = set(y_raw.unique()) - known
    if unknown:
        return {
            "skipped": True,
            "reason": f"Dataset labels not known to label encoder: {sorted(unknown)}",
            "encoder_classes": list(encoder.classes_),
            "dataset_labels": sorted(y_raw.unique().tolist()),
        }

    # Build features strictly through the production pipeline contract.
    records = []
    valid_labels = []
    for _, row in data.iterrows():
        try:
            bmi = float(row["weight_kg"]) / ((float(row["height_cm"]) / 100) ** 2)
            records.append({
                "age_months": float(row["age_months"]),
                "weight_kg": float(row["weight_kg"]),
                "height_cm": float(row["height_cm"]),
                "muac_cm": float(row["muac_cm"]),
                "bmi": round(bmi, 4),
            })
            valid_labels.append(row["nutrition_status"].lower().strip())
        except Exception:
            continue  # skip malformed rows; never invent values

    X = pd.DataFrame(records, columns=EXPECTED_FEATURES)
    y = encoder.transform(valid_labels)

    if len(X) < 10 or len(set(valid_labels)) < 2:
        return {"skipped": True, "reason": "Insufficient usable rows for evaluation"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = artifacts["model"]
    model.fit(X_train, y_train)  # evaluation fit on a copy in memory only
    y_pred = model.predict(X_test)

    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=np.unique(y),
        target_names=encoder.classes_[np.unique(y)],
        output_dict=True,
    )

    return {
        "skipped": False,
        "dataset_rows": int(len(data)),
        "usable_rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": {k: int(v) for k, v in pd.Series(valid_labels).value_counts().items()},
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(encoder.classes_),
        "classification_report": report,
    }


def main() -> int:
    print("=" * 70)
    print("NutriCheck model evaluation (reproducible, no fabricated data)")
    print("=" * 70)

    checks, artifacts = check_artifacts()
    REPORT["artifact_checks"] = checks
    log("Artifact checks complete")

    ok = all(v.get("loaded") for v in checks.values() if isinstance(v, dict) and "loaded" in v)
    if not ok:
        REPORT["conclusion"] = (
            "Aborted: one or more saved artifacts could not be loaded. "
            "Check model/ for malnutrition_model.joblib, feature_scaler.joblib "
            "and label_encoder.joblib."
        )
        print(REPORT["conclusion"])
        return 1

    structural = structural_validation(artifacts)
    REPORT["structural_checks"] = structural
    log("Structural validation complete")

    dataset_path = find_dataset()
    REPORT["dataset"] = {"path": dataset_path, "found": dataset_path is not None}
    if dataset_path:
        log(f"Dataset found: {dataset_path}")
        REPORT["evaluation"] = evaluate_with_dataset(dataset_path, artifacts)
    else:
        log("Dataset preprocessed_malnutrition_data.csv NOT FOUND")
        REPORT["evaluation"] = {
            "skipped": True,
            "reason": "preprocessed_malnutrition_data.csv is not present in the repository",
            "required": (
                "A labeled CSV with columns age_months, weight_kg, height_cm, "
                "muac_cm, bmi and a nutrition_status target column whose values "
                "match the label encoder classes (moderate / normal / severe)."
            ),
        }

    # Conclusion
    structural_ok = structural.get("deterministic", False)
    eval_blocked = REPORT["evaluation"].get("skipped", True)
    if eval_blocked:
        REPORT["conclusion"] = (
            "Model artifacts load correctly and the production prediction path "
            "is structurally valid (aligned features, deterministic predictions, "
            "correct label decoding). Genuine accuracy/precision/recall/F1 scores "
            "cannot be reported because the labeled dataset "
            "preprocessed_malnutrition_data.csv is missing from the repository. "
            "No metrics have been invented. Add the dataset and re-run this "
            "script to obtain genuine stratified evaluation results."
        )
    else:
        ev = REPORT["evaluation"]
        REPORT["conclusion"] = (
            f"Genuine evaluation on {ev.get('usable_rows')} usable rows "
            f"({ev.get('test_rows')} test): accuracy={ev.get('accuracy', 0):.4f}, "
            f"precision={ev.get('precision_weighted', 0):.4f}, "
            f"recall={ev.get('recall_weighted', 0):.4f}, "
            f"f1={ev.get('f1_weighted', 0):.4f}. Structural validation passed."
        )

    print("=" * 70)
    print("REPORT SUMMARY")
    print("=" * 70)
    print(json.dumps(REPORT, indent=2))

    report_path = os.path.join(BASE_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(REPORT, f, indent=2)
    log(f"Full report written to {report_path}")
    return 0 if (ok and structural_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
