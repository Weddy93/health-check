"""
NutriCheck backend (Flask).

Single authoritative assessment workflow:
  1. React validates the form
  2. React POSTs assessment data to /api/assess-health
  3. Flask validates the incoming data (ml_pipeline.validate_assessment_input)
  4. Flask derives features consistently (ml_pipeline.build_feature_frame)
  5. Flask loads the saved ML model + preprocessing objects (ml_pipeline)
  6. Flask generates the prediction and returns structured JSON
  7. React displays the returned result

Frontend code never performs its own ML classification.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from ml_pipeline import pipeline, FEATURES, CLASS_DISPLAY_LABELS

# --------------------------------------------------------------------------- #
# Application setup
# --------------------------------------------------------------------------- #
app = Flask(__name__)

# Vite dev server runs on 5173; production is served by Flask itself.
CORS(
    app,
    origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:5000",
    ],
    supports_credentials=False,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Load ML artifacts on startup. Failure is degraded gracefully: /api/health
# reports model_loaded: false and /api/assess-health returns a 503.
MODELS_READY = pipeline.load()
if MODELS_READY:
    logger.info("ML models ready")
else:
    logger.error("ML models failed to load: %s", pipeline.load_error)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def error_response(error_code: str, message: str, status: int, details=None):
    body = {"success": False, "error": error_code, "message": message}
    if details is not None:
        body["details"] = details
    return jsonify(body), status


def symptom_nutrient_insights(symptoms: list[str]) -> list[dict]:
    """Map reported symptoms to general nutritional considerations.

    These are educational pointers only - never diagnoses of medical
    conditions derived from symptoms alone.
    """
    mapping = {
        "fatigue": {"nutrient": "Iron", "foods": ["Spinach", "Lentils", "Red meat", "Fortified cereals"]},
        "weakness": {"nutrient": "Protein", "foods": ["Eggs", "Chicken", "Fish", "Beans", "Greek yogurt"]},
        "hairloss": {"nutrient": "Iron / Zinc", "foods": ["Beef", "Pumpkin seeds", "Chickpeas", "Spinach"]},
        "skin": {"nutrient": "Vitamin A / Zinc", "foods": ["Carrots", "Sweet potatoes", "Kale", "Eggs"]},
        "weight-loss": {"nutrient": "Protein / Calories", "foods": ["Nuts", "Avocado", "Whole grains", "Dried fruits"]},
        "weight-gain": {"nutrient": "Balanced energy", "foods": ["Vegetables", "Fruits", "Lean protein", "Whole grains"]},
        "bones": {"nutrient": "Calcium / Vitamin D", "foods": ["Dairy products", "Fortified plant milk", "Leafy greens"]},
        "digestive": {"nutrient": "Fiber", "foods": ["Whole grains", "Vegetables", "Fruits", "Legumes"]},
    }
    insights = []
    seen = set()
    for symptom in symptoms:
        if symptom in mapping and symptom not in seen:
            seen.add(symptom)
            insights.append(mapping[symptom])
    return insights


def bmi_category_label(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"


# --------------------------------------------------------------------------- #
# API: health
# --------------------------------------------------------------------------- #
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        **pipeline.health_status(),
    })

# --------------------------------------------------------------------------- #
# API: assess-health  (single authoritative assessment endpoint)
# --------------------------------------------------------------------------- #
@app.route("/api/assess-health", methods=["POST"])
def assess_health():
    """Assess health using the saved ML model.

    Request (JSON):
        name, age (years), gender, weight (kg), height (cm), muac (cm),
        activity, diet, life_stage, symptoms (list of strings)

    Response (JSON, 200):
        { success, prediction, nutritional_status, bmi, bmi_category,
          symptoms, symptoms_reported, no_symptoms_reported,
          recommendations, model_information, profile_summary, timestamp }

    Error responses use { success: false, error, message, details? } with
    appropriate HTTP status codes (400 / 503 / 500).
    """
    if not MODELS_READY:
        return error_response(
            "model_unavailable",
            "The assessment model is not available on the server. Please try again later.",
            503,
        )

    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return error_response(
                "invalid_input",
                "No assessment data provided. Send a JSON object with the assessment fields.",
                400,
            )

        # Normalize frontend field names to the pipeline contract.
        normalized = {
            "age": data.get("age"),
            "weight_kg": data.get("weight"),
            "height_cm": data.get("height"),
            "muac_cm": data.get("muac"),
            "gender": data.get("gender"),
            "symptoms": data.get("symptoms", []),
            "name": data.get("name", ""),
            "activity": data.get("activity", ""),
            "diet": data.get("diet", ""),
            "life_stage": data.get("life_stage", ""),
        }

        validation_errors = pipeline.validate_assessment_input(normalized)
        if validation_errors:
            return error_response(
                "validation_error",
                "Assessment data failed validation. Please correct the fields below.",
                400,
                details=[{"field": e.field, "message": e.message} for e in validation_errors],
            )

        # Single authoritative prediction path.
        result = pipeline.predict(normalized)

        symptoms = normalized["symptoms"]
        no_symptoms = any(s in {"no-symptoms", "none", "no symptoms"} for s in symptoms)
        actual_symptoms = [s for s in symptoms if s not in {"no-symptoms", "none", "no symptoms"}]

        recommendations = []
        if result.nutritional_status_key in {"moderate", "severe"}:
            recommendations.append(
                "Share this result with a healthcare professional for further assessment."
            )
        if actual_symptoms:
            recommendations.append(
                "Discuss your reported symptoms with a qualified clinician."
            )
        if result.bmi_category in {"underweight", "overweight", "obese"}:
            recommendations.append(
                f"Your BMI category ({result.bmi_category}) warrants a conversation with a nutritionist."
            )
        if not recommendations:
            recommendations.append(
                "Continue with a balanced diet and regular health check-ups."
            )

        response = {
            "success": True,
            "prediction": {
                "nutritional_status": result.nutritional_status,
                "nutritional_status_key": result.nutritional_status_key,
                "model_type": result.model_type,
                "features_used": result.features_used,
                "classes": result.classes,
            },
            "nutritional_status": result.nutritional_status,
            "bmi": result.bmi,
            "bmi_category": result.bmi_category,
            "symptoms": actual_symptoms,
            "symptoms_reported": bool(actual_symptoms),
            "no_symptoms_reported": bool(no_symptoms),
            "recommendations": recommendations,
            "symptom_insights": symptom_nutrient_insights(actual_symptoms),
            "model_information": {
                "model_type": result.model_type,
                "features": result.features_used,
                "classes": result.classes,
                "note": "Preliminary ML-based nutritional screening - not a medical diagnosis.",
            },
            "profile_summary": {
                "name": normalized["name"] or "Not provided",
                "age": normalized["age"],
                "gender": normalized["gender"],
                "activity_level": normalized["activity"],
                "diet_type": normalized["diet"],
                "life_stage": normalized["life_stage"],
            },
            "timestamp": datetime.now().isoformat(),
        }
        logger.info(
            "Assessment completed for %s: %s (BMI %s)", normalized["name"], result.nutritional_status, result.bmi
        )
        return jsonify(response), 200

    except error_response as err:
        return err
    except Exception as exc:
        logger.exception("Assessment error: %s", exc)
        return error_response(
            "prediction_error",
            "The assessment could not be completed. Please try again or contact support.",
            500,
        )

# --------------------------------------------------------------------------- #
# API: generate-meal-plan
# --------------------------------------------------------------------------- #
MEAL_TEMPLATES = {
    "balanced": {
        "breakfast": ["Whole grain cereal with milk", "Fresh fruit", "Boiled egg"],
        "lunch": ["Vegetable soup", "Sandwich with lean protein", "Side salad"],
        "dinner": ["Baked fish", "Steamed vegetables", "Brown rice"],
        "snacks": ["Yogurt", "Fresh fruit"],
    },
    "high_protein": {
        "breakfast": ["Scrambled eggs with spinach", "Whole grain toast", "Greek yogurt"],
        "lunch": ["Grilled chicken salad", "Quinoa", "Mixed vegetables"],
        "dinner": ["Baked salmon", "Steamed broccoli", "Sweet potato"],
        "snacks": ["Handful of almonds", "Protein shake"],
    },
    "iron_rich": {
        "breakfast": ["Fortified cereal with berries", "Orange juice", "Hard-boiled eggs"],
        "lunch": ["Spinach and lentil soup", "Whole grain bread", "Lean red meat"],
        "dinner": ["Beef stir-fry with bell peppers", "Brown rice", "Steamed kale"],
        "snacks": ["Dried apricots", "Pumpkin seeds"],
    },
    "energy_boost": {
        "breakfast": ["Oatmeal with bananas", "Walnuts", "Milk"],
        "lunch": ["Chicken and avocado wrap", "Fruit salad", "Yogurt"],
        "dinner": ["Turkey meatballs", "Whole wheat pasta", "Green beans"],
        "snacks": ["Apple with peanut butter", "Trail mix"],
    },
}

NUTRIENT_FOODS = {
    "Protein": ["Eggs", "Chicken", "Fish", "Beans", "Nuts"],
    "Iron": ["Spinach", "Red meat", "Lentils", "Fortified cereals"],
    "Calcium": ["Dairy", "Leafy greens", "Fortified plant milks"],
    "Vitamin A": ["Carrots", "Sweet potatoes", "Leafy greens"],
    "Vitamin D": ["Fatty fish", "Fortified foods"],
    "Zinc": ["Meat", "Shellfish", "Legumes", "Nuts"],
    "Folic Acid": ["Leafy greens", "Citrus fruits", "Beans"],
    "Omega-3": ["Fatty fish", "Flaxseeds", "Walnuts"],
    "Vitamin B12": ["Meat", "Dairy", "Eggs", "Fortified cereals"],
    "Fiber": ["Whole grains", "Vegetables", "Fruits", "Legumes"],
}


def _meal_plan_for(status_key: str) -> str:
    if status_key == "severe":
        return "high_protein"
    if status_key == "moderate":
        return "iron_rich"
    if status_key == "normal":
        return "balanced"
    return "energy_boost"


@app.route("/api/generate-meal-plan", methods=["POST"])
def generate_meal_plan():
    """Generate meal suggestions aligned with the assessment outcome.

    Educational guidance only - not a substitute for a dietitian's plan.
    """
    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return error_response(
                "invalid_input",
                "No assessment data provided for meal planning.",
                400,
            )

        status_key = str(data.get("nutritional_status_key", "normal")).lower()
        bmi = float(data.get("bmi", 22) or 22)
        symptoms = data.get("symptoms", []) or []
        diet = str(data.get("diet_type", "")).lower()

        template_key = _meal_plan_for(status_key)
        plan = MEAL_TEMPLATES[template_key]

        considerations = [
            "Eat regular, balanced meals throughout the day.",
            "Include a source of protein at each main meal.",
            "Choose whole grains, vegetables and fruit daily.",
            "Drink plenty of water - aim for about 6-8 glasses per day unless advised otherwise by a clinician.",
        ]
        if status_key in {"moderate", "severe"}:
            considerations.insert(
                0,
                "You have been flagged for possible nutritional risk - consult a healthcare professional before making major dietary changes.",
            )

        nutrient_recs = []
        seen_nutrients = set()
        for symptom in symptoms:
            for nutrient, foods in NUTRIENT_FOODS.items():
                if nutrient.lower() in symptom or symptom in nutrient.lower():
                    if nutrient not in seen_nutrients:
                        seen_nutrients.add(nutrient)
                        nutrient_recs.append({"nutrient": nutrient, "foods": foods})

        return jsonify({
            "success": True,
            "template": template_key,
            "based_on": {
                "nutritional_status": status_key,
                "bmi": bmi,
                "diet_type": diet,
            },
            "breakfast": plan["breakfast"],
            "lunch": plan["lunch"],
            "dinner": plan["dinner"],
            "snacks": plan["snacks"],
            "considerations": considerations,
            "nutrient_suggestions": nutrient_recs,
            "disclaimer": (
                "These meal suggestions are general educational guidance. "
                "They are not a medical prescription and do not claim to treat or cure any condition."
            ),
            "timestamp": datetime.now().isoformat(),
        }), 200
    except Exception as exc:
        logger.exception("Meal plan error: %s", exc)
        return error_response(
            "server_error",
            "Meal plan could not be generated.",
            500,
        )


# --------------------------------------------------------------------------- #
# API: model/info
# --------------------------------------------------------------------------- #
@app.route("/model/info", methods=["GET"])
def model_info():
    if not MODELS_READY:
        return jsonify({"error": "Model not loaded", "message": pipeline.load_error}), 503
    try:
        status = pipeline.health_status()
        return jsonify({
            "model_type": status["model_type"],
            "classes": status["classes"],
            "n_features": status["n_features"],
            "features": status["features"],
            "enhanced_classifier_available": status["enhanced_classifier_available"],
            "note": "Confidence scores are not presented - the saved model's heuristic confidence is not statistically calibrated.",
        }), 200
    except Exception as exc:
        logger.error("Model info error: %s", exc)
        return jsonify({"error": "Could not retrieve model information"}), 500


# --------------------------------------------------------------------------- #
# Error handlers
# --------------------------------------------------------------------------- #
@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "not_found", "message": "Endpoint not found"}), 404
    return not_found_page(error)


def not_found_page(error):
    return jsonify({"success": False, "error": "not_found", "message": "Resource not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "error": "internal_server_error",
            "message": "An unexpected server error occurred.",
        }), 500
    return jsonify({"success": False, "error": "internal_server_error", "message": "An unexpected server error occurred."}), 500


# --------------------------------------------------------------------------- #
# Serve React app (production build)
# --------------------------------------------------------------------------- #
REACT_BUILD_PATH = os.path.join(os.path.dirname(__file__), "..", "client", "dist")


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react_app(path):
    """Serve the built React app for all non-API routes (SPA fallback)."""
    if path.startswith("api/"):
        return jsonify({"success": False, "error": "not_found", "message": "API endpoint not found"}), 404

    try:
        if path == "" or path == "/":
            return send_from_directory(REACT_BUILD_PATH, "index.html")
        full_path = os.path.join(REACT_BUILD_PATH, path)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            return send_from_directory(REACT_BUILD_PATH, path)
        return send_from_directory(REACT_BUILD_PATH, "index.html")
    except Exception as exc:
        logger.error("Error serving React app: %s", exc)
        return jsonify({"error": "Frontend not available. Build the client first."}), 500


if __name__ == "__main__":
    if MODELS_READY:
        logger.info("Starting Flask server on http://0.0.0.0:5000")
        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        logger.error("Failed to load models. Exiting.")
        exit(1)
