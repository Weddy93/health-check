from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import traceback
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:5000", "https://health-check-ruby.vercel.app"])  # Enable CORS for specific origins

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
model = None
scaler = None
label_encoder = None
enhanced_classifier = None

def load_models():
    """Load all required models and preprocessing objects"""
    global model, scaler, label_encoder, enhanced_classifier

    try:
        # Load base model and preprocessing
        model = joblib.load('malnutrition_model.joblib')
        scaler = joblib.load('feature_scaler.joblib')
        label_encoder = joblib.load('label_encoder.joblib')

        # Load enhanced classifier if available
        if os.path.exists('enhanced_malnutrition_classifier.joblib'):
            enhanced_classifier = joblib.load('enhanced_malnutrition_classifier.joblib')
            logger.info("Enhanced classifier loaded successfully")
        else:
            logger.warning("Enhanced classifier not found, using base model only")

        logger.info("All models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def calculate_zscore(measurement, age_months, gender, indicator):
    """Calculate z-score based on WHO standards"""
    if indicator == 'wfa':
        expected_weight = (age_months * 0.3) + 3.5
        sd = expected_weight * 0.1
        return (measurement - expected_weight) / sd
    elif indicator == 'hfa':
        expected_height = (age_months * 0.5) + 50
        sd = expected_height * 0.05
        return (measurement - expected_height) / sd
    elif indicator == 'bfa':
        expected_bmi = 16 + (age_months * 0.05)
        sd = 2
        return (measurement - expected_bmi) / sd
    elif indicator == 'mfa':
        expected_muac = 14 + (age_months * 0.02)
        sd = 1.5
        return (measurement - expected_muac) / sd
    return 0

def engineer_features(df):
    """Add enhanced features for better prediction"""
    enhanced = df.copy()

    # Calculate z-scores
    enhanced['weight_for_age_z'] = enhanced.apply(
        lambda x: calculate_zscore(x['weight_kg'], x['age_months'], x.get('gender', 'F'), 'wfa'), axis=1)
    enhanced['height_for_age_z'] = enhanced.apply(
        lambda x: calculate_zscore(x['height_cm'], x['age_months'], x.get('gender', 'F'), 'hfa'), axis=1)
    enhanced['bmi_for_age_z'] = enhanced.apply(
        lambda x: calculate_zscore(x['bmi'], x['age_months'], x.get('gender', 'F'), 'bfa'), axis=1)
    enhanced['muac_for_age_z'] = enhanced.apply(
        lambda x: calculate_zscore(x['muac_cm'], x['age_months'], x.get('gender', 'F'), 'mfa'), axis=1)

    # Add clinical indicators
    enhanced['stunting_score'] = enhanced['height_for_age_z'].apply(lambda x: 1 if x < -2 else 0)
    enhanced['wasting_score'] = enhanced['weight_for_age_z'].apply(lambda x: 1 if x < -2 else 0)
    enhanced['underweight_score'] = enhanced['bmi_for_age_z'].apply(lambda x: 1 if x < -2 else 0)
    enhanced['weight_height_interaction'] = enhanced['weight_for_age_z'] * enhanced['height_for_age_z']
    enhanced['bmi_muac_interaction'] = enhanced['bmi_for_age_z'] * enhanced['muac_for_age_z']

    return enhanced

def validate_input_data(data):
    """Validate input data for prediction"""
    required_fields = ['age_months', 'weight_kg', 'height_cm', 'muac_cm', 'bmi']
    errors = []

    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
            continue

        value = data[field]
        if not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"Invalid value for {field}: must be positive number")

        # Field-specific validation
        if field == 'age_months' and (value < 1 or value > 240):
            errors.append("Age must be between 1-240 months (0-20 years)")
        elif field == 'weight_kg' and (value < 2 or value > 200):
            errors.append("Weight must be between 2-200 kg")
        elif field == 'height_cm' and (value < 30 or value > 250):
            errors.append("Height must be between 30-250 cm")
        elif field == 'muac_cm' and (value < 8 or value > 40):
            errors.append("MUAC must be between 8-40 cm")
        elif field == 'bmi' and (value < 8 or value > 60):
            errors.append("BMI must be between 8-60")

    return errors

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'enhanced_classifier_loaded': enhanced_classifier is not None
    })

@app.route('/api/assess-health', methods=['POST'])
def assess_health():
    """Assess health based on user profile data"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract profile data
        name = data.get('name', 'Unknown')
        age_years = data.get('age', 25)
        gender = data.get('gender', 'female')
        weight_kg = data.get('weight', 60)
        height_cm = data.get('height', 160)
        activity = data.get('activity', 'moderate')
        diet = data.get('diet', 'balanced')
        life_stage = data.get('lifeStage', '')
        symptoms = data.get('symptoms', [])

        # Calculate BMI
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)

        # Convert age to months
        age_months = age_years * 12

        # Estimate MUAC based on BMI and age (rough approximation)
        # Normal MUAC is typically 13.5-16.5 cm for adults
        if bmi < 18.5:
            muac_cm = 13.0  # Underweight
        elif bmi < 25:
            muac_cm = 14.5  # Normal
        elif bmi < 30:
            muac_cm = 15.5  # Overweight
        else:
            muac_cm = 16.0  # Obese

        # Prepare data for malnutrition prediction
        prediction_data = {
            'age_months': age_months,
            'weight_kg': weight_kg,
            'height_cm': height_cm,
            'muac_cm': muac_cm,
            'bmi': bmi
        }

        # Validate input
        validation_errors = validate_input_data(prediction_data)
        if validation_errors:
            # If validation fails, provide basic assessment
            assessment = {
                'health_status': 'unknown',
                'bmi': round(bmi, 1),
                'bmi_category': 'unknown',
                'diagnosis': 'Unable to assess malnutrition status due to invalid data',
                'nutrients_needed': [],
                'recommendations': ['Please ensure all measurements are accurate']
            }
        else:
            # Make malnutrition prediction
            input_df = pd.DataFrame([prediction_data])
            input_df['gender'] = gender[0].upper() if gender else 'F'

            enhanced_data = engineer_features(input_df)
            features = enhanced_data[['age_months', 'weight_kg', 'height_cm', 'muac_cm', 'bmi']]
            features_scaled = scaler.transform(features)

            if enhanced_classifier is not None:
                prediction_encoded, confidence = enhanced_classifier.predict_with_confidence(
                    pd.DataFrame(features_scaled, columns=features.columns)
                )
                confidence_score = float(confidence[0])
            else:
                prediction_encoded = model.predict(features_scaled)
                confidence_score = None

            prediction = label_encoder.inverse_transform(prediction_encoded)[0]

            # Determine BMI category
            if bmi < 18.5:
                bmi_category = 'underweight'
            elif bmi < 25:
                bmi_category = 'normal'
            elif bmi < 30:
                bmi_category = 'overweight'
            else:
                bmi_category = 'obese'

            # Create assessment response
            is_malnourished = prediction in ['Moderate', 'Severe']

            assessment = {
                'health_status': 'malnourished' if is_malnourished else 'healthy',
                'bmi': round(bmi, 1),
                'bmi_category': bmi_category,
                'diagnosis': f"{prediction} malnutrition detected" if is_malnourished else "Normal nutritional status",
                'nutrients_needed': [] if not is_malnourished else ['Protein', 'Iron', 'Vitamin A', 'Vitamin D'],
                'malnutrition_prediction': prediction,
                'confidence_score': confidence_score
            }

        # Add life stage considerations
        if life_stage:
            assessment['life_stage'] = life_stage
            if life_stage == 'pregnant':
                assessment['nutrients_needed'].extend(['Folic Acid', 'Calcium', 'Iron'])
            elif life_stage == 'lactating':
                assessment['nutrients_needed'].extend(['Calcium', 'Vitamin D', 'Omega-3'])

        # Add symptom-based recommendations
        if symptoms:
            symptom_nutrients = {
                'fatigue': ['Iron', 'Vitamin B12'],
                'weakness': ['Protein', 'Iron'],
                'hairloss': ['Iron', 'Zinc', 'Vitamin D'],
                'skin': ['Vitamin A', 'Zinc'],
                'weight-loss': ['Protein', 'Calories'],
                'weight-gain': ['Calories', 'Healthy Fats'],
                'bones': ['Calcium', 'Vitamin D'],
                'digestive': ['Fiber', 'Probiotics']
            }

            for symptom in symptoms:
                if symptom in symptom_nutrients:
                    assessment['nutrients_needed'].extend(symptom_nutrients[symptom])

            # Remove duplicates
            assessment['nutrients_needed'] = list(set(assessment['nutrients_needed']))

        assessment['timestamp'] = datetime.now().isoformat()
        assessment['profile_summary'] = {
            'name': name,
            'age': age_years,
            'gender': gender,
            'activity_level': activity,
            'diet_type': diet
        }

        # Add food recommendations based on nutrients needed
        if assessment['nutrients_needed']:
            nutrient_foods = {
                'Protein': ['eggs', 'chicken', 'fish', 'beans', 'nuts'],
                'Iron': ['spinach', 'red meat', 'lentils', 'fortified cereals'],
                'Calcium': ['dairy', 'leafy greens', 'fortified plant milks'],
                'Vitamin A': ['carrots', 'sweet potatoes', 'leafy greens'],
                'Vitamin D': ['fatty fish', 'fortified foods', 'sun exposure'],
                'Zinc': ['meat', 'shellfish', 'legumes', 'nuts'],
                'Folic Acid': ['leafy greens', 'citrus fruits', 'beans'],
                'Omega-3': ['fatty fish', 'flaxseeds', 'walnuts'],
                'Vitamin B12': ['meat', 'dairy', 'eggs', 'fortified cereals']
            }
            recommendations = []
            for nutrient in assessment['nutrients_needed']:
                if nutrient in nutrient_foods:
                    recommendations.append({
                        'nutrient': nutrient,
                        'foods': nutrient_foods[nutrient]
                    })
            assessment['recommendations'] = recommendations

        logger.info(f"Health assessment completed for {name}")
        return jsonify(assessment)

    except Exception as e:
        logger.error(f"Assessment error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Assessment failed',
            'message': str(e),
            'health_status': 'unknown',
            'bmi': 0,
            'bmi_category': 'unknown',
            'diagnosis': 'Assessment could not be completed',
            'nutrients_needed': []
        }), 500

@app.route('/api/generate-meal-plan', methods=['POST'])
def generate_meal_plan():
    """Generate personalized meal plan based on assessment"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No assessment data provided'}), 400

        # Extract assessment data
        health_status = data.get('healthStatus', 'healthy')
        nutrients_needed = data.get('nutrients', [])
        diagnosis = data.get('diagnosis', '')
        bmi = data.get('bmi', 22)

        # Generate meal plan based on needs
        meal_plan = {
            'breakfast': [],
            'lunch': [],
            'dinner': [],
            'snacks': [],
            'recommendations': []
        }

        # Basic meal plan structure
        if health_status == 'healthy':
            meal_plan['breakfast'] = ['Oatmeal with fruits', 'Whole grain toast with avocado', 'Greek yogurt with berries']
            meal_plan['lunch'] = ['Grilled chicken salad', 'Quinoa bowl with vegetables', 'Turkey sandwich on whole grain bread']
            meal_plan['dinner'] = ['Baked salmon with vegetables', 'Stir-fried tofu with brown rice', 'Lean beef with sweet potatoes']
            meal_plan['snacks'] = ['Apple with almond butter', 'Carrot sticks with hummus', 'Handful of nuts']
            meal_plan['recommendations'] = ['Continue balanced diet', 'Stay hydrated', 'Regular exercise']
        else:
            # Malnourished - focus on nutrient-dense foods
            meal_plan['breakfast'] = ['Eggs with spinach and whole grain toast', 'Smoothie with protein powder and fruits', 'Avocado toast with eggs']
            meal_plan['lunch'] = ['Lentil soup with vegetables', 'Chicken stir-fry with brown rice', 'Salmon salad with quinoa']
            meal_plan['dinner'] = ['Beef stew with vegetables', 'Tuna with sweet potatoes', 'Chickpea curry with rice']
            meal_plan['snacks'] = ['Greek yogurt with nuts', 'Protein bars', 'Fruit with cheese']
            meal_plan['recommendations'] = ['Focus on protein-rich foods', 'Include healthy fats', 'Eat nutrient-dense foods']

        # Add specific recommendations based on nutrients needed
        nutrient_foods = {
            'Protein': ['eggs', 'chicken', 'fish', 'beans', 'nuts'],
            'Iron': ['spinach', 'red meat', 'lentils', 'fortified cereals'],
            'Calcium': ['dairy', 'leafy greens', 'fortified plant milks'],
            'Vitamin A': ['carrots', 'sweet potatoes', 'leafy greens'],
            'Vitamin D': ['fatty fish', 'fortified foods', 'sun exposure'],
            'Zinc': ['meat', 'shellfish', 'legumes', 'nuts'],
            'Folic Acid': ['leafy greens', 'citrus fruits', 'beans'],
            'Omega-3': ['fatty fish', 'flaxseeds', 'walnuts']
        }

        for nutrient in nutrients_needed:
            if nutrient in nutrient_foods:
                meal_plan['recommendations'].append(f"Include more {nutrient.lower()}-rich foods: {', '.join(nutrient_foods[nutrient])}")

        meal_plan['timestamp'] = datetime.now().isoformat()
        meal_plan['based_on'] = {
            'health_status': health_status,
            'nutrients_needed': nutrients_needed,
            'bmi': bmi
        }

        logger.info("Meal plan generated successfully")
        return jsonify(meal_plan)

    except Exception as e:
        logger.error(f"Meal plan generation error: {str(e)}")
        return jsonify({
            'error': 'Meal plan generation failed',
            'message': str(e),
            'meal_plan': {
                'breakfast': ['Consult a nutritionist for personalized meal plan'],
                'lunch': ['Consult a nutritionist for personalized meal plan'],
                'dinner': ['Consult a nutritionist for personalized meal plan'],
                'snacks': ['Consult a nutritionist for personalized meal plan'],
                'recommendations': ['Please consult a healthcare professional for personalized nutrition advice']
            }
        }), 500

@app.route('/predict', methods=['POST'])
def predict_malnutrition():
    """Predict malnutrition status for a single individual"""
    try:
        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate input
        validation_errors = validate_input_data(data)
        if validation_errors:
            return jsonify({
                'error': 'Validation failed',
                'details': validation_errors
            }), 400

        # Prepare input data
        input_df = pd.DataFrame([data])

        # Add gender if not provided
        if 'gender' not in input_df.columns:
            input_df['gender'] = 'F'

        # Engineer features
        enhanced_data = engineer_features(input_df)

        # Prepare features for model - use only the basic features that the model was trained on
        basic_features = ['age_months', 'weight_kg', 'height_cm', 'muac_cm', 'bmi']
        features = enhanced_data[basic_features]

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        if enhanced_classifier is not None:
            # Use enhanced classifier with confidence
            prediction_encoded, confidence = enhanced_classifier.predict_with_confidence(
                pd.DataFrame(features_scaled, columns=features.columns)
            )
            confidence_score = float(confidence[0])
        else:
            # Fallback to base model
            prediction_encoded = model.predict(features_scaled)
            confidence_score = None

        # Decode prediction
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]

        # Prepare response
        response = {
            'prediction': prediction,
            'timestamp': datetime.now().isoformat(),
            'input_data': data,
            'confidence_score': confidence_score,
            'model_version': 'enhanced' if enhanced_classifier else 'base'
        }

        # Add interpretation
        if prediction == 'Normal':
            response['interpretation'] = 'Normal nutritional status'
            response['recommendation'] = 'Continue with balanced diet and regular health check-ups'
        elif prediction == 'Moderate':
            response['interpretation'] = 'Moderate malnutrition detected'
            response['recommendation'] = 'Consult healthcare provider for nutritional intervention'
        elif prediction == 'Severe':
            response['interpretation'] = 'Severe malnutrition detected'
            response['recommendation'] = 'Immediate medical attention required'

        logger.info(f"Prediction made: {prediction} (confidence: {confidence_score})")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict malnutrition status for multiple individuals"""
    try:
        # Get JSON data
        data = request.get_json()

        if not data or not isinstance(data, list):
            return jsonify({'error': 'Expected list of individual data'}), 400

        if len(data) > 100:
            return jsonify({'error': 'Batch size limited to 100 predictions'}), 400

        results = []

        for i, individual_data in enumerate(data):
            try:
                # Validate individual input
                validation_errors = validate_input_data(individual_data)
                if validation_errors:
                    results.append({
                        'index': i,
                        'error': 'Validation failed',
                        'details': validation_errors
                    })
                    continue

                # Prepare input data
                input_df = pd.DataFrame([individual_data])

                # Add gender if not provided
                if 'gender' not in input_df.columns:
                    input_df['gender'] = 'F'

                # Engineer features
                enhanced_data = engineer_features(input_df)

                # Prepare features for model
                features = enhanced_data.drop(['gender'], axis=1, errors='ignore')

                # Scale features
                features_scaled = scaler.transform(features)

                # Make prediction
                if enhanced_classifier is not None:
                    prediction_encoded, confidence = enhanced_classifier.predict_with_confidence(
                        pd.DataFrame(features_scaled, columns=features.columns)
                    )
                    confidence_score = float(confidence[0])
                else:
                    prediction_encoded = model.predict(features_scaled)
                    confidence_score = None

                # Decode prediction
                prediction = label_encoder.inverse_transform(prediction_encoded)[0]

                results.append({
                    'index': i,
                    'prediction': prediction,
                    'confidence_score': confidence_score,
                    'input_data': individual_data
                })

            except Exception as e:
                results.append({
                    'index': i,
                    'error': f'Prediction failed: {str(e)}'
                })

        return jsonify({
            'total_predictions': len(data),
            'successful_predictions': len([r for r in results if 'prediction' in r]),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get feature importance
        feature_importance = {}
        for feature, importance in zip(scaler.feature_names_in_, model.feature_importances_):
            feature_importance[feature] = float(importance)

        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            'model_type': type(model).__name__,
            'classes': label_encoder.classes_.tolist(),
            'feature_importance': dict(sorted_features),
            'enhanced_classifier_available': enhanced_classifier is not None,
            'n_features': len(scaler.feature_names_in_),
            'last_updated': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'error': 'Could not retrieve model information',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    """Serve React app for all non-API routes"""
    if path.startswith('api/'):
        return not_found(404)

    # Try to serve the file from the React build directory
    try:
        # Path to the React build directory
        react_build_path = os.path.join(os.path.dirname(__file__), '..', 'client', 'dist')

        # If the path is empty or points to a file that exists, serve it
        if path == '' or path == '/':
            return send_from_directory(react_build_path, 'index.html')
        else:
            # Check if the file exists
            full_path = os.path.join(react_build_path, path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                return send_from_directory(react_build_path, path)
            else:
                # For SPA routing, serve index.html for any non-API route
                return send_from_directory(react_build_path, 'index.html')
    except Exception as e:
        logger.error(f"Error serving React app: {str(e)}")
        return jsonify({'error': 'Frontend not available'}), 500

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        logger.info("Starting Flask server with integrated React frontend...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to load models. Exiting.")
        exit(1)
