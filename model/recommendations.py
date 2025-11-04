import pandas as pd
import numpy as np


class NutrientRecommendationSystem:
    """Standalone nutrient recommendation system for testing.
    This is a trimmed copy of the class used in the notebook, adapted
    so unit tests can instantiate it with a dummy model.
    """
    def __init__(self, malnutrition_model, feature_names):
        self.model = malnutrition_model
        self.feature_names = feature_names

        # Minimal RDA buckets used in tests
        self.nutrient_rda = {
            'normal': {
                'calories_per_kg': 30,
                'protein_per_kg': 0.8,
                'carbs_percent': 55,
                'fats_percent': 30,
                'vitamins': {},
                'minerals': {}
            },
            'moderate': {
                'calories_per_kg': 35,
                'protein_per_kg': 1.2,
                'carbs_percent': 60,
                'fats_percent': 25,
                'vitamins': {},
                'minerals': {}
            },
            'severe': {
                'calories_per_kg': 40,
                'protein_per_kg': 1.5,
                'carbs_percent': 65,
                'fats_percent': 20,
                'vitamins': {},
                'minerals': {}
            }
        }

    def calculate_bmr(self, weight, height, age, gender):
        # height in cm, age in years
        if str(gender).lower() == 'male':
            return (10 * weight) + (6.25 * height) - (5 * age) + 5
        else:
            return (10 * weight) + (6.25 * height) - (5 * age) - 161

    def get_activity_multiplier(self, activity_level):
        multipliers = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }
        return multipliers.get(str(activity_level).lower(), 1.2)

    def engineer_features(self, features):
        df = pd.DataFrame([features])
        weight = features.get('weight_kg')
        height_m = features.get('height_cm', 0) / 100
        # safe BMI
        try:
            df['bmi'] = weight / (height_m * height_m)
        except Exception:
            df['bmi'] = 0

        df['weight_height_ratio'] = weight / max(features.get('height_cm', 1), 1)
        df['bmi_muac_interaction'] = df['bmi'] * features.get('muac_cm', 0)
        df['weight_muac_ratio'] = weight / max(features.get('muac_cm', 1), 1)

        # placeholders
        df['weight_age_percentile'] = 50
        df['height_age_percentile'] = 50
        df['muac_for_age_zscore'] = 0

        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        return df[self.feature_names]

    def generate_recommendations(self, features, activity_level='moderate'):
        X = self.engineer_features(features)
        if hasattr(self.model, 'predict'):
            raw_status = self.model.predict(X)[0]
        else:
            # Handle dummy model (numpy array)
            raw_status = self.model[0] if len(self.model) > 0 else 'moderate'
        status_str = str(raw_status).lower()

        status_key = None
        if status_str in self.nutrient_rda:
            status_key = status_str
        else:
            for k in self.nutrient_rda.keys():
                if k.startswith(status_str) or status_str.startswith(k[:len(status_str)]):
                    status_key = k
                    break
            if status_key is None:
                for k in self.nutrient_rda.keys():
                    if status_str in k or k in status_str:
                        status_key = k
                        break
        if status_key is None:
            status_key = 'moderate'

        rda = self.nutrient_rda[status_key]

        weight = features['weight_kg']
        height = features['height_cm']
        age_years = features['age_months'] / 12
        gender = features.get('gender', 'female')

        bmr = self.calculate_bmr(weight, height, age_years, gender)
        activity_mult = self.get_activity_multiplier(activity_level)
        daily_calories = bmr * activity_mult

        protein_needs = weight * rda['protein_per_kg']
        carbs_calories = (daily_calories * (rda['carbs_percent'] / 100))
        fats_calories = (daily_calories * (rda['fats_percent'] / 100))

        carbs_grams = carbs_calories / 4
        fats_grams = fats_calories / 9

        recommendations = {
            'nutritional_status': status_key,
            'daily_needs': {
                'calories': round(daily_calories),
                'protein': round(protein_needs),
                'carbohydrates': round(carbs_grams),
                'fats': round(fats_grams)
            }
        }

        return recommendations

    def generate_meal_plan(self, recommendations):
        daily = recommendations['daily_needs']['calories']
        meal_distribution = {
            'breakfast': 0.25,
            'morning_snack': 0.1,
            'lunch': 0.3,
            'afternoon_snack': 0.1,
            'dinner': 0.25
        }
        plan = {}
        for k, frac in meal_distribution.items():
            plan[k] = {'calories': round(daily * frac)}
        return plan
