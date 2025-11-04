import math
import numpy as np
from recommendations import NutrientRecommendationSystem


class DummyModel:
    def __init__(self, return_value):
        # return_value can be string or numpy string
        self.return_value = return_value

    def predict(self, X):
        # return iterable with a single element
        return np.array([self.return_value], dtype=object)


def test_status_mapping_handles_truncated_label():
    # Model returns a truncated numpy string
    model = DummyModel(np.str_('modera'))
    feature_names = ['bmi', 'weight_height_ratio', 'bmi_muac_interaction', 'weight_muac_ratio',
                     'weight_age_percentile', 'height_age_percentile', 'muac_for_age_zscore']
    system = NutrientRecommendationSystem(model, feature_names)

    features = {
        'age_months': 36,
        'weight_kg': 15,
        'height_cm': 95,
        'muac_cm': 16,
        'gender': 'female'
    }

    rec = system.generate_recommendations(features)
    assert rec['nutritional_status'] == 'moderate'


def test_macronutrient_calculation_matches_formula():
    # Model returns 'normal'
    model = DummyModel('normal')
    feature_names = ['bmi', 'weight_height_ratio', 'bmi_muac_interaction', 'weight_muac_ratio',
                     'weight_age_percentile', 'height_age_percentile', 'muac_for_age_zscore']
    system = NutrientRecommendationSystem(model, feature_names)

    features = {
        'age_months': 360,  # 30 years
        'weight_kg': 60,
        'height_cm': 170,
        'muac_cm': 25,
        'gender': 'male'
    }

    rec = system.generate_recommendations(features, activity_level='moderate')

    # Expected protein = weight * 0.8 (for 'normal')
    expected_protein = 60 * 0.8
    assert rec['daily_needs']['protein'] == round(expected_protein)

    # Calculate expected calories using same formula
    age_years = features['age_months'] / 12
    bmr = system.calculate_bmr(features['weight_kg'], features['height_cm'], age_years, features['gender'])
    daily_calories_expected = bmr * system.get_activity_multiplier('moderate')
    assert rec['daily_needs']['calories'] == round(daily_calories_expected)


def test_meal_plan_calorie_distribution_sums_to_daily():
    model = DummyModel('normal')
    feature_names = ['bmi', 'weight_height_ratio', 'bmi_muac_interaction', 'weight_muac_ratio',
                     'weight_age_percentile', 'height_age_percentile', 'muac_for_age_zscore']
    system = NutrientRecommendationSystem(model, feature_names)

    features = {
        'age_months': 360,
        'weight_kg': 60,
        'height_cm': 170,
        'muac_cm': 25,
        'gender': 'male'
    }

    rec = system.generate_recommendations(features, activity_level='moderate')
    plan = system.generate_meal_plan(rec)

    total = sum(slot['calories'] for slot in plan.values())
    # allow small rounding differences (<= 3 calories)
    assert abs(total - rec['daily_needs']['calories']) <= 3


def test_handles_missing_optional_fields_and_defaults():
    # Ensure missing muac or gender doesn't crash and defaults are used
    model = DummyModel('normal')
    feature_names = ['bmi', 'weight_height_ratio', 'bmi_muac_interaction', 'weight_muac_ratio',
                     'weight_age_percentile', 'height_age_percentile', 'muac_for_age_zscore']
    system = NutrientRecommendationSystem(model, feature_names)

    features = {
        'age_months': 24,
        'weight_kg': 12,
        'height_cm': 85
        # muac_cm and gender intentionally missing
    }

    rec = system.generate_recommendations(features)
    assert 'daily_needs' in rec


def test_unknown_activity_level_uses_default_multiplier():
    model = DummyModel('normal')
    feature_names = ['bmi', 'weight_height_ratio', 'bmi_muac_interaction', 'weight_muac_ratio',
                     'weight_age_percentile', 'height_age_percentile', 'muac_for_age_zscore']
    system = NutrientRecommendationSystem(model, feature_names)

    features = {
        'age_months': 360,
        'weight_kg': 60,
        'height_cm': 170,
        'muac_cm': 25,
        'gender': 'male'
    }

    # use a nonsense activity level
    rec = system.generate_recommendations(features, activity_level='unknown_level')
    # default multiplier is 1.2 so calories should equal BMR * 1.2
    age_years = features['age_months'] / 12
    expected = round(system.calculate_bmr(features['weight_kg'], features['height_cm'], age_years, features['gender']) * 1.2)
    assert rec['daily_needs']['calories'] == expected


def test_infant_small_values_do_not_crash_and_compute_reasonable_values():
    model = DummyModel('severe')
    feature_names = ['bmi', 'weight_height_ratio', 'bmi_muac_interaction', 'weight_muac_ratio',
                     'weight_age_percentile', 'height_age_percentile', 'muac_for_age_zscore']
    system = NutrientRecommendationSystem(model, feature_names)

    # infant case: 6 months, small weight and height
    features = {
        'age_months': 6,
        'weight_kg': 6,
        'height_cm': 65,
        'muac_cm': 11,
        'gender': 'female'
    }

    rec = system.generate_recommendations(features)
    # severe status increases protein per kg so expect protein >= weight * 1.0
    assert rec['daily_needs']['protein'] >= round(features['weight_kg'] * 1.0)


def test_zero_or_tiny_height_handling_prevents_divide_by_zero():
    model = DummyModel('normal')
    feature_names = ['bmi', 'weight_height_ratio', 'bmi_muac_interaction', 'weight_muac_ratio',
                     'weight_age_percentile', 'height_age_percentile', 'muac_for_age_zscore']
    system = NutrientRecommendationSystem(model, feature_names)

    features = {
        'age_months': 120,
        'weight_kg': 30,
        'height_cm': 0,  # malformed
        'muac_cm': 0,
        'gender': 'male'
    }

    rec = system.generate_recommendations(features)
    assert 'daily_needs' in rec
