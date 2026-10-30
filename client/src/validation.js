// Shared form validation rules for the NutriCheck assessment.
// These mirror the backend contract in model/ml_pipeline.py so that
// invalid data never reaches the ML model.

export const RULES = {
  age: { min: 1, max: 120, unit: 'years' },
  weight: { min: 2, max: 300, unit: 'kg' },
  height: { min: 30, max: 250, unit: 'cm' },
  muac: { min: 8, max: 60, unit: 'cm' },
};

export const REQUIRED_TEXT = 'This field is required';

export function required(value) {
  return value !== null && value !== undefined && String(value).trim() !== '';
}

export function inRange(value, rule) {
  const num = Number(value);
  if (!Number.isFinite(num)) return false;
  return num >= rule.min && num <= rule.max;
}

export function validateAssessment(values) {
  const errors = {};

  if (!required(values.age)) {
    errors.age = REQUIRED_TEXT;
  } else if (!inRange(values.age, RULES.age)) {
    errors.age = 'Age must be between ' + RULES.age.min + ' and ' + RULES.age.max + ' years';
  }

  if (!required(values.gender)) {
    errors.gender = REQUIRED_TEXT;
  }

  if (!required(values.weight)) {
    errors.weight = REQUIRED_TEXT;
  } else if (!inRange(values.weight, RULES.weight)) {
    errors.weight = 'Weight must be between ' + RULES.weight.min + ' and ' + RULES.weight.max + ' kg';
  }

  if (!required(values.height)) {
    errors.height = REQUIRED_TEXT;
  } else if (!inRange(values.height, RULES.height)) {
    errors.height = 'Height must be between ' + RULES.height.min + ' and ' + RULES.height.max + ' cm';
  }

  if (!required(values.muac)) {
    errors.muac = REQUIRED_TEXT;
  } else if (!inRange(values.muac, RULES.muac)) {
    errors.muac = 'MUAC must be between ' + RULES.muac.min + ' and ' + RULES.muac.max + ' cm';
  }

  return {
    errors,
    valid: Object.keys(errors).length === 0,
  };
}

export function computeBMI(weightKg, heightCm) {
  const h = heightCm / 100;
  if (!h || h <= 0) return null;
  return weightKg / (h * h);
}

export function bmiCategory(bmi) {
  if (bmi == null) return 'not-calculated';
  if (bmi < 18.5) return 'underweight';
  if (bmi < 25) return 'normal';
  if (bmi < 30) return 'overweight';
  return 'obese';
}

export const SYMPTOM_OPTIONS = [
  { label: 'Fatigue / low energy', value: 'fatigue' },
  { label: 'Muscle weakness', value: 'weakness' },
  { label: 'Hair loss', value: 'hairloss' },
  { label: 'Skin problems', value: 'skin' },
  { label: 'Unexplained weight loss', value: 'weight-loss' },
  { label: 'Unexplained weight gain', value: 'weight-gain' },
  { label: 'Bone / joint pain', value: 'bones' },
  { label: 'Digestive issues', value: 'digestive' },
  { label: 'No symptoms - I currently feel healthy', value: 'no-symptoms' },
];

export const ACTUAL_SYMPTOM_VALUES = SYMPTOM_OPTIONS
  .filter((s) => s.value !== 'no-symptoms')
  .map((s) => s.value);
