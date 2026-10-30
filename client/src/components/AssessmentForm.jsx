import { useState, useCallback } from 'react';
import {
  validateAssessment,
  computeBMI,
  bmiCategory,
  SYMPTOM_OPTIONS,
  ACTUAL_SYMPTOM_VALUES,
} from '../validation';
import { postAssessment } from '../api';

// Screen 2: Guided health assessment (four steps).
// The backend (Flask) is the single authority for the ML classification;
// the frontend only validates input, shows a client-side BMI estimate, and
// sends the assessment to /api/assess-health.
const STEPS = [
  { id: 'personal', title: 'Personal information' },
  { id: 'measurements', title: 'Physical measurements' },
  { id: 'lifestyle', title: 'Lifestyle & diet' },
  { id: 'symptoms', title: 'Symptoms' },
];

const LIFE_STAGE_OPTIONS = [
  { value: 'none', label: 'None' },
  { value: 'pregnant', label: 'Pregnant' },
  { value: 'lactating', label: 'Lactating' },
];

const ACTIVITY_OPTIONS = [
  { value: 'sedentary', label: 'Sedentary (little or no exercise)' },
  { value: 'light', label: 'Light (1-3 times/week)' },
  { value: 'moderate', label: 'Moderate (3-5 times/week)' },
  { value: 'active', label: 'Active (daily exercise)' },
  { value: 'very-active', label: 'Very active (intense daily exercise)' },
];

const DIET_OPTIONS = [
  { value: 'balanced', label: 'Balanced' },
  { value: 'vegetarian', label: 'Vegetarian' },
  { value: 'vegan', label: 'Vegan' },
  { value: 'keto', label: 'Keto' },
  { value: 'high-protein', label: 'High protein' },
  { value: 'unhealthy', label: 'Unhealthy / junk food' },
];

export default function AssessmentForm({ onAssessmentComplete }) {
  const [stepIndex, setStepIndex] = useState(0);
  const [form, setForm] = useState({
    name: '',
    age: '',
    gender: '',
    life_stage: 'none',
    weight: '',
    height: '',
    muac: '',
    activity: '',
    diet: '',
    symptoms: [],
  });
  const [errors, setErrors] = useState({});
  const [touch, setTouch] = useState({});
  const [loading, setLoading] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  const currentStep = STEPS[stepIndex];
  const isLastStep = stepIndex === STEPS.length - 1;

  const setField = useCallback((key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
    // Clear field error once the user starts correcting it.
    if (errors[key]) {
      setErrors((prev) => { const next = { ...prev }; delete next[key]; return next; });
    }
  }, [errors]);

  // --- "No symptoms" mutual exclusion ----------------------------------- //
  const toggleSymptom = useCallback((value) => {
    setForm((prev) => {
      const current = prev.symptoms || [];
      if (value === 'no-symptoms') {
        // "No symptoms" excludes every actual symptom.
        return { ...prev, symptoms: current.includes(value) ? current : [value] };
      }
      if (current.includes('no-symptoms')) {
        // Picking an actual symptom removes "No symptoms".
        return { ...prev, symptoms: [...current.filter((s) => s !== 'no-symptoms'), value] };
      }
      return {
        ...prev,
        symptoms: current.includes(value) ? current.filter((s) => s !== value) : [...current, value],
      };
    });
  }, []);

  const isSymptomSelected = (value) => (form.symptoms || []).includes(value);

  const stepErrors = () => {
    const next = {};
    const { errors: all } = validateAssessment({
      age: form.age,
      gender: form.gender,
      weight: form.weight,
      height: form.height,
      muac: form.muac,
    });
    Object.assign(next, all);
    if (stepIndex === 0 && !form.gender) next.gender = 'Please select a gender';
    return next;
  };

  const fieldError = (key) => {
    const current = stepErrors();
    return touch[key] && current[key] ? current[key] : null;
  };

  const setTouched = () => {
    setTouch(
      Object.fromEntries(
        ['age', 'gender', 'weight', 'height', 'muac'].map((k) => [k, true])
      )
    );
  };

  const handleNext = () => {
    const current = stepErrors();
    setTouched();
    setErrors(current);
    if (Object.keys(current).length > 0) return;
    setStepIndex((i) => (i < STEPS.length - 1 ? i + 1 : i));
  };

  const handleBack = () => setStepIndex((i) => (i > 0 ? i - 1 : 0));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitError(null);
    const current = stepErrors();
    setTouched();
    setErrors(current);
    if (Object.keys(current).length > 0) return;

    const weight = Number(form.weight);
    const height = Number(form.height);
    const bmiEstimate = computeBMI(weight, height);

    setLoading(true);
    try {
      const result = await postAssessment({
        ...form,
        weight: weight,
        height: height,
        muac: Number(form.muac),
        age: Number(form.age),
        bmi_estimate: bmiEstimate ? Number(bmiEstimate.toFixed(1)) : null,
      });
      setLoading(false);
      onAssessmentComplete({
        ...result,
        client_bmi_estimate: bmiEstimate ? Number(bmiEstimate.toFixed(1)) : null,
        client_bmi_category: bmiEstimate ? bmiCategory(bmiEstimate) : null,
      });
    } catch (error) {
      setLoading(false);
      setSubmitError(error.message || 'Assessment failed. Please try again.');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const renderField = ({ id, label, type, value, placeholder, unit, required = true }) => {
    const err = fieldError(id);
    return (
      <div key={id} className="space-y-2">
        <label className="block text-sm font-semibold text-slate-700">
          {label}
          {required && <span className="text-brand ml-1">*</span>}
          {unit && <span className="text-slate-400 text-xs font-normal ml-2">({unit})</span>}
        </label>
        <input
          id={id}
          type={type}
          value={value}
          onChange={(e) => setField(id, e.target.value)}
          onBlur={() => setTouch((prev) => ({ ...prev, [id]: true }))}
          placeholder={placeholder}
          className={`w-full p-3 rounded-xl border text-slate-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand transition-all ${
            err
              ? 'border-red-500 focus-visible:ring-red-500'
              : 'border-slate-300 focus-visible:border-brand'
          }`}
        />
        {err && (
          <p className="text-xs text-red-600 font-medium">{err}</p>
        )}
      </div>
    );
  };

  const renderSelect = ({ id, label, options, value, required = true }) => {
    const err = fieldError(id);
    return (
      <div className="space-y-2">
        <label className="block text-sm font-semibold text-slate-700">
          {label}
          {required && <span className="text-brand ml-1">*</span>}
        </label>
        <select
          id={id}
          value={value}
          onChange={(e) => setField(id, e.target.value)}
          onBlur={() => setTouch((prev) => ({ ...prev, [id]: true }))}
          className="w-full p-3 rounded-xl border border-slate-300 text-slate-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand transition-all"
        >
          <option value="">Select</option>
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        {err && <p className="text-xs text-red-600 font-medium">{err}</p>}
      </div>
    );
  };

  const step = stepIndex;
  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      <div className="mb-10">
        <h1 className="font-display font-bold text-3xl text-slate-900 mb-2">Health Assessment</h1>
        <p className="text-slate-600">
          Tell us about your health so we can give you useful nutritional guidance.
        </p>
      </div>
      <div className="mb-8">
        <div className="flex items-center justify-between text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">
          <span>Step {step + 1} of {STEPS.length}</span>
          <span>{currentStep.title}</span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-slate-200 overflow-hidden">
          <div
            className="h-full bg-brand"
            style={{ width: `${((step + 1) / STEPS.length) * 100}%` }}
          />
        </div>
      </div>
      <form onSubmit={handleSubmit} className="space-y-8">
        <section
          id="step-personal"
          className={`${
            step === 0 ? 'block' : 'hidden'
          } animate-fade-in space-y-6`}
        >
          <div className="grid md:grid-cols-2 gap-6">
            {renderField({
              id: 'name',
              label: 'Full name (optional)',
              type: 'text',
              value: form.name,
              placeholder: 'e.g. Maria',
              required: false,
            })}
            {renderSelect({
              id: 'gender',
              label: 'Gender',
              options: [
                { value: 'male', label: 'Male' },
                { value: 'female', label: 'Female' },
                { value: 'other', label: 'Other / prefer not to say' },
              ],
              value: form.gender,
            })}
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-semibold text-slate-700">
              Life stage <span className="text-slate-400 text-xs font-normal ml-2">(optional)</span>
            </label>
            <div className="grid md:grid-cols-3 gap-3">
              {LIFE_STAGE_OPTIONS.map((opt) => (
                <label
                  key={opt.value}
                  className={`flex items-center gap-3 p-3 rounded-xl border cursor-pointer text-sm ${
                    form.life_stage === opt.value
                      ? 'bg-brand-50 border-brand text-brand font-semibold'
                      : 'bg-white border-slate-300 text-slate-700 hover:border-slate-400 transition-all'
                  }`}
                >
                  <input
                    type="radio"
                    name="life_stage"
                    value={opt.value}
                    checked={form.life_stage === opt.value}
                    onChange={() => setField('life_stage', opt.value)}
                    className="w-4 h-4 text-brand"
                  />
                  {opt.label}
                </label>
              ))}
            </div>
          </div>
        </section>
        <section
          id="step-measurements"
          className={`${
            step === 1 ? 'block' : 'hidden'
          } animate-fade-in space-y-6`}
        >
          <div className="grid md:grid-cols-3 gap-6">
            {renderField({
              id: 'weight',
              label: 'Weight',
              type: 'number',
              value: form.weight,
              placeholder: 'e.g. 70',
              unit: 'kg',
            })}
            {renderField({
              id: 'height',
              label: 'Height',
              type: 'number',
              value: form.height,
              placeholder: 'e.g. 170',
              unit: 'cm',
            })}
            {renderField({
              id: 'muac',
              label: 'Mid-upper arm circumference (MUAC)',
              type: 'number',
              value: form.muac,
              placeholder: 'e.g. 26',
              unit: 'cm',
            })}
          </div>
          <div className="bg-brand-50 p-5 rounded-2xl border border-brand-100">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-brand-900">Estimated BMI</span>
              <span className="text-sm font-mono text-slate-600">
                {form.weight && form.height
                  ? (computeBMI(Number(form.weight), Number(form.height)) || 0).toFixed(1)
                  : '—'}
              </span>
            </div>
            <div className="flex items-center gap-2 text-sm text-slate-600">
              <span className="w-2 h-2 rounded-full bg-brand" />
              Client-side estimate only — the server performs the authoritative calculation.
            </div>
          </div>
        </section>

        {/* Step 3: Lifestyle */}
        <section
          id="step-lifestyle"
          className={`${
            step === 2 ? 'block' : 'hidden'
          } animate-fade-in space-y-6`}
        >
          <div className="space-y-4">
            {renderSelect({
              id: 'activity',
              label: 'Activity level',
              options: ACTIVITY_OPTIONS,
              value: form.activity,
            })}
            {renderSelect({
              id: 'diet',
              label: 'Usual diet type',
              options: DIET_OPTIONS,
              value: form.diet,
            })}
          </div>
        </section>

        {/* Step 4: Symptoms */}
        <section id="step-symptoms" className="animate-fade-in space-y-6">
          <div className="bg-slate-50 p-5 rounded-2xl border border-slate-200 mb-4">
            <p className="text-sm text-slate-600">
              Select any symptoms you currently experience. If you feel healthy, choose
              <strong>"No symptoms"</strong> — it will clear any other selections.
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-3">
            {SYMPTOM_OPTIONS.map((symptom) => (
              <label key={symptom.value} className="cursor-pointer">
                <button
                  type="button"
                  onClick={() => toggleSymptom(symptom.value)}
                  className={`w-full flex flex-col items-center gap-2 p-4 rounded-xl border transition-all ${
                    isSymptomSelected(symptom.value)
                      ? 'bg-brand-50 border-brand-200 text-brand font-medium shadow-sm'
                      : 'bg-white border-slate-200 text-slate-700 hover:border-slate-300 hover:bg-slate-50'
                  }`}
                >
                  <span className="text-2xl">{symptom.value === 'no-symptoms' ? '😊' : '🩺'}</span>
                  <span className="text-sm font-medium leading-tight text-center">{symptom.label}</span>
                </button>
              </label>
            ))}
          </div>
          {(form.symptoms || []).length > 0 && (
            <p className="text-xs text-slate-500 text-center italic">
              Selected: {(form.symptoms || [])
                .map((s) => SYMPTOM_OPTIONS.find((o) => o.value === s)?.label || s)
                .join(', ')}
            </p>
          )}
        </section>

        {/* Footer controls */}
        <div className="flex flex-col sm:flex-row gap-4 justify-between items-end pt-6 border-t border-slate-200">
          <button
            type="button"
            onClick={handleBack}
            disabled={step === 0 || loading}
            className="px-6 py-3 rounded-xl border border-slate-300 text-slate-700 font-semibold hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Back
          </button>
          {isLastStep ? (
            <button
              type="submit"
              disabled={loading}
              className="px-8 py-3 rounded-xl bg-brand text-white font-semibold hover:bg-brand-600 hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2"
            >
              {loading ? 'Assessing...' : 'Get My Assessment'}
            </button>
          ) : (
            <button
              type="button"
              onClick={handleNext}
              disabled={loading}
              className="px-8 py-3 rounded-xl bg-brand text-white font-semibold hover:bg-brand-600 hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2"
            >
              Continue
            </button>
          )}
        </div>

        {submitError && (
          <div className="mt-6 p-5 rounded-xl bg-red-50 border border-red-200">
            <h3 className="text-sm font-bold text-red-800 mb-2">Assessment could not be completed</h3>
            <p className="text-sm text-red-700">{submitError}</p>
          </div>
        )}
      </form>
    </div>
  );
}