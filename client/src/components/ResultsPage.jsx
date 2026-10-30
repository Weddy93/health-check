// Screen 3: Assessment results.
// The ML prediction is presented as a preliminary screening signal — never a
// confirmed medical diagnosis. No fabricated confidence scores are shown.
export default function ResultsPage({ assessment, onRetake, onRecommendations, onHome }) {
  const key = assessment.prediction?.nutritional_status_key || 'normal';
  const bmi = assessment.bmi;
  const bmiCategory = assessment.bmi_category || 'not-calculated';
  const symptoms = assessment.symptoms || [];
  const noSymptoms = assessment.no_symptoms_reported;
  const recommendations = assessment.recommendations || [];

  const statusConfig = {
    normal: {
      label: 'Normal nutritional status',
      tone: 'text-brand-900',
      bg: 'bg-brand-50',
      border: 'border-brand-200',
      icon: '✅',
    },
    moderate: {
      label: 'Moderate malnutrition (screening)',
      tone: 'text-amber-900',
      bg: 'bg-amber-50',
      border: 'border-amber-200',
      icon: '⚠️',
    },
    severe: {
      label: 'Severe malnutrition (screening)',
      tone: 'text-red-900',
      bg: 'bg-red-50',
      border: 'border-red-200',
      icon: '🚨',
    },
  };

  const cfg = statusConfig[key] || statusConfig.normal;

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      <nav className="mb-8 flex items-center gap-2 text-sm">
        <button
          onClick={onHome}
          className="text-slate-500 hover:text-brand underline underline-offset-4"
        >
          ← Return Home
        </button>
      </nav>

      <h1 className="font-display font-bold text-3xl text-slate-900 mb-2">Assessment Results</h1>
      <p className="text-slate-600 mb-10">
        Here is your personalised nutritional screening summary.
      </p>

      {/* Result card */}
      <section className={`rounded-3xl p-8 mb-10 shadow-sm border ${cfg.bg} ${cfg.border}`}>
        <div className="flex flex-col md:flex-row md:items-center gap-6 mb-8">
          <div className="flex flex-col items-center text-center">
            <span className="text-6xl mb-3">{cfg.icon}</span>
            <h2 className={`font-bold text-2xl ${cfg.tone}`}>{cfg.label}</h2>
          </div>
          <div className="flex-1 grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div className="bg-white p-4 rounded-xl border border-slate-200">
              <p className="text-xs text-slate-500 uppercase font-semibold">BMI</p>
              <p className="text-2xl font-bold text-slate-900">
                {bmi == null ? '—' : bmi}
              </p>
            </div>
            <div className="bg-white p-4 rounded-xl border border-slate-200">
              <p className="text-xs text-slate-500 uppercase font-semibold">BMI category</p>
              <p className="text-sm font-semibold text-slate-700 capitalize">
                {bmiCategory}
              </p>
            </div>
            <div className="bg-white p-4 rounded-xl border border-slate-200">
              <p className="text-xs text-slate-500 uppercase font-semibold">Model</p>
              <p className="text-sm font-semibold text-slate-700 truncate">
                {assessment.prediction?.model_type || 'ML classifier'}
              </p>
            </div>
            <div className="bg-white p-4 rounded-xl border border-slate-200">
              <p className="text-xs text-slate-500 uppercase font-semibold">Symptoms</p>
              <p className="text-sm font-semibold text-slate-700">
                {noSymptoms ? 'None reported' : symptoms.length ? symptoms.join(', ') : '—'}
              </p>
            </div>
          </div>
        </div>

        {/* What this means */}
        <div className="bg-white p-6 rounded-xl border border-slate-200">
          <h3 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
            <span>ℹ️</span> What this result means
          </h3>
          <ul className="text-sm text-slate-600 space-y-3 list-disc pl-6">
            <li>
              This is a <strong>preliminary screening signal</strong> from a machine-learning model,
              not a confirmed medical diagnosis.
            </li>
            <li>
              Your BMI category ({bmiCategory}) is shown separately from the ML nutritional
              classification — they measure different things.
            </li>
            {noSymptoms ? (
              <li>
                You reported <strong>no symptoms</strong>. This is recorded as an empty symptom
                list; no symptom was treated as a medical finding.
              </li>
            ) : (
              <li>
                Reported symptoms: <strong>{symptoms.join(', ')}</strong>. These inform general
                nutritional considerations only.
              </li>
            )}
            <li>
              {key === 'severe' || key === 'moderate'
                ? 'We recommend sharing this result with a healthcare professional for further assessment.'
                : 'Continue regular check-ups and a balanced diet.'}
            </li>
          </ul>
        </div>

        {recommendations.length > 0 && (
          <div className="mt-6">
            <h3 className="font-semibold text-slate-900 mb-3">Recommended next steps</h3>
            <ul className="text-sm text-slate-600 space-y-2 list-disc pl-6">
              {recommendations.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Medical safety disclaimer */}
        <div className="mt-6 p-5 rounded-xl bg-amber-50 border border-amber-200">
          <h3 className="text-sm font-bold text-amber-900 mb-2 flex items-center gap-2">
            <span>⚠️</span> Not a medical device
          </h3>
          <p className="text-sm text-amber-800 leading-relaxed">
            NutriCheck is an educational preliminary screening tool. It does not replace diagnosis or
            treatment by a qualified clinician or dietitian. If you have concerning symptoms or an
            unexpected result, please seek professional medical assessment.
          </p>
        </div>
      </section>

      {/* Actions */}
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <button
          onClick={onRetake}
          className="flex-1 px-8 py-3.5 rounded-xl border border-slate-300 text-slate-700 font-semibold hover:bg-slate-50 transition-colors"
        >
          Retake Assessment
        </button>
        <button
          onClick={onRecommendations}
          className="flex-1 px-8 py-3.5 rounded-xl bg-brand text-white font-semibold hover:bg-brand-600 hover:shadow-lg transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2"
        >
          View Nutritional Recommendations
        </button>
      </div>
    </div>
  );
}