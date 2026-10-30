// Screen 4: Nutritional recommendations and meal planning.
// General educational guidance aligned with the assessment outcome — no claims
// that any food treats or cures a medical condition.
export default function RecommendationsPage({
  assessment,
  onRetake,
  onResults,
  onHome,
  mealPlan,
}) {
  const key = assessment?.prediction?.nutritional_status_key || 'normal';
  const status = assessment?.nutritional_status || 'Normal';
  const bmi = assessment?.bmi;
  const symptoms = assessment?.symptoms || [];
  const noSymptoms = assessment?.no_symptoms_reported;

  const foodGroups = [
    {
      name: 'Vegetables & fruits',
      why: 'Vitamins, minerals and fibre; aim for variety and colour.',
      examples: ['Spinach', 'Kale', 'Carrots', 'Bananas', 'Apples', 'Berries', 'Broccoli'],
    },
    {
      name: 'Proteins',
      why: 'Supports muscle and tissue health.',
      examples: ['Eggs', 'Fish', 'Chicken', 'Lentils', 'Beans', 'Tofu', 'Greek yogurt'],
    },
    {
      name: 'Whole grains',
      why: 'Sustained energy and fibre.',
      examples: ['Oats', 'Brown rice', 'Quinoa', 'Whole grain bread'],
    },
    {
      name: 'Dairy or fortified alternatives',
      why: 'Calcium and vitamin D for bones.',
      examples: ['Milk', 'Fortified plant milk', 'Cheese', 'Yogurt'],
    },
    {
      name: 'Healthy fats',
      why: 'Supports nutrient absorption.',
      examples: ['Olive oil', 'Nuts', 'Seeds', 'Avocado'],
    },
  ];

  const hydration = [
    'Drink water regularly throughout the day.',
    'Aim for about 6–8 glasses unless a clinician advises otherwise.',
    'Limit sugary drinks and excess caffeine.',
  ];

  const generalGuidance = [
    'Eat regular meals and include a protein source at each main meal.',
    'Choose whole grains, vegetables and fruit daily.',
    'If your result indicates nutritional risk, discuss dietary changes with a clinician or dietitian first.',
  ];

  const statusGuidance = {
    normal: [
      'Continue with your current balanced eating pattern.',
      'Keep routine health check-ups.',
    ],
    moderate: [
      'Focus on nutrient-dense foods: proteins, vegetables, fruits and whole grains.',
      'Consider speaking with a healthcare professional about your screening result.',
    ],
    severe: [
      'Please share this result with a healthcare professional promptly.',
      'Do not make major dietary changes without clinical guidance.',
    ],
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <nav className="mb-8 flex items-center gap-2 text-sm">
        <button
          onClick={onResults}
          className="text-slate-500 hover:text-brand underline underline-offset-4"
        >
          ← Back to Results
        </button>
      </nav>

      <h1 className="font-display font-bold text-3xl text-slate-900 mb-2">
        Nutritional Recommendations
      </h1>
      <p className="text-slate-600 mb-10">
        Meal ideas and eating guidance aligned with your assessment
        ({status}{bmi != null ? ', BMI ' + bmi : ''}).
      </p>

      {/* Meal plan from backend */}
      {mealPlan && (
        <section className="mb-10 rounded-3xl p-8 border border-slate-200 bg-white shadow-sm">
          <h2 className="font-display font-bold text-2xl text-slate-900 mb-6">
            Suggested meals
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            {[
              { label: 'Breakfast', icon: '🌅', items: mealPlan.breakfast || [] },
              { label: 'Lunch', icon: '🍽️', items: mealPlan.lunch || [] },
              { label: 'Dinner', icon: '🌙', items: mealPlan.dinner || [] },
            ].map((meal) => (
              <div
                key={meal.label}
                className="p-5 rounded-xl border border-slate-200 bg-slate-50"
              >
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-2xl">{meal.icon}</span>
                  <span className="font-semibold text-slate-900">{meal.label}</span>
                </div>
                <ul className="text-sm text-slate-700 space-y-2 list-disc pl-5">
                  {meal.items.map((item, i) => (
                    <li key={i}>{item}</li>
                  ))}
                </ul>
              </div>
            ))}
            <div className="p-5 rounded-xl border border-slate-200 bg-slate-50">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-2xl">🥨</span>
                <span className="font-semibold text-slate-900">Snacks</span>
              </div>
              <ul className="text-sm text-slate-700 space-y-2 list-disc pl-5">
                {(mealPlan.snacks || []).map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            </div>
          </div>
          {mealPlan.considerations && (
            <div className="mt-6 p-5 rounded-xl bg-brand-50 border border-brand-200">
              <h3 className="font-bold text-brand-900 mb-2">Things to keep in mind</h3>
              <ul className="text-sm text-brand-900/90 space-y-2 list-disc pl-6">
                {mealPlan.considerations.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </div>
          )}
          {mealPlan.disclaimer && (
            <p className="mt-5 text-xs text-slate-500 italic">{mealPlan.disclaimer}</p>
          )}
        </section>
      )}

      {/* Food groups */}
      <section className="mb-10">
        <h2 className="font-display font-bold text-2xl text-slate-900 mb-6">Recommended food groups</h2>
        <div className="grid md:grid-cols-3 gap-4">
          {foodGroups.map((group) => (
            <div key={group.name} className="p-5 rounded-2xl border border-slate-200 bg-white">
              <h3 className="font-semibold text-slate-900 mb-2">{group.name}</h3>
              <p className="text-sm text-slate-600 mb-3">{group.why}</p>
              <p className="text-xs text-slate-500 italic">
                e.g. {group.examples.join(', ')}.
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Guidance aligned to status */}
      <section className="mb-10 rounded-3xl p-8 border border-slate-200 bg-white">
        <h2 className="font-display font-bold text-2xl text-slate-900 mb-4">Personalised guidance</h2>
        <ul className="text-sm text-slate-700 space-y-3 list-disc pl-6">
          {[...statusGuidance[key], ...generalGuidance].map((g, i) => (
            <li key={i}>{g}</li>
          ))}
        </ul>
      </section>

      {/* Symptom-based considerations */}
      {!noSymptoms && symptoms.length > 0 && (
        <section className="mb-10 rounded-3xl p-8 border border-amber-200 bg-amber-50">
          <h2 className="font-display font-bold text-2xl text-amber-900 mb-4">
            Based on your symptoms
          </h2>
          <p className="text-sm text-amber-800 mb-4">
            These are general nutritional pointers only — not a diagnosis of any deficiency or
            condition.
          </p>
          <ul className="text-sm text-amber-800 space-y-2 list-disc pl-6">
            {symptoms.map((s) => (
              <li key={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</li>
            ))}
          </ul>
        </section>
      )}

      {/* Hydration */}
      <section className="mb-10 flex items-start gap-3 rounded-2xl p-5 border border-slate-200 bg-slate-50">
        <span className="text-3xl mt-0.5">💧</span>
        <div>
          <h3 className="font-semibold text-slate-900 mb-1">Hydration</h3>
          <ul className="text-sm text-slate-700 space-y-1 list-disc pl-5">
            {hydration.map((h, i) => (
              <li key={i}>{h}</li>
            ))}
          </ul>
        </div>
      </section>

      {/* Disclaimer */}
      <div className="mt-10 p-5 rounded-xl bg-amber-50 border border-amber-200">
        <h3 className="text-sm font-bold text-amber-900 mb-2">Important</h3>
        <p className="text-sm text-amber-800 leading-relaxed">
          These meal suggestions are general educational guidance. They are not a medical
          prescription and do not claim to treat or cure any condition. Please consult a qualified
          healthcare professional or dietitian for personalised advice.
        </p>
      </div>

      {/* Actions */}
      <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center">
        <button
          onClick={onRetake}
          className="flex-1 px-8 py-3.5 rounded-xl border border-slate-300 text-slate-700 font-semibold hover:bg-slate-50 transition-colors"
        >
          Retake Assessment
        </button>
        <button
          onClick={onHome}
          className="flex-1 px-8 py-3.5 rounded-xl bg-brand text-white font-semibold hover:bg-brand-600 transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2"
        >
          Return Home
        </button>
      </div>
    </div>
  );
}