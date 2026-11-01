import { useState, useEffect } from 'react';

// Screen 1: Landing page. This is the first screen users see.
export default function LandingPage({ onStartAssessment, scrollTo }) {
  const [activeStep] = useState(0);

  const steps = [
    {
      title: 'Enter your health information',
      text: 'Age, gender and life stage so the assessment is interpreted in the right context.',
    },
    {
      title: 'Provide physical measurements',
      text: 'Weight, height and MUAC; your BMI is calculated automatically.',
    },
    {
      title: 'Select symptoms, or none',
      text: 'Choose common symptoms, or tell us you currently feel healthy.',
    },
    {
      title: 'Get your results and guidance',
      text: 'A machine-learning supported nutritional classification with personalised recommendations.',
    },
  ];

  const features = [
    {
      title: 'Health & nutrition assessment',
      text: 'A guided form covering personal details, physical measurements, lifestyle and symptoms.',
    },
    {
      title: 'ML-supported classification',
      text: 'A saved scikit-learn model classifies nutritional status from your measurements.',
    },
    {
      title: 'Symptom-based insights',
      text: 'Reported symptoms inform general nutritional considerations, never a diagnosis.',
    },
    {
      title: 'Personalised recommendations',
      text: 'Food groups, meal ideas and hydration guidance aligned with your result.',
    },
  ];

  useEffect(() => {
    if (scrollTo) {
      const id = scrollTo.replace('#', '');
      const el = document.getElementById(id);
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }
  }, [scrollTo]);

  return (
    <div className="min-h-screen flex flex-col text-slate-900">
      <nav className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-wrap items-center justify-between gap-4 py-3">
            <button
              onClick={() => { /* home: already here */ }}
              className="flex items-center gap-2"
            >
              <span className="rounded-xl bg-brand p-1.5">🍏</span>
              <span className="font-display font-bold text-brand tracking-tight text-lg">NutriCheck</span>
            </button>
            <div className="hidden md:flex gap-8 items-center text-sm font-medium">
              {[
                { id: 'home', label: 'Home', section: '#home' },
                { id: 'how-it-works', label: 'How It Works', section: '#how-it-works' },
                { id: 'about', label: 'About', section: '#about' },
              ].map((link) => (
                <button
                  key={link.id}
                  onClick={() => scrollTo(link.section)}
                  className="px-3 py-2 rounded-md text-slate-600 hover:text-brand transition-colors"
                >
                  {link.label}
                </button>
              ))}
            </div>
            <button
              className="md:hidden p-2 text-slate-600 hover:text-brand rounded-lg hover:bg-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand"
              aria-label="Toggle menu"
            >
              <span className="block w-5 h-0.5 bg-current mb-1.5"></span>
              <span className="block w-5 h-0.5 bg-current mb-1.5"></span>
              <span className="block w-5 h-0.5 bg-current"></span>
            </button>
          </div>
        </div>
      </nav>
      {/* Hero */}
      <section id="home" className="relative px-4 py-12 sm:py-20 lg:py-24 overflow-hidden">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-10 items-center">
            <div>
              <h1 className="font-display font-bold text-4xl sm:text-5xl lg:text-6xl text-brand leading-tight mb-6">
                Nutritional health assessment,<br />
                made clear and personal.
              </h1>
              <p className="text-lg text-slate-600 mb-8 max-w-lg leading-relaxed">
                NutriCheck helps you understand your nutritional status through a guided health
                assessment: your details, measurements, lifestyle and symptoms are reviewed with a
                machine-learning supported model, and you receive practical nutritional guidance.
              </p>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={onStartAssessment}
                  className="bg-brand text-white px-7 py-3.5 rounded-xl font-semibold text-base hover:bg-brand-600 hover:shadow-lg transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2"
                >
                  Start Assessment
                </button>
                <button
                  onClick={() => scrollTo('#how-it-works')}
                  className="border border-slate-300 text-slate-700 px-7 py-3.5 rounded-xl font-semibold text-base hover:bg-slate-50 transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                >
                  Learn More
                </button>
              </div>
              <p className="mt-8 text-sm text-slate-500">
                Preliminary educational screening only — not a medical diagnosis. Consult a qualified
                health professional for medical advice.
              </p>
            </div>
            <div className="hidden lg:block">
              <div className="relative rounded-2xl overflow-hidden border border-slate-200 shadow-lg">
                <div className="absolute inset-0 bg-gradient-to-br from-green-50 via-slate-50 to-brand-50 opacity-40" />
                <div className="relative p-8 sm:p-12 grid grid-cols-2 gap-4 text-center">
                  <div className="bg-white p-6 rounded-xl border border-slate-100 shadow-sm">
                    <div className="text-2xl mb-2">📋</div>
                    <h3 className="font-semibold text-slate-900">Personal profile</h3>
                    <p className="text-xs text-slate-500 mt-1">Age · gender · life stage</p>
                  </div>
                  <div className="bg-white p-6 rounded-xl border border-slate-100 shadow-sm">
                    <div className="text-2xl mb-2">📏</div>
                    <h3 className="font-semibold text-slate-900">Measurements</h3>
                    <p className="text-xs text-slate-500 mt-1">Weight · height · MUAC · BMI</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="px-4 py-16 sm:py-20 bg-slate-50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="font-display font-bold text-3xl sm:text-4xl text-slate-900 mb-3">How it works</h2>
            <p className="text-slate-600">Four simple steps to your personalised nutritional picture</p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {steps.map((step, index) => (
              <div key={index} className="flex flex-col items-center text-center p-6 rounded-2xl bg-white border border-slate-200 shadow-sm hover:shadow-md transition-all group">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center mb-4 font-bold text-xl ${
                  activeStep === index ? 'bg-brand text-white' : 'bg-slate-200 text-slate-500'
                }`}>
                  {index + 1}
                </div>
                <h3 className="font-semibold text-slate-900 mb-2">{step.title}</h3>
                <p className="text-sm text-slate-600 leading-relaxed">{step.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
      {/* Features */}
      <section id="features" className="px-4 py-16 sm:py-20">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="font-display font-bold text-3xl sm:text-4xl text-slate-900 mb-3">What you get</h2>
            <p className="text-slate-600">A credible, transparent digital nutrition platform</p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature) => (
              <div key={feature.title} className="p-6 rounded-2xl border border-slate-200 bg-white shadow-sm hover:shadow-md transition-all">
                <h3 className="font-semibold text-slate-900 mb-3">{feature.title}</h3>
                <p className="text-sm text-slate-600 leading-relaxed">{feature.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* About / important information */}
      <section id="about" className="px-4 py-16 sm:py-20 bg-slate-50">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-start">
            <div>
              <h2 className="font-display font-bold text-3xl sm:text-4xl text-slate-900 mb-4">About NutriCheck</h2>
              <p className="text-slate-600 leading-relaxed mb-6">
                NutriCheck is a digital health and nutrition screening tool built with React and a
                scikit-learn machine-learning model. It turns your health information, physical
                measurements and reported symptoms into a clear nutritional classification and
                practical dietary guidance.
              </p>
              <p className="text-slate-600 leading-relaxed mb-6">
                The platform supports adults and provides general educational insight. It does not
                replace a clinician, dietitian or medical test.
              </p>
            </div>
            <div className="rounded-2xl border border-brand-200 bg-brand-50 p-6 shadow-sm">
              <h3 className="font-bold text-brand mb-3 flex items-start gap-2">
                <span>⚠️</span> Important information
              </h3>
              <p className="text-sm text-brand-900/90 leading-relaxed mb-4">
                This application provides <strong>preliminary nutritional guidance only</strong>. It
                is not a replacement for professional medical diagnosis, treatment or dietary advice
                from a qualified healthcare professional.
              </p>
              <ul className="text-sm text-brand-900/80 space-y-2 list-disc pl-5 mb-4">
                <li>ML predictions are screening signals, not confirmed medical conditions.</li>
                <li>BMI categories depend on age and population context.</li>
                <li>Symptoms alone cannot diagnose a nutritional deficiency.</li>
                <li>If you have concerning symptoms, seek professional assessment promptly.</li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}