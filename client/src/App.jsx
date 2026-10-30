import { useState } from 'react';
import { postMealPlan } from './api';
import Navbar from './components/Navbar';
import LandingPage from './components/LandingPage';
import AssessmentForm from './components/AssessmentForm';
import ResultsPage from './components/ResultsPage';
import RecommendationsPage from './components/RecommendationsPage';

// NutriCheck application shell.
// Screens: landing (first) -> assessment -> results -> recommendations.
// The backend Flask API is the single authority for the ML classification;
// the frontend never classifies health status on its own.
export default function App() {
  const [page, setPage] = useState('landing');
  const [section, setSection] = useState(null);
  const [assessment, setAssessment] = useState(null);
  const [mealPlan, setMealPlan] = useState(null);
  const [loading, setLoading] = useState(false);

  const navigate = (target, sectionId) => {
    setPage(target);
    if (sectionId) setSection(sectionId);
  };

  const startAssessment = () => {
    setAssessment(null);
    setMealPlan(null);
    setPage('assessment');
  };

  const handleAssessmentComplete = async (result) => {
    setLoading(true);
    try {
      // Fetch meal recommendations aligned with the assessment outcome.
      // This is optional: if it fails, results still display.
      const plan = await postMealPlan(result);
      setMealPlan(plan);
    } catch {
      // Ignore - meal plan is supplementary.
    }
    setAssessment(result);
    setLoading(false);
    setPage('results');
  };

  const handleRetake = () => {
    setAssessment(null);
    setMealPlan(null);
    setPage('assessment');
  };

  const handleRecommendations = () => {
    setPage('recommendations');
  };

  const handleHome = () => {
    setAssessment(null);
    setMealPlan(null);
    setPage('landing');
    setSection(null);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {page !== 'landing' && (
        <Navbar
          onNavigate={navigate}
          onStartAssessment={startAssessment}
          active={page}
        />
      )}
      {page === 'landing' && (
        <Navbar
          onNavigate={navigate}
          onStartAssessment={startAssessment}
          active="home"
        />
      )}

      {page === 'landing' && (
        <LandingPage
          onStartAssessment={startAssessment}
          scrollTo={section || null}
        />
      )}

      {page === 'assessment' && (
        <AssessmentForm onAssessmentComplete={handleAssessmentComplete} />
      )}

      {page === 'results' && assessment && (
        <ResultsPage
          assessment={assessment}
          onRetake={handleRetake}
          onRecommendations={handleRecommendations}
          onHome={handleHome}
        />
      )}

      {page === 'recommendations' && assessment && (
        <RecommendationsPage
          assessment={assessment}
          mealPlan={mealPlan}
          onRetake={handleRetake}
          onResults={() => setPage('results')}
          onHome={handleHome}
        />
      )}

      {loading && (
        <div className="fixed inset-0 bg-slate-900/60 flex items-center justify-center z-50">
          <div className="flex flex-col items-center gap-3 text-white">
            <div className="w-8 h-8 border-4 border-brand border-t-transparent rounded-full animate-spin" />
            <p className="text-sm font-medium">Processing your assessment…</p>
          </div>
        </div>
      )}
    </div>
  );
}
