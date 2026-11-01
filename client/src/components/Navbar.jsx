import { useState } from 'react';

// NutriCheck navigation bar. Used on every screen.
// Links: Home, How It Works, About (sections on the landing page),
// Start Assessment (navigates to the assessment flow).
export default function Navbar({ onNavigate, onStartAssessment, active = 'home' }) {
  const [menuOpen, setMenuOpen] = useState(false);

  const navLinks = [
    { id: 'home', label: 'Home', section: '#home' },
    { id: 'how-it-works', label: 'How It Works', section: '#how-it-works' },
    { id: 'about', label: 'About', section: '#about' },
  ];

  const handleLink = (section) => {
    setMenuOpen(false);
    if (section === '#home') {
      onNavigate('landing');
    } else {
      onNavigate('landing', section);
    }
  };

  return (
    <nav className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur-md backdrop-saturate-150">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-wrap items-center justify-between gap-4 py-3">
          <button
            onClick={() => {
              setMenuOpen(false);
              onNavigate('landing');
            }}
            className="flex items-center gap-2"
          >
            <span className="rounded-xl bg-brand p-1.5">🍏</span>
            <span className="font-display font-bold text-brand tracking-tight text-lg">NutriCheck</span>
          </button>

          <div className="hidden md:flex gap-8 items-center text-sm font-medium">
            {navLinks.map((link) => (
              <button
                key={link.id}
                onClick={() => handleLink(link.section)}
                className={`px-3 py-2 rounded-md text-slate-600 hover:text-brand transition-colors ${
                  active === link.id ? 'text-brand font-semibold' : ''
                }`}
              >
                {link.label}
              </button>
            ))}
            {active !== 'home' && (
              <button
                onClick={onStartAssessment}
                className="bg-brand text-white px-5 py-2.5 rounded-xl font-semibold text-sm hover:bg-brand-600 hover:shadow-md transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2"
              >
                Start Assessment
              </button>
            )}
          </div>

          {/* Mobile menu toggle */}
          <button
            onClick={() => setMenuOpen((open) => !open)}
            className="md:hidden p-2 text-slate-600 hover:text-brand rounded-lg hover:bg-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand"
            aria-label="Toggle menu"
            aria-expanded={menuOpen}
          >
            <span className="block w-5 h-0.5 bg-current mb-1.5"></span>
            <span className="block w-5 h-0.5 bg-current mb-1.5"></span>
            <span className="block w-5 h-0.5 bg-current"></span>
          </button>
        </div>

        {/* Mobile menu panel */}
        {menuOpen && (
          <div className="md:hidden border-t border-slate-100 bg-white py-4 space-y-2">
            {navLinks.map((link) => (
              <button
                key={link.id}
                onClick={() => handleLink(link.section)}
                className="block w-full text-left px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50 hover:text-brand rounded-lg transition-colors"
              >
                {link.label}
              </button>
            ))}
            {active !== 'home' && (
              <button
                onClick={() => { setMenuOpen(false); onStartAssessment(); }}
                className="w-full px-4 py-3 text-sm font-medium text-white bg-brand rounded-lg hover:bg-brand-600 transition-colors"
              >
                Start Assessment
              </button>
            )}
          </div>
        )}
      </div>
    </nav>
  );
}
