// API client for the NutriCheck Flask backend.
// All assessment requests go through the single authoritative endpoint
// /api/assess-health. Relative URLs work both in Vite dev (proxied to
// http://localhost:5000 via vite.config.js) and in production (Flask serves
// the SPA itself).

export class ApiError extends Error {
  constructor(code, message, status, details) {
    super(message);
    this.name = 'ApiError';
    this.code = code; // 'validation_error' | 'model_unavailable' | ...
    this.status = status;
    this.details = details || null;
  }
}

async function parseJsonOrNull(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function toAssessmentPayload(form) {
  return {
    name: form.name,
    age: form.age,
    gender: form.gender,
    weight: form.weight,
    height: form.height,
    muac: form.muac,
    activity: form.activity,
    diet: form.diet,
    life_stage: form.life_stage,
    symptoms: form.symptoms || [],
  };
}

export async function postAssessment(form) {
  const response = await fetch('/api/assess-health', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(toAssessmentPayload(form)),
  });

  const body = await parseJsonOrNull(response);

  if (!response.ok) {
    throw new ApiError(
      (body && body.error) || 'request_failed',
      (body && body.message) || 'The server returned status ' + response.status,
      response.status,
      body && body.details
    );
  }

  if (!body || body.success !== true) {
    throw new ApiError(
      'invalid_response',
      'The server returned an unexpected response.',
      response.status
    );
  }

  return body;
}

export async function postMealPlan(assessment) {
  const response = await fetch('/api/generate-meal-plan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      nutritional_status_key: assessment.nutritional_status_key,
      nutritional_status: assessment.nutritional_status,
      bmi: assessment.bmi,
      bmi_category: assessment.bmi_category,
      symptoms: assessment.symptoms || [],
      diet_type: assessment.diet_type,
    }),
  });

  const body = await parseJsonOrNull(response);

  if (!response.ok) {
    throw new ApiError(
      (body && body.error) || 'request_failed',
      (body && body.message) || 'The server returned status ' + response.status,
      response.status
    );
  }

  return body || {};
}

export async function getHealth() {
  const response = await fetch('/api/health');
  const body = await parseJsonOrNull(response);
  if (!response.ok) {
    throw new ApiError(
      (body && body.error) || 'request_failed',
      'Backend health check failed',
      response.status
    );
  }
  return body || {};
}

export function extractMessage(error) {
  if (error instanceof ApiError) return error.message;
  if (error && typeof error.message === 'string') return error.message;
  return 'An unexpected error occurred.';
}
