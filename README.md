# NutriCheck - Nutritional Health Assessment

NutriCheck is a full-stack web application for preliminary nutritional health screening.
It combines a guided health-assessment interface (React + Vite + Tailwind CSS) with a
Flask backend that runs a saved scikit-learn model to classify nutritional status from
physical measurements. The result is presented as a **preliminary screening signal**, never
a medical diagnosis, and is accompanied by general nutritional recommendations and meal
ideas.

---

## 1. Project overview

- **Frontend:** React 19, Vite 7, Tailwind CSS 4, JavaScript (no extra runtime libraries).
- **Backend:** Python 3, Flask 2.3, Flask-CORS, pandas, NumPy, scikit-learn 1.9, joblib.
- **Machine learning:** a saved `RandomForestClassifier` (`malnutrition_model.joblib`) plus a
  `StandardScaler` and a `LabelEncoder`. An optional enhanced wrapper
  (`enhanced_malnutrition_classifier.joblib`) exists but the base model remains the
  authoritative estimator.
- **Assessment workflow (single authoritative path):**

  React validates the form
        |
  React POSTs to /api/assess-health
        |
  Flask validates input (ml_pipeline.validate_assessment_input)
        |
  Flask derives features consistently (ml_pipeline.build_feature_frame)
        |
  Flask loads the saved model + scaler + label encoder
        |
  Flask predicts and returns structured JSON
        |
  React displays the result

  The frontend never performs its own ML classification.

---

## 2. Problem statement

Malnutrition screening is often fragmented: measurements, symptoms and lifestyle factors are
collected inconsistently, and tools frequently conflate BMI categories with malnutrition
diagnoses (e.g. treating overweight/obesity as "malnutrition"). NutriCheck addresses this by
separating BMI category, the ML-predicted nutritional classification and symptom-based
insights, and by presenting the ML prediction as a preliminary screening signal with an
explicit medical disclaimer.

---

## 3. Main objectives

1. Provide a modern, accessible, responsive healthcare-style UI.
2. Introduce a landing page as the first screen.
3. Implement a guided, validated health-assessment workflow.
4. Support a "No symptoms - I currently feel healthy" option.
5. Establish a single authoritative frontend/backend assessment workflow.
6. Correct preprocessing and feature alignment with the saved model artifacts.
7. Audit the saved models and document their limitations honestly.
8. Provide a reproducible ML evaluation script (no fabricated metrics).
9. Improve validation, error handling and security.
10. Update documentation to match the actual code.

---

## 4. Features

- Landing page (hero, how-it-works, features, about, medical disclaimer).
- Guided four-step health assessment (personal, measurements, lifestyle, symptoms).
- BMI calculation (client estimate for preview; authoritative server-side calculation).
- "No symptoms" option with mutual exclusion against real symptoms.
- Single authoritative ML prediction via `RandomForestClassifier`.
- Dedicated results screen (BMI, BMI category, ML classification, symptoms, next steps).
- Nutritional recommendations and meal planning aligned with the result.
- Hydration advice and symptom-based nutritional considerations.
- Graceful handling of missing/invalid data and missing model files.
- Responsive navigation, keyboard focus indicators, accessible colour contrast.

---

## 5. Technology stack

| Layer | Technologies |
|---|---|
| Frontend | React 19, Vite 7, Tailwind CSS 4, JavaScript |
| Backend | Python 3, Flask 2.3, Flask-CORS |
| Machine learning | scikit-learn 1.9, pandas, NumPy, joblib |
| Tooling | ESLint 9, Vite build |

---

## 6. Project architecture

    health-check/
    ├── client/                          # React frontend (Vite)
    │   ├── index.html
    │   ├── package.json
    │   ├── vite.config.js               # dev proxy /api -> http://localhost:5000
    │   └── src/
    │       ├── main.jsx
    │       ├── index.css                 # Tailwind 4 @theme brand palette
    │       ├── App.jsx                    # screen router
    │       ├── api.js                    # API client (fetch)
    │       ├── validation.js             # form validation + symptom options
    │       └── components/
    │           ├── Navbar.jsx
    │           ├── LandingPage.jsx
    │           ├── AssessmentForm.jsx
    │           ├── ResultsPage.jsx
    │           └── RecommendationsPage.jsx
    ├── model/                           # Flask backend
    │   ├── app.py                        # API endpoints + SPA serving
    │   ├── ml_pipeline.py                # single ML preprocessing/prediction pipeline
    │   ├── enhanced_classifier.py        # optional enhanced wrapper class
    │   ├── evaluate_model.py             # reproducible evaluation script
    │   ├── requirements.txt
    │   ├── malnutrition_model.joblib      # base RandomForest model
    │   ├── enhanced_malnutrition_classifier.joblib
    │   ├── feature_scaler.joblib         # StandardScaler
    │   ├── label_encoder.joblib          # LabelEncoder (classes: moderate/normal/severe)
    │   ├── evaluation_report.json        # generated evaluation report
    │   └── *.ipynb                      # original notebooks (reference)
    ├── .gitignore
    └── README.md

---

## 7. Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ (tested with Node 24)
- Git

### Backend setup

    cd model
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate

    pip install -r requirements.txt

### Frontend setup

    cd client
    npm install

---

## 8. Running the application

### Development mode

1. Start the backend (port 5000):

       cd model
       python app.py

2. Start the frontend dev server (port 5173):

       cd client
       npm run dev

   Open `http://localhost:5173`. The Vite dev proxy forwards `/api/*` to Flask.

### Production build

    cd client
    npm run build
    cd ../model
    python app.py

Flask serves the built SPA from `client/dist` at `/`.

---

## 9. Model files and their roles

| File | Role |
|---|---|
| `malnutrition_model.joblib` | Base `RandomForestClassifier` (authoritative estimator) |
| `enhanced_malnutrition_classifier.joblib` | Optional wrapper around the same RF; adds an uncalibrated heuristic confidence score (not exposed to users) |
| `feature_scaler.joblib` | `StandardScaler` fitted on `['age_months', 'weight_kg', 'height_cm', 'muac_cm', 'bmi']` |
| `label_encoder.joblib` | `LabelEncoder` with classes `['moderate', 'normal', 'severe']` (lowercase) |

Feature contract (enforced by `ml_pipeline.py`):

| Feature | Meaning | Units |
|---|---|---|
| `age_months` | age x 12 | months |
| `weight_kg` | body weight | kg |
| `height_cm` | body height | cm |
| `muac_cm` | mid-upper arm circumference | cm |
| `bmi` | derived server-side from weight and height | kg/m^2 |

BMI is **derived on the backend** from weight and height - it is never accepted as user input.
MUAC is a required input (the saved model needs all five features); it is never fabricated.

---

## 10. API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check; reports whether models are loaded |
| POST | `/api/assess-health` | Single authoritative assessment (see below) |
| POST | `/api/generate-meal-plan` | Meal suggestions aligned with an assessment |
| GET | `/model/info` | Model metadata (no confidence scores) |

### `POST /api/assess-health`

Request (JSON):

    {
      "name": "Ana",
      "age": 25,
      "gender": "female",
      "weight": 60,
      "height": 160,
      "muac": 25,
      "activity": "moderate",
      "diet": "balanced",
      "life_stage": "none",
      "symptoms": ["fatigue", "weakness"]
    }

Use `"symptoms": ["no-symptoms"]` to indicate no symptoms (the backend records an empty
symptom list and sets `no_symptoms_reported: true`).

Success response (200): `{ "success": true, "nutritional_status": "Normal", "bmi": 23.4,
"bmi_category": "normal", "symptoms": [...], "symptoms_reported": bool,
"no_symptoms_reported": bool, "recommendations": [...], "symptom_insights": [...],
"model_information": {...}, "profile_summary": {...}, "timestamp": "..." }`.

Error response (4xx/5xx): `{ "success": false, "error": "...", "message": "...",
"details": [...] }`. Status codes: `400` (invalid/missing input), `503` (model unavailable),
`500` (server error).

---

## 11. Model evaluation results

The saved model was trained with scikit-learn 1.7.2; this environment runs 1.9.0, which
raises an `InconsistentVersionWarning`. Predictions still run, but their reliability should
be verified against a real labeled dataset before any clinical use.

A reproducible evaluation script is provided:

    cd model
    python evaluate_model.py

It verifies that all saved artifacts load; enforces the production feature contract
(`age_months`, `weight_kg`, `height_cm`, `muac_cm`, `bmi`); checks deterministic
predictions and correct label decoding; performs a genuine stratified train/test evaluation
**only if** `preprocessed_malnutrition_data.csv` is present.

**The labeled dataset `preprocessed_malnutrition_data.csv` is not present in this
repository.** Therefore no genuine accuracy, precision, recall, F1 or confusion-matrix metrics
are reported, and none have been invented. The script writes `model/evaluation_report.json`.

Structural spot checks show the saved model returns `normal` for varied inputs (child
underweight, adult normal, adult obese) with probability 1.0. Without labeled data, the
model's discriminative ability cannot be confirmed - this is documented as a limitation, not
a verified result.

---

## 12. Known limitations

- The labeled training dataset is not available in the repository, so the model's accuracy
  cannot be genuinely evaluated here.
- The saved model was trained with scikit-learn 1.7.2 but runs on 1.9.0 (version mismatch
  warning; predictions may differ).
- The model appears to predict `normal` across varied inputs; discriminative ability is
  unverified without real labeled data.
- The enhanced classifier's confidence score is heuristic and not statistically calibrated;
  it is deliberately not shown to users.
- MUAC is required by the model and must be measured; it is never fabricated.
- BMI categories are adult screening references; age- and sex-specific child growth
  standards (WHO) are not implemented.
- Symptom-based insights are general educational pointers, never diagnoses.
- No authentication, database or persistent storage is included (out of scope).

---

## 13. Medical disclaimer

NutriCheck is an **educational preliminary screening tool**. It is not a medical device and
does not replace diagnosis, treatment or personalised dietary advice from a qualified
healthcare professional or dietitian. Machine-learning predictions are screening signals, not
confirmed medical conditions. BMI interpretation depends on age and population context. If
you have concerning symptoms or an unexpected result, seek professional medical assessment
promptly.

---

## 14. Future improvements

- Obtain and evaluate against a real, labeled malnutrition dataset.
- Retrain and version-pin scikit-learn to remove the version-mismatch warning.
- Implement age- and sex-specific child growth assessment (WHO standards).
- Add calibrated confidence scores only with sufficient validation data.
- Persist assessment history (optional, with appropriate consent and privacy controls).
- Add automated frontend and backend tests (unit + integration).
- Internationalisation and additional accessibility audits.

---

## 15. Testing

### Frontend

    cd client
    npm install
    npm run lint
    npm run build

Both `npm run lint` and `npm run build` pass on the current codebase.

### Backend

    cd model
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python app.py

Smoke-tested via the Flask test client against `/api/health`, `/api/assess-health`
(valid, no-symptoms, invalid), `/api/generate-meal-plan` and `/model/info`. A live run of
`python app.py` serves the built SPA at `/`.

### Machine learning

    cd model
    python evaluate_model.py

Performs structural validation (artifact loading, feature contract, determinism, label
decoding) and a genuine stratified evaluation only when the labeled dataset is present.

---

## 16. Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make changes and add tests.
4. Submit a pull request.

---

## 17. License

MIT License.

---

## 18. Version history

- **v1.0.0**: Initial release with basic malnutrition detection.
- **v1.1.0**: Added enhanced classifier and confidence scoring.
- **v1.2.0**: Integrated React frontend.
- **v1.3.0**: Added meal planning and batch processing.
- **v1.4.0**: Redesigned UI, single authoritative assessment workflow, corrected ML
  preprocessing, honest model evaluation, improved validation and security.

