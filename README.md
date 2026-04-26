# Health Check - Nutrition Assessment System

A comprehensive web application for malnutrition detection and nutritional assessment using machine learning and modern web technologies.

 Features

- **Malnutrition Detection**: Advanced ML models for detecting malnutrition levels (Normal, Moderate, Severe)
- **Health Assessment**: Comprehensive health evaluation based on BMI, age, gender, and other factors
- **Meal Planning**: Personalized meal recommendations based on nutritional needs
- **Web Interface**: Modern React-based frontend with responsive design
- **API Endpoints**: RESTful API for integration with other systems
- **Batch Processing**: Support for processing multiple assessments simultaneously

 Architecture

### Backend (Flask)
- **Framework**: Flask with CORS support
- **ML Models**: Scikit-learn based malnutrition detection models
- **Features**: Z-score calculations, feature engineering, confidence scoring
- **Endpoints**:
  - `/api/health` - Health check
  - `/api/assess-health` - Individual health assessment
  - `/api/generate-meal-plan` - Personalized meal planning
  - `/predict` - Single prediction
  - `/predict/batch` - Batch predictions
  - `/model/info` - Model information

 Frontend (React + Vite)
- **Framework**: React 19 with Vite
- **Styling**: Tailwind CSS
- **Features**: Responsive design, modern UI components

 Machine Learning Models
- **Base Model**: Random Forest classifier for malnutrition detection
- **Enhanced Model**: Advanced classifier with confidence scoring
- **Preprocessing**: Feature scaling, z-score calculations, feature engineering
- **Validation**: Input validation and error handling

 Prerequisites

- Python 3.8+
- Node.js 16+
- Git

 Installation

### Backend Setup

1. Navigate to the model directory:
```bash
cd model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the client directory:
```bash
cd client/nutricheck-vite
```

2. Install Node.js dependencies:
```bash
npm install
```

##  Running the Application

### Development Mode

1. Start the backend server:
```bash
cd model
python app.py
```
The backend will run on `http://localhost:5000`

2. Start the frontend development server:
```bash
cd client/nutricheck-vite
npm run dev
```
The frontend will run on `http://localhost:5173`

### Production Build

1. Build the frontend:
```bash
cd client/nutricheck-vite
npm run build
```

2. The Flask app will automatically serve the built React app from the `/` route.

##  API Usage

### Health Assessment

```bash
curl -X POST http://localhost:5000/api/assess-health \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "age": 25,
    "gender": "male",
    "weight": 70,
    "height": 175,
    "activity": "moderate",
    "diet": "balanced",
    "lifeStage": "",
    "symptoms": ["fatigue"]
  }'
```

### Malnutrition Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age_months": 300,
    "weight_kg": 70,
    "height_cm": 175,
    "muac_cm": 25,
    "bmi": 22.9
  }'
```

### Batch Predictions

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "age_months": 300,
      "weight_kg": 70,
      "height_cm": 175,
      "muac_cm": 25,
      "bmi": 22.9
    }
  ]'
```

##  Machine Learning Models

The system uses trained machine learning models for malnutrition detection:

- **Input Features**: Age (months), Weight (kg), Height (cm), MUAC (cm), BMI
- **Output Classes**: Normal, Moderate, Severe malnutrition
- **Model Types**: Random Forest, Enhanced Classifier with confidence scoring
- **Preprocessing**: Feature scaling, z-score calculations, feature engineering

##  Project Structure

```
health-check/
├── client/
│   └── nutricheck-vite/          # React frontend
│       ├── src/
│       │   ├── App.jsx
│       │   ├── main.jsx
│       │   └── assets/
│       ├── package.json
│       └── vite.config.js
├── model/                        # Flask backend
│   ├── app.py                    # Main Flask application
│   ├── enhanced_classifier.py    # ML model classes
│   ├── evaluate_model.py         # Model evaluation scripts
│   ├── requirements.txt          # Python dependencies
│   └── *.joblib                  # Trained ML models
├── tests/                        # Test files
├── .gitattributes               # Git LFS configuration
└── README.md                    # This file
```

##  Configuration

### Model Configuration

Models are automatically loaded on startup. Place trained model files in the `model/` directory:
- `malnutrition_model.joblib` - Base model
- `enhanced_malnutrition_classifier.joblib` - Enhanced model
- `feature_scaler.joblib` - Feature scaler
- `label_encoder.joblib` - Label encoder

### Environment Variables

The application uses default configurations. For production deployment, consider setting:
- `FLASK_ENV=production`
- Database connections (if extending to persistent storage)

##  Testing

Run tests from the root directory:
```bash
python -m pytest tests/
```

##  Model Training

To retrain models:

1. Prepare training data in CSV format
2. Run data preprocessing:
```bash
cd model
python data_preprocessing.ipynb
```

3. Train models:
```bash
python enhanced_classifier.py
```

4. Evaluate models:
```bash
python evaluate_model.py
```

##  Deployment

### Docker Deployment

1. Build Docker image:
```bash
docker build -t health-check .
```

2. Run container:
```bash
docker run -p 5000:5000 health-check
```

### Cloud Deployment

The application can be deployed to:
- Heroku
- AWS Elastic Beanstalk
- Google App Engine
- DigitalOcean App Platform

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Support

For support and questions:
- Create an issue on GitHub
- Contact the development team

##  Version History

- **v1.0.0**: Initial release with basic malnutrition detection
- **v1.1.0**: Added enhanced classifier and confidence scoring
- **v1.2.0**: Integrated React frontend
- **v1.3.0**: Added meal planning and batch processing

---

**Note**: This application is for educational and research purposes. Always consult healthcare professionals for medical advice and nutritional guidance.
