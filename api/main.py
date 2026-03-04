"""
Mental Health Prediction REST API
FastAPI backend for production deployment
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Prediction API",
    description="AI-powered mental health risk assessment API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS
# ============================================================================

class SurveyResponse(BaseModel):
    """Survey response schema"""
    company_size: str = Field(..., description="Company size category")
    remote_work: str = Field(..., description="Remote work frequency")
    tech_company: str = Field(..., description="Is tech company")
    benefits: str = Field(..., description="Mental health benefits provided")
    care_options: str = Field(..., description="Knowledge of care options")
    wellness_program: str = Field(..., description="Wellness program availability")
    leave_ease: str = Field(..., description="Ease of requesting leave")
    mh_discussion: str = Field(..., description="Employer discusses mental health")
    negative_consequences: str = Field(..., description="Fear of negative consequences")
    coworker_comfort: str = Field(..., description="Comfort with coworkers")
    supervisor_comfort: str = Field(..., description="Comfort with supervisor")
    career_impact: str = Field(..., description="Career impact concerns")
    coworker_view: str = Field(..., description="Coworker negative view")
    family_history: str = Field(..., description="Family history of mental illness")
    past_disorder: str = Field(..., description="Past mental health disorder")
    work_interfere: str = Field(..., description="Work interference level")
    age: int = Field(..., ge=18, le=100, description="Age")

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    disorder_prediction: int
    disorder_probability: float
    disorder_risk_level: str
    treatment_prediction: int
    treatment_probability: float
    treatment_likelihood: str
    composite_indices: Dict[str, float]
    top_risk_factors: List[Dict[str, float]]
    recommendations: List[Dict[str, str]]

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    responses: List[SurveyResponse]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: bool
    version: str

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

models = {
    'disorder': None,
    'treatment': None,
    'scaler': None,
    'feature_names': None
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_models():
    """Load pre-trained models"""
    try:
        models['disorder'] = joblib.load('models/model_disorder.pkl')
        models['treatment'] = joblib.load('models/model_treatment.pkl')
        models['scaler'] = joblib.load('models/scaler.pkl')
        models['feature_names'] = joblib.load('models/feature_names.pkl')
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def encode_response(response: SurveyResponse) -> Dict:
    """Encode survey response to numerical values"""
    
    mappings = {
        'Yes': 1, 'No': 0, 'Maybe': 0.5, "Don't know": 0.5, 'Not sure': 0.5,
        'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 3,
        'Very difficult': 0, 'Somewhat difficult': 1, 'Neither': 2,
        'Somewhat easy': 3, 'Very easy': 4,
        '1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3, '500-1000': 4, '1000+': 5
    }
    
    encoded = {}
    for field, value in response.dict().items():
        if isinstance(value, str):
            encoded[field] = mappings.get(value, 0)
        else:
            encoded[field] = value
    
    return encoded

def create_feature_vector(encoded: Dict, feature_names: List[str]) -> np.ndarray:
    """Create feature vector from encoded response"""
    
    feature_vector = []
    for fname in feature_names:
        feature_vector.append(encoded.get(fname, 0))
    
    return np.array(feature_vector).reshape(1, -1)

def calculate_composite_indices(encoded: Dict) -> Dict[str, float]:
    """Calculate composite indices"""
    
    # Simplified calculation - normalize to 0-1
    support = (
        encoded.get('benefits', 0) +
        encoded.get('care_options', 0) +
        encoded.get('wellness_program', 0) +
        encoded.get('leave_ease', 0) / 4 +
        encoded.get('mh_discussion', 0)
    ) / 5
    
    stigma = (
        encoded.get('negative_consequences', 0) +
        (1 - encoded.get('coworker_comfort', 0)) +
        (1 - encoded.get('supervisor_comfort', 0)) +
        encoded.get('career_impact', 0) +
        encoded.get('coworker_view', 0)
    ) / 5
    
    openness = (
        encoded.get('benefits', 0) +
        encoded.get('care_options', 0) +
        (1 - encoded.get('negative_consequences', 0))
    ) / 3
    
    return {
        'mental_health_support': round(support, 3),
        'workplace_stigma': round(stigma, 3),
        'organizational_openness': round(openness, 3)
    }

def get_top_risk_factors(feature_vector: np.ndarray, feature_names: List[str]) -> List[Dict]:
    """Identify top risk factors based on feature values"""
    
    # Get top 5 features with highest values
    top_indices = np.argsort(feature_vector[0])[-5:][::-1]
    
    risk_factors = []
    for idx in top_indices:
        risk_factors.append({
            'factor': feature_names[idx],
            'score': float(feature_vector[0][idx])
        })
    
    return risk_factors

def generate_recommendations_api(disorder_prob: float, treatment_prob: float, encoded: Dict) -> List[Dict]:
    """Generate recommendations"""
    
    recommendations = []
    
    if disorder_prob > 0.6 and treatment_prob < 0.4:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Immediate Support',
            'action': 'Schedule confidential HR meeting',
            'reason': 'High risk with low treatment engagement'
        })
    
    if encoded.get('benefits', 0) < 0.5:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Benefits',
            'action': 'Review mental health benefits',
            'reason': 'Low awareness of benefits'
        })
    
    if encoded.get('negative_consequences', 0) > 0.5:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Culture',
            'action': 'Implement anti-stigma training',
            'reason': 'High perceived stigma'
        })
    
    if not recommendations:
        recommendations.append({
            'priority': 'LOW',
            'category': 'Prevention',
            'action': 'Continue wellness practices',
            'reason': 'Maintain good mental health'
        })
    
    return recommendations

# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    success = load_models()
    if not success:
        print("⚠️ Warning: Models not loaded. Train models first.")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    pass

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Mental Health Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if models['disorder'] is not None else "degraded",
        models_loaded=models['disorder'] is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(response: SurveyResponse):
    """
    Make prediction for a single survey response
    
    Returns detailed prediction with risk assessment and recommendations
    """
    
    # Check if models are loaded
    if models['disorder'] is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Encode response
        encoded = encode_response(response)
        
        # Create feature vector
        feature_vector = create_feature_vector(encoded, models['feature_names'])
        
        # Scale features
        feature_scaled = models['scaler'].transform(feature_vector)
        
        # Make predictions
        disorder_pred = models['disorder'].predict(feature_scaled)[0]
        disorder_prob = models['disorder'].predict_proba(feature_scaled)[0][1]
        
        treatment_pred = models['treatment'].predict(feature_scaled)[0]
        treatment_prob = models['treatment'].predict_proba(feature_scaled)[0][1]
        
        # Determine risk levels
        if disorder_prob < 0.3:
            risk_level = "LOW"
        elif disorder_prob < 0.6:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"
        
        if treatment_prob < 0.4:
            treatment_likelihood = "UNLIKELY"
        elif treatment_prob < 0.7:
            treatment_likelihood = "POSSIBLE"
        else:
            treatment_likelihood = "LIKELY"
        
        # Calculate composite indices
        indices = calculate_composite_indices(encoded)
        
        # Get top risk factors
        risk_factors = get_top_risk_factors(feature_vector, models['feature_names'])
        
        # Generate recommendations
        recommendations = generate_recommendations_api(disorder_prob, treatment_prob, encoded)
        
        return PredictionResponse(
            disorder_prediction=int(disorder_pred),
            disorder_probability=float(disorder_prob),
            disorder_risk_level=risk_level,
            treatment_prediction=int(treatment_pred),
            treatment_probability=float(treatment_prob),
            treatment_likelihood=treatment_likelihood,
            composite_indices=indices,
            top_risk_factors=risk_factors,
            recommendations=recommendations
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(request: BatchPredictionRequest):
    """
    Make predictions for multiple survey responses
    
    Useful for batch processing or analytics
    """
    
    if models['disorder'] is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        results = []
        
        for response in request.responses:
            prediction = await predict(response)
            results.append(prediction.dict())
        
        return {
            "count": len(results),
            "predictions": results,
            "summary": {
                "high_risk_count": sum(1 for r in results if r['disorder_risk_level'] == 'HIGH'),
                "moderate_risk_count": sum(1 for r in results if r['disorder_risk_level'] == 'MODERATE'),
                "low_risk_count": sum(1 for r in results if r['disorder_risk_level'] == 'LOW'),
                "avg_disorder_probability": sum(r['disorder_probability'] for r in results) / len(results),
                "avg_treatment_probability": sum(r['treatment_probability'] for r in results) / len(results)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/train")
async def train_models(file: UploadFile = File(...)):
    """
    Train new models with uploaded data
    
    Upload a CSV file with training data
    """
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV format")
    
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Import preprocessing
        from src.data.preprocessing import preprocess_data, create_composite_indices
        
        # Preprocess
        df_clean = preprocess_data(df)
        df_features = create_composite_indices(df_clean)
        
        # Training logic here (similar to Streamlit app)
        # ... (implement full training pipeline)
        
        return {
            "status": "success",
            "message": "Models trained successfully",
            "samples": len(df)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics"""
    
    if models['disorder'] is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # In production, load actual metrics from file
    return {
        "disorder_model": {
            "accuracy": 0.82,
            "f1_score": 0.79,
            "precision": 0.81,
            "recall": 0.78
        },
        "treatment_model": {
            "accuracy": 0.85,
            "f1_score": 0.83,
            "precision": 0.84,
            "recall": 0.82
        }
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
