"""
API Testing Script
Test the FastAPI endpoints
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*50)
    print("Testing Health Endpoint")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
def test_predict():
    """Test prediction endpoint"""
    print("\n" + "="*50)
    print("Testing Prediction Endpoint")
    print("="*50)
    
    # Sample request
    data = {
        "company_size": "100-500",
        "remote_work": "Sometimes",
        "tech_company": "Yes",
        "benefits": "Yes",
        "care_options": "Yes",
        "wellness_program": "No",
        "leave_ease": "Somewhat easy",
        "mh_discussion": "Yes",
        "negative_consequences": "No",
        "coworker_comfort": "Yes",
        "supervisor_comfort": "Yes",
        "career_impact": "No",
        "coworker_view": "No",
        "family_history": "Yes",
        "past_disorder": "No",
        "work_interfere": "Sometimes",
        "age": 32
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n🎯 Prediction Results:")
        print(f"   Disorder Risk: {result['disorder_risk_level']} ({result['disorder_probability']:.1%})")
        print(f"   Treatment Likelihood: {result['treatment_likelihood']} ({result['treatment_probability']:.1%})")
        print(f"\n📊 Composite Indices:")
        for key, value in result['composite_indices'].items():
            print(f"   {key}: {value:.3f}")
        print(f"\n💡 Recommendations: {len(result['recommendations'])} items")
        for rec in result['recommendations']:
            print(f"   [{rec['priority']}] {rec['category']}: {rec['action']}")
    else:
        print(f"Error: {response.text}")

def test_metrics():
    """Test metrics endpoint"""
    print("\n" + "="*50)
    print("Testing Metrics Endpoint")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("\n🧪 Starting API Tests...")
    print(f"API URL: {BASE_URL}")
    
    try:
        test_health()
        test_predict()
        test_metrics()
        
        print("\n" + "="*50)
        print("✅ All tests completed!")
        print("="*50)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure API is running: uvicorn api:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")
