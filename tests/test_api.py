import requests

def test_predict_endpoint():
    url = "http://localhost:5000/predict"
    payload = {
        "feature1": 0.5,
        "feature2": 1.5
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
