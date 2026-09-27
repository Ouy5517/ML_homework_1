from backend.app import create_app


class FixedModel:
    def predict(self, frame):
        return [321.4]


def test_predict_endpoint_returns_model_value_and_metadata():
    app = create_app(model=FixedModel())
    client = app.test_client()
    response = client.post(
        "/predict",
        json={
            "date": "2024-01-05",
            "hour": 9,
            "temperature": 12,
            "humidity": 55,
            "rainfall": 0,
            "snowfall": 0,
            "holiday": "No Holiday",
            "functioning_day": "Yes",
        },
    )
    assert response.status_code == 200
    assert response.get_json() == {"prediction": 321}


def test_predict_endpoint_reports_invalid_payload():
    app = create_app(model=FixedModel())
    response = app.test_client().post("/predict", json={"date": "2024-01-05"})
    assert response.status_code == 400
    assert "error" in response.get_json()
