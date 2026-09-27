import json
from pathlib import Path

import pandas as pd
import pytest

from backend.model import FEATURE_COLUMNS, build_pipeline, frame_from_payload, prepare_training_frame


def sample_frame():
    return pd.DataFrame(
        [
            {"date": "2024-01-01", "hour": 8, "temperature": 5.0, "humidity": 50, "rainfall": 0.0, "snowfall": 0.0, "holiday": "No Holiday", "functioning_day": "Yes", "rented_bike_count": 120},
            {"date": "2024-01-02", "hour": 17, "temperature": 10.0, "humidity": 60, "rainfall": 0.0, "snowfall": 0.0, "holiday": "No Holiday", "functioning_day": "Yes", "rented_bike_count": 260},
            {"date": "2024-01-03", "hour": 12, "temperature": 20.0, "humidity": 45, "rainfall": 0.0, "snowfall": 0.0, "holiday": "Holiday", "functioning_day": "Yes", "rented_bike_count": 180},
            {"date": "2024-01-04", "hour": 22, "temperature": 0.0, "humidity": 85, "rainfall": 2.0, "snowfall": 0.0, "holiday": "No Holiday", "functioning_day": "No", "rented_bike_count": 0},
        ]
    )


def test_training_frame_derives_calendar_features_and_temperature_square():
    frame = prepare_training_frame(sample_frame())
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    assert frame.loc[0, "weekday"] == "Monday"
    assert frame.loc[0, "month"] == 1
    assert frame.loc[1, "temperature_sq"] == 100


def test_pipeline_fits_with_same_feature_schema_used_by_prediction():
    frame = prepare_training_frame(sample_frame())
    pipeline = build_pipeline()
    pipeline.fit(frame[FEATURE_COLUMNS], frame["rented_bike_count"])
    payload_frame = frame_from_payload({
        "date": "2024-01-05",
        "hour": 9,
        "temperature": 12,
        "humidity": 55,
        "rainfall": 0,
        "snowfall": 0,
        "holiday": "No Holiday",
        "functioning_day": "Yes",
    })
    prediction = pipeline.predict(payload_frame[FEATURE_COLUMNS])
    assert len(prediction) == 1
    assert prediction[0] >= 0


def test_payload_rejects_missing_required_feature():
    with pytest.raises(ValueError, match="temperature"):
        frame_from_payload({"date": "2024-01-05", "hour": 9})
