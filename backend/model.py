"""Reusable data preparation and model training utilities for RideCast."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "rented_bike_count"
FEATURE_COLUMNS = [
    "hour",
    "weekday",
    "month",
    "temperature",
    "temperature_sq",
    "humidity",
    "rainfall",
    "snowfall",
    "holiday",
    "functioning_day",
]
NUMERIC_COLUMNS = ["hour", "month", "temperature", "temperature_sq", "humidity", "rainfall", "snowfall"]
CATEGORICAL_COLUMNS = ["weekday", "holiday", "functioning_day"]
REQUIRED_PAYLOAD_FIELDS = [
    "date",
    "hour",
    "temperature",
    "humidity",
    "rainfall",
    "snowfall",
    "holiday",
    "functioning_day",
]

COLUMN_ALIASES = {
    "Date": "date",
    "Hour": "hour",
    "Rented Bike Count": TARGET_COLUMN,
    "Temperature(°C)": "temperature",
    "Humidity(%)": "humidity",
    "Rainfall(mm)": "rainfall",
    "Snowfall (cm)": "snowfall",
    "Holiday": "holiday",
    "Functioning Day": "functioning_day",
}


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename the published UCI columns and accept already-canonical columns."""
    renamed = frame.rename(columns=COLUMN_ALIASES).copy()
    required = {"date", "hour", "temperature", "humidity", "rainfall", "snowfall", "holiday", "functioning_day"}
    missing = sorted(required - set(renamed.columns))
    if missing:
        raise ValueError(f"数据缺少字段: {', '.join(missing)}")
    return renamed


def prepare_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean source rows and create exactly the feature schema used by the API."""
    frame = _canonical_columns(raw)
    frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, errors="coerce")
    for column in ["hour", "temperature", "humidity", "rainfall", "snowfall", TARGET_COLUMN]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["weekday"] = frame["date"].dt.day_name()
    frame["month"] = frame["date"].dt.month
    frame["temperature_sq"] = frame["temperature"] ** 2
    frame = frame.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].clip(lower=0)
    return frame


def frame_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Validate one web request and return the model's feature frame."""
    missing = [field for field in REQUIRED_PAYLOAD_FIELDS if field not in payload or payload[field] in (None, "")]
    if missing:
        raise ValueError(f"缺少字段: {', '.join(missing)}")
    date = pd.to_datetime(payload["date"], errors="coerce")
    if pd.isna(date):
        raise ValueError("date 必须是有效日期")
    try:
        numeric = {key: float(payload[key]) for key in ["hour", "temperature", "humidity", "rainfall", "snowfall"]}
    except (TypeError, ValueError) as exc:
        raise ValueError("hour、temperature、humidity、rainfall、snowfall 必须是数字") from exc
    ranges = {
        "hour": (0, 23),
        "temperature": (-25, 45),
        "humidity": (0, 100),
        "rainfall": (0, 50),
        "snowfall": (0, 10),
    }
    for key, (lower, upper) in ranges.items():
        if not lower <= numeric[key] <= upper:
            raise ValueError(f"{key} 应在 {lower} 到 {upper} 之间")
    if numeric["hour"] != int(numeric["hour"]):
        raise ValueError("hour 必须是整数")
    row = {
        "hour": int(numeric["hour"]),
        "weekday": date.day_name(),
        "month": int(date.month),
        "temperature": numeric["temperature"],
        "temperature_sq": numeric["temperature"] ** 2,
        "humidity": numeric["humidity"],
        "rainfall": numeric["rainfall"],
        "snowfall": numeric["snowfall"],
        "holiday": str(payload["holiday"]),
        "functioning_day": str(payload["functioning_day"]),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def build_pipeline(include_temperature_square: bool = True) -> Pipeline:
    """Build the same preprocessing + regression pipeline for train and inference."""
    numeric = NUMERIC_COLUMNS if include_temperature_square else [column for column in NUMERIC_COLUMNS if column != "temperature_sq"]
    features = numeric + CATEGORICAL_COLUMNS
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=10.0)),
        ]
    )


def evaluate_predictions(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def extract_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    names = preprocessor.get_feature_names_out()
    return pd.DataFrame({"feature": names, "coefficient": regressor.coef_}).sort_values("coefficient", key=np.abs, ascending=False)


def train_and_evaluate(frame: pd.DataFrame) -> dict[str, Any]:
    """Use a chronological 80/20 split and return fitted model, metrics and predictions."""
    if len(frame) < 10:
        raise ValueError("训练数据至少需要 10 条记录")
    split_at = int(len(frame) * 0.8)
    train = frame.iloc[:split_at]
    test = frame.iloc[split_at:]
    base_features = [column for column in FEATURE_COLUMNS if column != "temperature_sq"]
    baseline = build_pipeline(include_temperature_square=False)
    baseline.fit(train[base_features], train[TARGET_COLUMN])
    baseline_predicted = np.maximum(0, baseline.predict(test[base_features]))
    baseline_metrics = evaluate_predictions(test[TARGET_COLUMN], baseline_predicted)
    # Fit the selected model with the temperature square term requested in the assignment.
    pipeline = build_pipeline(include_temperature_square=True)
    pipeline.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    predicted = np.maximum(0, pipeline.predict(test[FEATURE_COLUMNS]))
    metrics = evaluate_predictions(test[TARGET_COLUMN], predicted)
    predictions = test[["date", "hour", TARGET_COLUMN]].copy()
    predictions["prediction"] = predicted
    return {"pipeline": pipeline, "metrics": metrics, "baseline_metrics": baseline_metrics, "predictions": predictions, "coefficients": extract_coefficients(pipeline), "split_at": split_at}


def save_artifacts(result: dict[str, Any], model_path: Path, report_dir: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["pipeline"], model_path)
    (report_dir / "model_metrics.json").write_text(
        pd.Series({**result["metrics"], "baseline_mae": result["baseline_metrics"]["mae"], "baseline_r2": result["baseline_metrics"]["r2"], "test_rows": len(result["predictions"]), "split_at": result["split_at"]}).to_json(indent=2),
        encoding="utf-8",
    )
    result["coefficients"].to_csv(report_dir / "coefficients.csv", index=False)
    result["predictions"].to_csv(report_dir / "test_predictions.csv", index=False)
    predictions = result["predictions"].copy()
    predictions["absolute_error"] = (predictions[TARGET_COLUMN] - predictions["prediction"]).abs()
    largest_error = predictions.sort_values("absolute_error", ascending=False).head(1)
    largest_error["date"] = largest_error["date"].astype(str)
    (report_dir / "largest_error.json").write_text(
        largest_error.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_pipeline(path: Path):
    return joblib.load(path)
