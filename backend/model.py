"""Reusable data preparation and model training utilities for RideCast."""

from __future__ import annotations

import json
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
CANDIDATE_ALPHAS = (0.1, 1.0, 3.0, 10.0, 30.0)

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


def build_pipeline(include_temperature_square: bool = True, alpha: float = 10.0) -> Pipeline:
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
            ("regressor", Ridge(alpha=alpha)),
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
    """Select a model on the middle time block and evaluate once on the final block."""
    frame = frame.sort_values(["date", "hour"]).reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("训练数据至少需要 10 条记录")
    train_end = int(len(frame) * 0.64)
    validation_end = int(len(frame) * 0.80)
    if train_end < 2 or validation_end <= train_end or validation_end >= len(frame):
        raise ValueError("训练、验证和测试数据均需要至少包含有效记录")

    train = frame.iloc[:train_end]
    validation = frame.iloc[train_end:validation_end]
    train_validation = frame.iloc[:validation_end]
    test = frame.iloc[validation_end:]

    candidates = []
    for include_temperature_square in (False, True):
        features = [column for column in FEATURE_COLUMNS if include_temperature_square or column != "temperature_sq"]
        model_name = "ridge_polynomial" if include_temperature_square else "ridge_linear"
        for alpha in CANDIDATE_ALPHAS:
            candidate = build_pipeline(include_temperature_square=include_temperature_square, alpha=alpha)
            candidate.fit(train[features], train[TARGET_COLUMN])
            validation_predicted = np.maximum(0, candidate.predict(validation[features]))
            validation_metrics = evaluate_predictions(validation[TARGET_COLUMN], validation_predicted)
            candidates.append(
                {
                    "model": model_name,
                    "include_temperature_square": include_temperature_square,
                    "alpha": alpha,
                    **validation_metrics,
                }
            )

    selected = min(candidates, key=lambda candidate: (candidate["mae"], -candidate["r2"]))
    selected_features = [
        column for column in FEATURE_COLUMNS if selected["include_temperature_square"] or column != "temperature_sq"
    ]
    pipeline = build_pipeline(
        include_temperature_square=selected["include_temperature_square"],
        alpha=selected["alpha"],
    )
    pipeline.fit(train_validation[selected_features], train_validation[TARGET_COLUMN])
    predicted = np.maximum(0, pipeline.predict(test[selected_features]))
    metrics = evaluate_predictions(test[TARGET_COLUMN], predicted)
    predictions = test[["date", "hour", TARGET_COLUMN]].copy()
    predictions["prediction"] = predicted
    baseline = min(
        (candidate for candidate in candidates if not candidate["include_temperature_square"]),
        key=lambda candidate: (candidate["mae"], -candidate["r2"]),
    )
    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "validation_metrics": {"mae": selected["mae"], "r2": selected["r2"]},
        "baseline_validation_metrics": {"mae": baseline["mae"], "r2": baseline["r2"]},
        "predictions": predictions,
        "coefficients": extract_coefficients(pipeline),
        "candidates": candidates,
        "selection": {
            "source": "validation",
            "selected_model": selected["model"],
            "selected_alpha": selected["alpha"],
            "include_temperature_square": selected["include_temperature_square"],
        },
        "split": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
    }


def save_artifacts(result: dict[str, Any], model_path: Path, report_dir: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["pipeline"], model_path)
    report = {
        **result["metrics"],
        "validation_mae": result["validation_metrics"]["mae"],
        "validation_r2": result["validation_metrics"]["r2"],
        "baseline_validation_mae": result["baseline_validation_metrics"]["mae"],
        "baseline_validation_r2": result["baseline_validation_metrics"]["r2"],
        **result["selection"],
        **result["split"],
        "candidates": result["candidates"],
    }
    (report_dir / "model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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
