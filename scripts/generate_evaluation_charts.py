"""Generate classroom-ready model-function and evaluation charts for RideCast."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.model import FEATURE_COLUMNS, TARGET_COLUMN, prepare_training_frame


DATA_PATH = ROOT / "data" / "SeoulBikeData.csv"
MODEL_PATH = ROOT / "models" / "ridecast_pipeline.joblib"
REPORT_DIR = ROOT / "reports"


def _representative_inputs(frame: pd.DataFrame) -> dict[str, object]:
    """Return train/validation-only reference values for conditional plots."""
    train_validation = frame.iloc[: int(len(frame) * 0.80)]
    numeric = ["month", "temperature", "humidity", "rainfall", "snowfall"]
    values: dict[str, object] = {
        column: float(train_validation[column].median()) for column in numeric
    }
    values["weekday"] = str(train_validation["weekday"].mode().iloc[0])
    values["holiday"] = str(train_validation["holiday"].mode().iloc[0])
    values["functioning_day"] = str(train_validation["functioning_day"].mode().iloc[0])
    return values


def _prediction_frame(hours: np.ndarray, temperatures: np.ndarray | None, reference: dict[str, object]) -> pd.DataFrame:
    rows = []
    if temperatures is None:
        temperatures = np.full_like(hours, float(reference["temperature"]), dtype=float)
    for hour, temperature in zip(hours, temperatures):
        row = {
            "hour": int(hour),
            "weekday": reference["weekday"],
            "month": int(round(float(reference["month"]))),
            "temperature": float(temperature),
            "temperature_sq": float(temperature) ** 2,
            "humidity": float(reference["humidity"]),
            "rainfall": float(reference["rainfall"]),
            "snowfall": float(reference["snowfall"]),
            "holiday": reference["holiday"],
            "functioning_day": reference["functioning_day"],
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def _predict(pipeline, frame: pd.DataFrame) -> np.ndarray:
    # The selected model excludes temperature_sq; the pipeline itself determines the
    # transformed feature schema, so pass the same columns used by the final model.
    selected = [column for column in FEATURE_COLUMNS if column != "temperature_sq"]
    return np.maximum(0, pipeline.predict(frame[selected]))


def generate_curve(pipeline, frame: pd.DataFrame, out_path: Path) -> None:
    reference = _representative_inputs(frame)
    hours = np.arange(24)
    curve_frame = _prediction_frame(hours, None, reference)
    predictions = _predict(pipeline, curve_frame)

    # A near-square canvas keeps labels readable when the figure is placed
    # beside another chart on a 16:9 presentation slide.
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=220)
    ax.plot(hours, predictions, color="#3157D5", linewidth=3, marker="o", markersize=4)
    ax.fill_between(hours, predictions, alpha=0.12, color="#3157D5")
    ax.set_title("Conditional prediction curve", fontsize=16, weight="bold")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Predicted rented bikes / hour")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.03,
        "Other inputs fixed at train/validation median or mode;\nthis is a conditional slice of the multivariable model.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 4},
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_heatmap(pipeline, frame: pd.DataFrame, out_path: Path) -> None:
    reference = _representative_inputs(frame)
    train_validation = frame.iloc[: int(len(frame) * 0.80)]
    temperatures = np.linspace(
        float(train_validation["temperature"].quantile(0.05)),
        float(train_validation["temperature"].quantile(0.95)),
        36,
    )
    hours = np.arange(24)
    grid_hours, grid_temperatures = np.meshgrid(hours, temperatures)
    grid_frame = _prediction_frame(grid_hours.ravel(), grid_temperatures.ravel(), reference)
    values = _predict(pipeline, grid_frame).reshape(len(temperatures), len(hours))

    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=220)
    image = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        extent=[hours.min(), hours.max(), temperatures.min(), temperatures.max()],
        cmap="viridis",
    )
    ax.set_title("Conditional prediction surface", fontsize=16, weight="bold")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xticks(np.arange(0, 24, 2))
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Predicted rented bikes / hour")
    ax.text(
        0.02,
        0.03,
        "Other inputs fixed at representative train/validation values",
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 3},
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_dashboard(out_path: Path) -> None:
    predictions = pd.read_csv(REPORT_DIR / "test_predictions.csv")
    metrics = json.loads((REPORT_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    actual = predictions[TARGET_COLUMN].to_numpy(dtype=float)
    predicted = predictions["prediction"].to_numpy(dtype=float)
    residual = actual - predicted

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=200)
    scatter = axes[0].scatter(actual, predicted, s=14, alpha=0.42, color="#3157D5", edgecolors="none")
    limit = float(max(actual.max(), predicted.max()))
    axes[0].plot([0, limit], [0, limit], "--", color="#D62F63", linewidth=2, label="Ideal line y=x")
    axes[0].set_title("Actual vs predicted", weight="bold")
    axes[0].set_xlabel("Actual rented bikes / hour")
    axes[0].set_ylabel("Predicted rented bikes / hour")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(predicted, residual, s=14, alpha=0.42, color="#F08A24", edgecolors="none")
    axes[1].axhline(0, linestyle="--", color="#333333", linewidth=1.5)
    axes[1].set_title("Residuals vs predicted", weight="bold")
    axes[1].set_xlabel("Predicted rented bikes / hour")
    axes[1].set_ylabel("Residual = actual − predicted")
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        f"Final test-set evaluation  |  MAE = {metrics['mae']:.2f}  |  R² = {metrics['r2']:.3f}  |  n = {len(predictions):,}",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = pd.read_csv(DATA_PATH, encoding="cp1252")
    frame = prepare_training_frame(raw).sort_values(["date", "hour"]).reset_index(drop=True)
    pipeline = joblib.load(MODEL_PATH)
    generate_curve(pipeline, frame, REPORT_DIR / "prediction_curve.png")
    generate_heatmap(pipeline, frame, REPORT_DIR / "prediction_heatmap.png")
    generate_dashboard(REPORT_DIR / "evaluation_dashboard.png")
    outputs = [REPORT_DIR / name for name in ("prediction_curve.png", "prediction_heatmap.png", "evaluation_dashboard.png")]
    for output in outputs:
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"Chart was not created: {output}")
        print(f"{output}: {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
