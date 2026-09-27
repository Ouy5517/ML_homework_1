"""Download, train and report the RideCast model."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .model import TARGET_COLUMN, prepare_training_frame, save_artifacts, train_and_evaluate
except ImportError:  # pragma: no cover - supports direct script execution
    from model import TARGET_COLUMN, prepare_training_frame, save_artifacts, train_and_evaluate

DATA_URL = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dataset(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载数据集: {DATA_URL}")
    with urlopen(DATA_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if not names:
        raise RuntimeError("数据压缩包中没有 CSV 文件")
    source = names[0]
    path.write_bytes(archive.read(source))
    return path


def write_plot(predictions: pd.DataFrame, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(predictions[TARGET_COLUMN], predictions["prediction"], s=12, alpha=0.35, color="#5865f2")
    low = min(predictions[TARGET_COLUMN].min(), predictions["prediction"].min())
    high = max(predictions[TARGET_COLUMN].max(), predictions["prediction"].max())
    axis.plot([low, high], [low, high], linestyle="--", color="#de2761", linewidth=1.5)
    axis.set_xlabel("Actual rented bike count (bikes/hour)")
    axis.set_ylabel("Predicted rented bike count (bikes/hour)")
    axis.set_title("RideCast test set: actual vs predicted")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(report_dir / "actual_vs_predicted.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "SeoulBikeData.csv")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "ridecast_pipeline.joblib")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()
    source = ensure_dataset(args.data)
    try:
        raw = pd.read_csv(source, encoding="utf-8")
    except UnicodeDecodeError:
        # UCI distributes this CSV with a single cp1252 degree-symbol byte.
        raw = pd.read_csv(source, encoding="cp1252")
    frame = prepare_training_frame(raw)
    result = train_and_evaluate(frame)
    save_artifacts(result, args.model, args.reports)
    write_plot(result["predictions"], args.reports)
    print(f"训练完成: {len(frame)} 条记录")
    print(f"测试 MAE: {result['metrics']['mae']:.2f} bikes/hour")
    print(f"测试 R2: {result['metrics']['r2']:.3f}")
    print(f"模型: {args.model}")


if __name__ == "__main__":
    main()
