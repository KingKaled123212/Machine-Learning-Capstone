"""
Load saved evaluation results for all models and datasets,
then generate a comparison table, charts, and a summary report.

Usage:
    python compare_models.py
    python compare_models.py --mock   # use placeholder values (no trained models needed)
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
from pathlib import Path

from utils.visualize import (
    plot_comparison_bar,
    plot_speed_accuracy_scatter,
    plot_training_curves,
)
from utils.metrics import MetricResults


# ---------------------------------------------------------------------------
# Expected result paths
# ---------------------------------------------------------------------------

RESULT_PATHS = {
    "Faster R-CNN\n(Penn-Fudan)":  "outputs/faster_rcnn/pennfudan/test_results.json",
    "YOLOv8n\n(Penn-Fudan)":       "outputs/yolo/pennfudan/test_results.json",
    "Faster R-CNN\n(Oxford Pets)": "outputs/faster_rcnn/oxford_pets/test_results.json",
    "YOLOv8n\n(Oxford Pets)":      "outputs/yolo/oxford_pets/test_results.json",
}

# Mock results used when --mock flag is passed (or files are missing)
MOCK_RESULTS = {
    "Faster R-CNN\n(Penn-Fudan)":  MetricResults(map50=0.847, precision=0.891, recall=0.823, inference_speed=12.1),
    "YOLOv8n\n(Penn-Fudan)":       MetricResults(map50=0.812, precision=0.856, recall=0.798, inference_speed=47.3),
    "Faster R-CNN\n(Oxford Pets)": MetricResults(map50=0.673, precision=0.712, recall=0.651, inference_speed=11.4),
    "YOLOv8n\n(Oxford Pets)":      MetricResults(map50=0.641, precision=0.679, recall=0.624, inference_speed=45.1),
}

TRAINING_TIMES = {
    "Faster R-CNN\n(Penn-Fudan)":  "18 min",
    "YOLOv8n\n(Penn-Fudan)":       "9 min",
    "Faster R-CNN\n(Oxford Pets)": "34 min",
    "YOLOv8n\n(Oxford Pets)":      "16 min",
}


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(use_mock: bool = False) -> dict:
    results = {}
    missing = []

    for label, path in RESULT_PATHS.items():
        p = Path(path)
        if p.exists() and not use_mock:
            data = json.loads(p.read_text())
            results[label] = MetricResults(
                map50=data.get("map50", 0.0),
                map50_95=data.get("map50_95", 0.0),
                precision=data.get("precision", 0.0),
                recall=data.get("recall", 0.0),
                inference_speed=data.get("inference_speed", 0.0),
                per_class_ap=data.get("per_class_ap", {}),
            )
        else:
            missing.append(label)
            results[label] = MOCK_RESULTS[label]

    if missing:
        print(f"[INFO] Using placeholder values for: {missing}")
        print("       Run training first, or use --mock to suppress this message.\n")

    return results


# ---------------------------------------------------------------------------
# Print comparison table to stdout
# ---------------------------------------------------------------------------

def print_table(results: dict) -> None:
    print("\n" + "=" * 90)
    print(f"{'Model':<30} {'mAP@0.5':>8} {'Precision':>10} {'Recall':>8} {'Train Time':>12} {'Inf Speed':>12}")
    print("-" * 90)

    for label, r in results.items():
        clean = label.replace("\n", " ")
        t = TRAINING_TIMES.get(label, "—")
        print(
            f"{clean:<30} "
            f"{r.map50:>8.3f} "
            f"{r.precision:>10.3f} "
            f"{r.recall:>8.3f} "
            f"{t:>12} "
            f"{r.inference_speed:>10.1f} img/s"
        )

    print("=" * 90)


# ---------------------------------------------------------------------------
# Generate all output artefacts
# ---------------------------------------------------------------------------

def generate_outputs(results: dict, out_dir: Path = Path("outputs/comparison")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart
    plot_comparison_bar(
        results,
        metrics=["map50", "precision", "recall"],
        save_path=str(out_dir / "metrics_bar.png"),
        title="Detection Metrics: Faster R-CNN vs YOLOv8n",
    )

    # Speed vs accuracy scatter
    plot_speed_accuracy_scatter(
        results,
        save_path=str(out_dir / "speed_vs_accuracy.png"),
    )

    # Training curves (load from history files if available)
    for model_tag, dataset in [("faster_rcnn", "pennfudan"), ("faster_rcnn", "oxford_pets"),
                                ("yolo", "pennfudan"), ("yolo", "oxford_pets")]:
        hist_path = Path(f"outputs/{model_tag}/{dataset}/history.json")
        if hist_path.exists():
            hist = json.loads(hist_path.read_text())
            plot_training_curves(
                hist,
                save_path=str(out_dir / f"curves_{model_tag}_{dataset}.png"),
                title=f"{model_tag.replace('_', ' ').title()} — {dataset}",
            )

    # Save combined JSON
    summary = {}
    for label, r in results.items():
        summary[label] = {
            "map50": r.map50,
            "precision": r.precision,
            "recall": r.recall,
            "inference_speed": r.inference_speed,
            "training_time": TRAINING_TIMES.get(label, "—"),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nAll comparison outputs saved to {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare trained models")
    parser.add_argument("--mock", action="store_true",
                        help="Use placeholder results (no trained models needed)")
    args = parser.parse_args()

    print("\n=== Model Comparison: Faster R-CNN vs YOLOv8n ===\n")

    results = load_results(use_mock=args.mock)
    print_table(results)
    generate_outputs(results)


if __name__ == "__main__":
    main()
