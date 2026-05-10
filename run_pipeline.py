"""
GeoRadar — End-to-End Pipeline Runner
Team Astrix · AESH 2026 · Green Radar Systems

Runs the full pipeline in sequence:
  1. Signal generation    → data/synthetic_labels/
  2. Duty-cycle analysis  → results/power_comparison.csv
  3. CNN training         → results/georadar_cnn.keras  (requires TensorFlow)
  4. Summary report       → results/pipeline_summary.txt

Usage:
    python run_pipeline.py
    python run_pipeline.py --n_samples 1000 --epochs 80 --skip_train
"""

import argparse
import os
import sys
import time

# Add src/ to path so we can import modules cleanly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from signal_gen  import generate_dataset
from duty_cycle  import surface_entropy_signal, run_duty_cycle, print_power_report, power_reduction_percent


def banner(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def run(args):
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../data/synthetic_labels", exist_ok=True)

    summary_lines = [
        "GeoRadar Pipeline Summary",
        f"Run time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # ── Stage 1: Signal Generation ─────────────────────────────────────────────
    banner("Stage 1 / 3 — Signal Generation")
    X, y = generate_dataset(
        n_samples=args.n_samples,
        f_start=args.f_start,
        f_stop=args.f_stop,
        output_dir="../data/synthetic_labels",
        seed=args.seed,
    )
    summary_lines += [
        f"Stage 1 — Signal Generation",
        f"  Samples generated : {args.n_samples}",
        f"  Frequency range   : {args.f_start/1e6:.0f}–{args.f_stop/1e6:.0f} MHz",
        f"  Label distribution: {dict(zip(*__import__('numpy').unique(y, return_counts=True)))}",
        "",
    ]

    # ── Stage 2: Duty-Cycle Analysis ───────────────────────────────────────────
    banner("Stage 2 / 3 — Adaptive Duty-Cycle Analysis")
    entropy_seq = surface_entropy_signal(n_positions=1000, seed=args.seed,
                                         water_fraction=0.30)
    flags, log = run_duty_cycle(entropy_seq, threshold=3.5)
    print_power_report(log)
    reduction = power_reduction_percent(log)
    summary_lines += [
        f"Stage 2 — Duty-Cycle Analysis",
        f"  Radar activations : {log.n_activated} / {log.n_positions} ({log.n_activated/log.n_positions*100:.1f}%)",
        f"  Power reduction   : {reduction:.1f}%",
        f"  GeoRadar energy   : {log.georadar_energy_J*1000:.2f} mJ",
        f"  Conventional GPR  : {log.conventional_energy_J*1000:.2f} mJ",
        "",
    ]

    # ── Stage 3: CNN Training ──────────────────────────────────────────────────
    if args.skip_train:
        banner("Stage 3 / 3 — CNN Training [SKIPPED]")
        print("  Pass --skip_train=False or omit flag to enable.")
        summary_lines += ["Stage 3 — CNN Training: SKIPPED", ""]
    else:
        banner("Stage 3 / 3 — CNN Training")
        try:
            from cnn_model import train
            report = train(
                data_dir="../data/synthetic_labels",
                model_path="../results/georadar_cnn.keras",
                epochs=args.epochs,
                seed=args.seed,
            )
            acc = report.get("accuracy", "N/A")
            summary_lines += [
                f"Stage 3 — CNN Training",
                f"  Test accuracy : {acc:.3f}" if isinstance(acc, float) else f"  Result: {acc}",
                "",
            ]
        except Exception as e:
            print(f"[ERROR] CNN training failed: {e}")
            summary_lines += [f"Stage 3 — CNN Training: FAILED ({e})", ""]

    # ── Write summary ──────────────────────────────────────────────────────────
    summary_path = "../results/pipeline_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSummary written to {summary_path}")
    banner("Pipeline complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoRadar end-to-end pipeline")
    parser.add_argument("--n_samples",  type=int,   default=500)
    parser.add_argument("--f_start",    type=float, default=200e6)
    parser.add_argument("--f_stop",     type=float, default=1e9)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip CNN training (useful for quick demo)")
    run(parser.parse_args())
