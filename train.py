"""Train the FlowCast arrivals model.

    python train.py --data data/uihc_ed_arrivals.csv     (real data, recommended)
    python train.py --synthetic                          (fake data, demo only)
"""
import argparse
import sys
from pathlib import Path

import forecaster as fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/uihc_ed_arrivals.csv", help="hourly ED arrivals CSV")
    ap.add_argument("--synthetic", action="store_true", help="use made-up data (accuracy means nothing)")
    args = ap.parse_args()

    if args.synthetic:
        print("Using SYNTHETIC data. The accuracy below says nothing about real hospitals.")
        series, source = fc.make_synthetic(), "synthetic data"
    else:
        path = Path(args.data)
        if not path.exists():
            sys.exit(f"Dataset not found: {path}\nPut the Iowa CSV there, or run with --synthetic to try the pipeline.")
        series, source = fc.load_dataset(path), path.name

    print(f"Loaded {len(series):,} hourly rows ({series.index[0].date()} to {series.index[-1].date()}). Training...")
    m = fc.train(series, source, synthetic=args.synthetic)

    print(f"\nHeld-out test: newest {m['test_hours']:,} hours, never seen during training")
    print(f"Average error, patients per hour (lower is better):")
    print(f"  Model              {m['mae_model']}")
    print(f"  Typical-hour guess {m['mae_typical_hour']}")
    print(f"  Last 6h average    {m['mae_last_6h_average']}")
    print(f"Model is {m['improvement_pct']}% better than the typical-hour guess.")
    print("\nBy hours ahead:")
    for h, v in m["by_horizon"].items():
        print(f"  {h:>2}h  model {v['model']}  typical {v['typical_hour']}  last-6h {v['last_6h_average']}")
    print(f"\nSaved model to {fc.MODEL_PATH}")


if __name__ == "__main__":
    main()
