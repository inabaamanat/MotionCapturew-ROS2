#!/usr/bin/env python3
"""CLI entry point for the treadmill <-> OpenCap fusion pipeline.

Usage:
    python run_fusion.py --config config.yaml
    python run_fusion.py --config config.yaml --no-overlay      # skip video render
    python run_fusion.py --config config.yaml --overlay-only     # only render
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion.pipeline import run


def main():
    p = argparse.ArgumentParser(description="Treadmill <-> OpenCap fusion")
    p.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    p.add_argument("--no-overlay", action="store_true",
                   help="skip the (slow) overlay video render")
    p.add_argument("--no-qc", action="store_true", help="skip QC figures")
    args = p.parse_args()

    res = run(args.config,
              do_overlay=(False if args.no_overlay else None),
              do_qc=not args.no_qc)
    print("\nDone. Outputs in:", res["out_dir"])


if __name__ == "__main__":
    main()
