"""Build the gated FictionPulper Counterfactual-v1 pretraining dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.counterfactual_narrative.core import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactual-narrative-v1.yaml"))
    parser.add_argument("--max-pairs", type=int, help="Development-only pair cap; output remains STOP")
    args = parser.parse_args()
    result = build_dataset(args.config, max_pairs=args.max_pairs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
