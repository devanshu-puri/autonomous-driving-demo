from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automobile_decision_ai.dataset import ProjectPaths, discover_records
from automobile_decision_ai.pipeline import analyze_sample, ensure_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one automobile driving AI demo.")
    parser.add_argument(
        "--sample",
        help="Sample id without the .png extension. If omitted, the first test sample is used.",
    )
    args = parser.parse_args()

    ensure_artifacts(ROOT)
    paths = ProjectPaths.from_root(ROOT)
    records = discover_records(paths)
    test_records = [record for record in records if record.split == "test"]

    sample_id = args.sample or test_records[0].sample_id
    result = analyze_sample(ROOT, sample_id)

    print(f"Sample: {result['sample_id']}")
    print(f"Split: {result['split']}")
    print(f"Driving action: {result['action']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print("Reasoning:")
    for line in result["explanation"]:
        print(f"- {line}")
    print(f"Saved demo image: {result['demo_card_path']}")


if __name__ == "__main__":
    main()
