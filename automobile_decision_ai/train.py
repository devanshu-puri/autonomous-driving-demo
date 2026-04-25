from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automobile_decision_ai.pipeline import train_and_save


def main() -> None:
    metrics = train_and_save(ROOT)
    print("Training completed.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
