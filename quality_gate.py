import json
import sys
from pathlib import Path

METRICS_PATH = Path("metrics.json")
REQUIRED_ACCURACY = 0.80

def main():
    if not METRICS_PATH.exists():
        print("ERROR: metrics.json was not found.")
        sys.exit(1)
    with METRICS_PATH.open(encoding="utf-8") as f:
        metrics = json.load(f)
    accuracy = float(metrics.get("accuracy", 0))
    print(f"Model accuracy: {accuracy}")
    print(f"Required accuracy: {REQUIRED_ACCURACY}")
    if accuracy < REQUIRED_ACCURACY:
        print("QUALITY GATE FAILED")
        sys.exit(1)
    print("QUALITY GATE PASSED")

if __name__ == "__main__":
    main()
