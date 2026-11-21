"""Check for the manually downloaded Kaggle Water Quality CSV.

Download waterQuality1.csv from the Kaggle dataset page and put it in data/.
Kaggle may require you to be signed in before downloading.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CANDIDATES = [
    DATA_DIR / "waterQuality1.csv",
    DATA_DIR / "water_quality.csv",
    DATA_DIR / "water-quality.csv",
]


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for path in CANDIDATES:
        if path.exists() and path.stat().st_size > 1_000:
            print(f"Dataset found: {path}")
            print("Next run: python train.py")
            return
    print("The expanded dataset has not been placed in the data folder yet.\n")
    print("1. Open: https://www.kaggle.com/datasets/mssmartypants/water-quality")
    print("2. Download the dataset and extract waterQuality1.csv")
    print(f"3. Copy the CSV into: {DATA_DIR}")
    print("4. Run: python train.py")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
