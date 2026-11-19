"""Download the public Water Potability CSV used by this educational project."""
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILE = DATA_DIR / "water_potability.csv"
URLS = [
    "https://raw.githubusercontent.com/edanurb/Water-Quality-Dataset-pytorch/main/water_potability.csv",
    "https://raw.githubusercontent.com/Divits0/Water-Potability-Prediction/main/water_potability.csv",
]


def download_dataset(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists() and not force and DATA_FILE.stat().st_size > 10_000:
        return DATA_FILE

    last_error = None
    for url in URLS:
        try:
            request = Request(url, headers={"User-Agent": "SmartWaterQualityStudentProject/1.0"})
            with urlopen(request, timeout=30) as response:
                content = response.read()
            if b"Potability" not in content[:500] or len(content) < 10_000:
                raise ValueError("Downloaded file does not look like the expected CSV dataset.")
            DATA_FILE.write_bytes(content)
            print(f"Dataset downloaded to: {DATA_FILE}")
            print(f"Source: {url}")
            return DATA_FILE
        except Exception as exc:  # try the next public mirror
            last_error = exc

    raise RuntimeError(
        "Could not download the dataset automatically. Download the CSV from "
        "https://www.kaggle.com/datasets/adityakadiwal/water-potability and save it as "
        f"{DATA_FILE}. Last error: {last_error}"
    )


if __name__ == "__main__":
    print(download_dataset())
