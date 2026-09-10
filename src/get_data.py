"""Fetch the Kaggle 'Customer Support on Twitter' dataset -> data/twcs.csv.

Two ways to get the data:
  1. Kaggle API: set KAGGLE_USERNAME / KAGGLE_KEY in .env (or ~/.kaggle/kaggle.json),
     then `python -m src.get_data`.
  2. Manual: download https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
     unzip, and drop `twcs.csv` into ./data/.
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile

from src.config import DATA, RAW_CSV

SLUG = "thoughtvector/customer-support-on-twitter"


def main() -> None:
    if RAW_CSV.exists():
        print(f"Already present: {RAW_CSV} ({RAW_CSV.stat().st_size / 1e6:.0f} MB)")
        return

    DATA.mkdir(exist_ok=True)
    if not (os.getenv("KAGGLE_KEY") or (os.path.expanduser("~/.kaggle/kaggle.json"))):
        sys.exit(
            "No Kaggle credentials found. Set KAGGLE_USERNAME/KAGGLE_KEY in .env "
            "or download twcs.csv manually into ./data/ (see module docstring)."
        )

    print(f"Downloading {SLUG} via Kaggle API ...")
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", SLUG,
         "-p", str(DATA)],
        check=True,
    )
    zpath = next(DATA.glob("*.zip"))
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA)
    zpath.unlink()
    print(f"Done: {RAW_CSV}")


if __name__ == "__main__":
    main()
