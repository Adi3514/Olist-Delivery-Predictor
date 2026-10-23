"""Download the Olist dataset from Kaggle into data/raw/.

Needs a Kaggle API token (~/.kaggle/kaggle.json) and `pip install -e ".[kaggle]"`.
Without one, download the zip by hand from
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce and unzip it into data/raw/.

Usage: python -m olist.download
"""
import subprocess
import sys

from olist.config import KAGGLE_DATASET, RAW_DIR, RAW_FILES


def missing_files():
    return [name for name in RAW_FILES.values() if not (RAW_DIR / name).exists()]


def main():
    if not missing_files():
        print(f"All {len(RAW_FILES)} files already in {RAW_DIR}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW_DIR), "--unzip"]
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        sys.exit(
            f"Kaggle download failed ({err}).\n"
            f"Download the zip from https://www.kaggle.com/datasets/{KAGGLE_DATASET} "
            f"and unzip it into {RAW_DIR}"
        )
    still_missing = missing_files()
    if still_missing:
        sys.exit(f"Download finished but these files are missing: {still_missing}")
    print(f"Downloaded {len(RAW_FILES)} files to {RAW_DIR}")


if __name__ == "__main__":
    main()
