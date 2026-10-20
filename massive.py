"""Load the French and Japanese parts of MASSIVE (https://github.com/alexa/massive)."""

import json
import tarfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
ARCHIVE = DATA_DIR / "amazon-massive-dataset-1.1.tar.gz"


def download():
    DATA_DIR.mkdir(exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)


def load(locale):
    """Return {"train": (texts, intents), "dev": ..., "test": ...} for a locale like "fr-FR"."""
    download()
    with tarfile.open(ARCHIVE) as tar:
        lines = tar.extractfile(f"1.1/data/{locale}.jsonl").read().decode().splitlines()

    splits = {"train": ([], []), "dev": ([], []), "test": ([], [])}
    for line in lines:
        row = json.loads(line)
        texts, intents = splits[row["partition"]]
        texts.append(row["utt"])
        intents.append(row["intent"])
    return splits
