import json
from pathlib import Path


def read_json(filename: Path) -> dict:
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data
