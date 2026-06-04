import json
from pathlib import Path


def load_catalog(path: str = "data/catalog.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_goldens(path: str = "goldens.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
