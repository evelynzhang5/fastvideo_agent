import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REGISTRY_PATH = BASE_DIR / "skills" / "registry.json"

def match_skill(task):

    registry = json.load(open(REGISTRY_PATH))

    for skill in registry:
        if skill["name"] in task.lower():
            return skill["name"]

    return "default"