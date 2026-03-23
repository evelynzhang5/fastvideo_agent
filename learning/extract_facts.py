import os
import json
from pathlib import Path

REPO_PATH = "../FastVideo"

facts = []

def extract_from_file(path):
    text = Path(path).read_text(errors="ignore")

    # simple heuristics
    if "pipeline" in text.lower():
        facts.append({
            "type": "architecture",
            "content": f"{path} contains pipeline-related logic"
        })

    if "def forward" in text:
        facts.append({
            "type": "architecture",
            "content": f"{path} defines forward pass logic"
        })

    if "register" in text.lower():
        facts.append({
            "type": "architecture",
            "content": f"{path} uses registry system"
        })


for root, _, files in os.walk(REPO_PATH):
    for file in files:
        if file.endswith(".py"):
            extract_from_file(os.path.join(root, file))

# save
with open("../memory/extracted_facts/auto_facts.json", "w") as f:
    json.dump(facts, f, indent=2)

print("Extracted", len(facts), "facts")