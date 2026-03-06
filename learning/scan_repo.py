import os
import json
from pathlib import Path

REPO_PATH = "FastVideo"

memory = {
    "repo": "FastVideo",
    "directories": [],
    "python_modules": [],
    "important_files": [],
    "commands": []
}

for root, dirs, files in os.walk(REPO_PATH):

    # record directory
    memory["directories"].append(root)

    for file in files:

        full_path = os.path.join(root, file)

        # record python modules
        if file.endswith(".py"):
            memory["python_modules"].append(full_path)

        # detect important files
        if file in ["README.md", "pyproject.toml", "requirements.txt"]:
            memory["important_files"].append(full_path)

            content = Path(full_path).read_text(errors="ignore")

            for line in content.split("\n"):
                if "pip install" in line or "conda" in line:
                    memory["commands"].append(line.strip())

# save memory
output = "memory/extracted_facts/fastvideo_repo_structure.json"

with open(output, "w") as f:
    json.dump(memory, f, indent=2)

print("Memory generated:", output)