import json
from pathlib import Path

sessions = Path("../memory/sessions")

facts = []

for f in sessions.glob("*.json"):
    data = json.loads(f.read_text())

    for fact in data.get("facts", []):
        facts.append(fact)

print("Extracted facts:")
for f in set(facts):
    print("-", f)