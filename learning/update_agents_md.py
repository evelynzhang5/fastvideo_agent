from pathlib import Path

facts_file = Path("../memory/extracted_facts/facts.json")

agents = Path("../AGENTS.md")

facts = facts_file.read_text()

agents.write_text(
    agents.read_text() +
    "\n\n## Learned Facts\n" +
    facts
)