import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
SKILLS_DIR = BASE_DIR / "skills"
REGISTRY_PATH = SKILLS_DIR / "registry.json"


KEYWORD_RULES = {
    "fastvideo_setup": [
        "install",
        "installation",
        "setup",
        "environment",
        "conda",
        "pip",
        "uv",
        "flash-attn",
        "flash attention",
        "requirements",
        "pre-commit",
        "tests",
        "clone",
        "from source",
    ],
    "repo_navigation": [
        "where",
        "located",
        "implemented",
        "directory",
        "folder",
        "file",
        "repo",
        "repository",
        "structure",
        "architecture",
        "component",
        "module",
    ],
    "continual_learning": [
        "learn",
        "memory",
        "facts",
        "extract",
        "update",
        "remember",
        "improve",
        "evaluation",
        "eval",
    ],
}


def load_registry():
    if not REGISTRY_PATH.exists():
        return []

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def read_skill_md(skill_name: str) -> str:
    skill_path = SKILLS_DIR / skill_name / "SKILL.md"

    if not skill_path.exists():
        return ""

    return skill_path.read_text(encoding="utf-8")


def match_skill(task: str) -> dict:
    """
    Choose the most relevant skill for a task and return both the skill name
    and the actual SKILL.md content.

    Returns:
        {
            "name": "fastvideo_setup",
            "content": "...skill instructions..."
        }
    """

    task_lower = task.lower()
    registry = load_registry()

    best_name = "default"
    best_score = 0

    for skill in registry:
        name = skill.get("name", "")
        description = skill.get("description", "")

        score = 0

        # Exact skill name match.
        if name and name.lower() in task_lower:
            score += 10

        # Description word overlap.
        for word in description.lower().split():
            if len(word) > 3 and word in task_lower:
                score += 1

        # Manual keyword rules.
        for keyword in KEYWORD_RULES.get(name, []):
            if keyword in task_lower:
                score += 3

        if score > best_score:
            best_score = score
            best_name = name

    content = read_skill_md(best_name) if best_name != "default" else ""

    return {
        "name": best_name,
        "content": content,
    }