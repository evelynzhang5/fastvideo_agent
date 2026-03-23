from core.skill_matcher import match_skill
from core.memory_loader import load_memory

def run(task):
    memory = load_memory(task)
    skill = match_skill(task)

    response = f"""
Task: {task}
Memory: {memory}
Selected Skill: {skill}
"""

    return response