from core.skill_matcher import match_skill
from core.memory_loader import load_memory

def run(task):

    memory = load_memory(task)

    answer = f"""
Based on FastVideo documentation:

{memory}

Answer:
{memory.splitlines()[-1]}
"""

    return answer