from skill_matcher import match_skill
from memory_loader import load_memory

def run(task):

    memory = load_memory()

    skill = match_skill(task)

    print("Memory loaded:", memory)
    print("Using skill:", skill)