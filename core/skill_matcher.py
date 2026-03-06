import json

def match_skill(task):

    registry = json.load(open("skills/registry.json"))

    for skill in registry["skills"]:
        for keyword in skill["keywords"]:
            if keyword in task.lower():
                return skill

    return None