import json

def load_memory():

    with open("memory/extracted_facts/facts.json") as f:
        return json.load(f)