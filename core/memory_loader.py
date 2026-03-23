from sentence_transformers import SentenceTransformer
import json
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# load facts once
with open("memory/extracted_facts/facts.json") as f:
    FACTS = json.load(f)

FACT_EMBEDS = model.encode([f["content"] for f in FACTS])


def load_memory(query):
    query_embed = model.encode([query])[0]

    scores = np.dot(FACT_EMBEDS, query_embed)

    # top-k facts
    top_k = np.argsort(scores)[-5:]

    return "\n".join([FACTS[i]["content"] for i in top_k])