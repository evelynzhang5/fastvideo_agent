from sentence_transformers import SentenceTransformer
import json
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# load facts once
with open("memory/extracted_facts/facts.json") as f:
    FACTS = json.load(f)

FACT_EMBEDS = model.encode([f["content"] for f in FACTS])


from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_memory(query):
    facts = json.load(open("memory/extracted_facts/auto_facts.json"))

    texts = [f["content"] for f in facts]

    embeddings = model.encode(texts)
    query_emb = model.encode([query])[0]

    scores = np.dot(embeddings, query_emb)

    top_k = np.argsort(scores)[-5:][::-1]

    return "\n\n".join([texts[i] for i in top_k])