import json
from agent import run_agent

with open("evaluations/questions.json") as f:
    questions = json.load(f)

correct = 0

for q in questions:

    response = run_agent(q["question"])

    success = False
    keywords = q["expected_keywords"]
    match_count = sum(k in response.lower() for k in keywords)

    success = match_count >= max(1, int(0.6 * len(keywords)))
    if success:
        correct += 1

    print("\nQuestion:", q["question"])
    print("Agent:", response)
    print("Correct:", success)

print("\nScore:", correct, "/", len(questions))