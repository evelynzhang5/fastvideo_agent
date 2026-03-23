import json
from agent import run_agent

with open("evaluations/questions.json") as f:
    questions = json.load(f)

correct = 0

for q in questions:

    response = run_agent(q["question"])

    success = False

    for keyword in q["expected_keywords"]:
        if keyword.lower() in response.lower():
            success = True

    if success:
        correct += 1

    print("\nQuestion:", q["question"])
    print("Agent:", response)
    print("Correct:", success)

print("\nScore:", correct, "/", len(questions))