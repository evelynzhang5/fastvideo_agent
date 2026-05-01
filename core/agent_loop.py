from core.memory_loader import load_memory
from core.skill_matcher import match_skill
import ollama


MODEL_NAME = "llama3"


def run(task: str) -> str:
    """
    Main agent loop:
    1. Retrieve relevant repo memory
    2. Match a skill
    3. Build a grounded prompt
    4. Ask local Ollama model
    5. Return answer
    """

    memory = load_memory(task, top_k=8)
    skill = match_skill(task)

    prompt = f"""
    You are an onboarding assistant for the FastVideo repository.

    Use the retrieved repository facts below to answer the user's question.
    Answer directly and concretely.
    Prefer exact file paths, class names, function names, method names, command names, and source-level details.
    If the question asks "what is X", define X first, then give where it is implemented and how it is used.
    If the question asks "how do you", give the command or steps first.
    If the retrieved facts do not contain enough information, say what is missing instead of guessing.

    Matched skill:
    {skill["name"]}

    Skill instructions:
    {skill["content"] or "No special skill instructions matched."}

    Retrieved repository facts:
    {memory}

    User question:
    {task}

Answer:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"]


if __name__ == "__main__":
    print(run("What does FastVideo do?"))