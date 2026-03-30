# from core.skill_matcher import match_skill
# from core.memory_loader import load_memory
# from openai import OpenAI

# client = OpenAI()   # requires OPENAI_API_KEY


# def run(task):

#     # 1. retrieve memory (RAG)
#     memory = load_memory(task)

#     # 2. match skill (even if not used yet — important for future)
#     skill = match_skill(task)

#     # 3. build structured prompt (VERY IMPORTANT 🔥)
#     prompt = f"""
# You are an expert FastVideo developer assistant.

# Your job:
# - Help users understand FastVideo codebase, setup, and workflows
# - Give actionable, precise answers
# - Reference relevant modules/files when possible

# ----------------------
# RELEVANT KNOWLEDGE:
# {memory}
# ----------------------

# AVAILABLE SKILL:
# {skill}
# ----------------------

# INSTRUCTIONS:
# - Use the knowledge above to answer the question
# - If relevant, mention file paths or components
# - Be concise but informative
# - If unsure, say what is missing

# QUESTION:
# {task}
# """

#     # 4. call LLM
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a senior AI engineer specializing in FastVideo and large ML systems."
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )

#     answer = response.choices[0].message.content

#     # 5. debugging (SUPER useful for eval)
#     print("\n--- DEBUG ---")
#     print("Task:", task)
#     print("Skill:", skill)
#     print("Retrieved memory:\n", memory[:500])
#     print("---------------\n")

#     return answer
from ollama import chat
from core.memory_loader import load_memory
from core.skill_matcher import match_skill


def run(task):
    memory = load_memory(task)
    skill = match_skill(task)

    prompt = f"""
You are a FastVideo expert.

Knowledge:
{memory}

Skill:
{skill}

Question:
{task}
"""

    response = chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]