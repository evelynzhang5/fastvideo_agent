# FastVideo Setup Skill

Use this skill when the user asks about installation, setup, environment creation, dependencies, tests, or pre-commit.

When answering:
1. Give commands first.
2. Mention the relevant docs path when available.
3. Separate CPU/MPS/GPU instructions if the facts mention different install paths.
4. Include exact package commands such as `conda create`, `uv pip install -e .`, or `uv pip install flash-attn --no-build-isolation -v`.
5. If the retrieved facts are incomplete, say what is missing.

Preferred answer format:
- Short explanation
- Commands
- Notes / troubleshooting