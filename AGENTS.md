# AGENTS.md

## Purpose

This repository provides memory and skills for an AI coding agent that assists
developers onboarding to the FastVideo repository.

The agent should:
- help contributors understand the repo structure
- guide environment setup
- remember previous setup issues
- reuse knowledge from previous sessions

---

# Memory System

The agent uses three memory layers:

1. Session memory
memory/sessions/

2. Extracted facts
memory/extracted_facts/

3. User preferences
memory/user_prefs/

At session start:
- load AGENTS.md
- load extracted facts
- optionally search session logs

At session end:
- save session summary
- extract durable facts

---

# Skill System

Skills live in:

skills/

Each skill contains:

SKILL.md — instructions
(optional) scripts

---

# FastVideo Knowledge

Important repository directories:

fastvideo/models
fastvideo/training
fastvideo/inference

---

# Rules

When learning something reusable:
update extracted_facts or AGENTS.md.

Never store secrets or API keys.