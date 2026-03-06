name: continual_learning

description:
Convert session logs into durable memory.

inputs:
memory/sessions/

steps:

1. scan session logs

2. extract reusable facts

examples:

"FastVideo requires torch >=2.1"

3. remove duplicates

4. update:

memory/extracted_facts/
AGENTS.md