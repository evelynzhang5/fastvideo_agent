from core.memory_loader import load_memory

questions = [
    "Where is NPU memory cleared?",
    "How does FastVideo choose the attention backend?",
    "Where are ComfyUI inputs validated?",
    "How does FastVideo run inference?",
]

for q in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", q)
    print("=" * 80)
    print(load_memory(q, top_k=3))
