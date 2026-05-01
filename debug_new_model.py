from core.memory_loader import load_memory

questions = [
    "How do you add a new model?",
    "What should I check for a new DiT model?",
    "Should text encoders use DistributedAttention?",
    "What config files are needed for new model support?",
]

for q in questions:
    print("\n" + "=" * 100)
    print("QUESTION:", q)
    print("=" * 100)
    print(load_memory(q, top_k=5))
