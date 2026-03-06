def dedupe(facts):
    seen = set()
    result = []

    for f in facts:
        if f not in seen:
            result.append(f)
            seen.add(f)

    return result