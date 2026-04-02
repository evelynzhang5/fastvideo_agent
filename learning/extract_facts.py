import os
import ast
import json
import hashlib
from typing import List, Dict, Optional

# =========================
# CONFIG
# =========================

REPO_PATH = "/Users/evelynzhang/Documents/GitHub/fastvideo_agent/FastVideo"
OUTPUT_PATH = "memory/extracted_facts/auto_facts.json"
CACHE_PATH = "memory/cache/summaries.json"
SUMMARY_VERSION = "v3"

IMPORTANT_KEYWORDS = [
    "pipeline",
    "pipelines",
    "entrypoints",
    "entrypoint",
    "model",
    "models",
    "training",
    "trainer",
    "train",
    "callback",
    "callbacks",
    "optimizer",
    "worker",
    "attention",
    "denoising",
    "encoding",
    "encoder",
    "decoding",
    "decoder",
    "dataset",
    "dataloader",
    "vae",
    "inference",
]

SKIP_DIRS = [".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist"]
MAX_CHARS_PER_CHUNK = 2000
MAX_CHUNKS = 2


# =========================
# FILE HELPERS
# =========================

def safe_read_file(path: str) -> Optional[str]:
    for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except Exception:
            continue
    return None


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache() -> Dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache: Dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


# =========================
# FILTERING
# =========================

def should_skip_root(root: str) -> bool:
    return any(skip in root for skip in SKIP_DIRS)


def is_important(file_path: str) -> bool:
    lower = file_path.lower()

    important_path_hits = any(k in lower for k in IMPORTANT_KEYWORDS)

    priority_dirs = [
    "/fastvideo/pipelines/",
    "/fastvideo/models/",
    "/fastvideo/training/",
    "/fastvideo/train/",
    "/fastvideo/train/methods/",
    "/fastvideo/train/callbacks/",
    "/fastvideo/train/utils/",
    "/fastvideo/attention/",
    "/fastvideo/dataset/",
    "/fastvideo/entrypoints/",
    "/fastvideo/worker/",
    "/comfyui/video_generator/",
    "/docs/inference/",
    "/docs/attention/",
    "/docs/design/",
]

    in_priority_dir = any(p in lower for p in priority_dirs)

    return important_path_hits or in_priority_dir


def should_skip_file(file_path: str, code: Optional[str]) -> bool:
    lower = file_path.lower()

    if "/tests/" in lower or "\\tests\\" in lower:
        return True

    if code is not None and len(code.strip()) < 50:
        return True

    return False


def is_low_value_file(file_path: str, info: Dict) -> bool:
    filename = os.path.basename(file_path)
    lower = file_path.lower()

    if filename == "__init__.py":
        return True

    if "/third_party/" in lower and not is_important(file_path):
        return True

    if "/benchmarks/" in lower and not is_important(file_path):
        return True

    if "/scripts/" in lower and not is_important(file_path):
        return True

    if "/examples/" in lower and not is_important(file_path):
        return True

    if not info["functions"] and not info["classes"] and not info["doc"]:
        return True

    if len(info["functions"]) == 0 and len(info["classes"]) <= 1 and not is_important(file_path):
        return True

    return False


# =========================
# CHUNKING
# =========================

def chunk_code(code: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    return [code[i:i + max_chars] for i in range(0, len(code), max_chars)]


# =========================
# AST EXTRACTION
# =========================

def extract_python_info(file_path: str):
    code = safe_read_file(file_path)
    if code is None:
        return None

    try:
        tree = ast.parse(code)
    except Exception:
        return None

    functions = []
    classes = []
    docstring = ast.get_docstring(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)

    return {
        "code": code,
        "functions": sorted(set(functions)),
        "classes": sorted(set(classes)),
        "doc": docstring
    }


# =========================
# HEURISTICS
# =========================
def guess_file_purpose(file_path: str) -> str:
    name = file_path.lower()

    if "pipeline" in name:
        return "defines pipeline logic and stage orchestration"
    if "workflow" in name:
        return "defines workflow orchestration and execution flow"
    if "config" in name:
        return "defines configuration objects and CLI/config parsing"
    if "registry" in name:
        return "registers and resolves models, pipelines, or components"
    if "logger" in name or "logging" in name:
        return "handles logging and runtime diagnostics"
    if "env" in name or "platform" in name:
        return "handles runtime environment or platform-specific behavior"
    if "hook" in name:
        return "defines hooks for modifying module behavior at runtime"
    if "layer" in name or "/layers/" in name:
        return "implements reusable neural network layers or tensor ops"
    if "denoising" in name:
        return "implements diffusion denoising step"
    if "encoding" in name or "encoder" in name:
        return "handles encoding of inputs or latent representations"
    if "decoding" in name or "decoder" in name:
        return "handles decoding to output representations"
    if "dataset" in name or "dataloader" in name:
        return "handles dataset loading and preprocessing"
    if "worker" in name:
        return "handles distributed worker execution"
    if "attention" in name:
        return "implements attention mechanisms"
    if "vae" in name:
        return "implements VAE encoding and decoding"
    if "model" in name or "dit" in name or "dits" in name:
        return "implements model architecture"
    if "entrypoints" in name or "entrypoint" in name:
        return "defines entrypoint APIs for inference or orchestration"
    if "trainer" in name or "training" in name or "/train/" in name:
        return "implements model training logic"
    if "inference" in name:
        return "implements inference-time execution"
    return "general utility or module"

def infer_tags(file_path: str) -> List[str]:
    lower = file_path.lower()
    tags = []

    tag_rules = {
        "training": ["/train/", "/training/", "trainer", "callback", "optimizer"],
        "entrypoint": ["/entrypoint/", "/entrypoints/"],
        "pipeline": ["pipeline", "pipelines"],
        "model": ["model", "models", "dit", "dits"],
        "vae": ["vae"],
        "encoder": ["encoder", "encoding"],
        "decoder": ["decoder", "decoding"],
        "distributed": ["worker", "distributed"],
        "attention": ["attention"],
        "dataset": ["dataset", "dataloader"],
        "inference": ["inference"],
        "denoising": ["denoising"],
        "workflow": ["workflow"],
        "config": ["config", "configs"],
        "registry": ["registry"],
        "logging": ["logger", "logging"],
        "platform": ["platform", "env", "envs"],
        "hook": ["hook", "hooks"],
        "layer": ["/layers/", "layernorm", "embedding", "mlp", "rotary"],
    }

    for tag, keywords in tag_rules.items():
        if any(k in lower for k in keywords):
            if tag not in tags:
                tags.append(tag)

    return tags

# =========================
# SUMMARIZATION
# =========================

def summarize_chunk_with_prompt(file_path: str, chunk: str, info: Dict) -> Dict:
    purpose = guess_file_purpose(file_path)
    tags = infer_tags(file_path)

    chunk_lower = chunk.lower()
    chunk_signals = []

    keyword_map = {
        "forward": "forward-pass logic",
        "backward": "backward or gradient logic",
        "checkpoint": "checkpointing-related logic",
        "scheduler": "scheduler-related logic",
        "optimizer": "optimizer-related logic",
        "dataloader": "dataloader-related logic",
        "tokenize": "tokenization or input-preparation logic",
        "cache": "cache management logic",
        "streaming": "streaming execution logic",
        "load": "loading or setup logic",
        "save": "saving or export logic",
        "validation": "validation-related logic",
        "train": "training-step logic",
        "infer": "inference-related logic",
        "compile": "compile or optimization logic",
        "latent": "latent-space processing logic",
        "attention": "attention-related logic",
    }

    for key, desc in keyword_map.items():
        if key in chunk_lower:
            chunk_signals.append(desc)

    doc = info["doc"][:220] if info["doc"] else None

    return {
        "purpose": purpose,
        "tags": tags,
        "signals": sorted(set(chunk_signals)),
        "doc": doc,
    }


def merge_chunk_summaries(file_path: str, info: Dict, chunk_summaries: List[Dict]) -> str:
    merged_tags = []
    merged_signals = []
    doc = None

    for s in chunk_summaries:
        for t in s.get("tags", []):
            if t not in merged_tags:
                merged_tags.append(t)

        for sig in s.get("signals", []):
            if sig not in merged_signals:
                merged_signals.append(sig)

        if not doc and s.get("doc"):
            doc = s["doc"]

    purpose = guess_file_purpose(file_path)
    functions = ", ".join(info["functions"][:8]) if info["functions"] else "None"
    classes = ", ".join(info["classes"][:8]) if info["classes"] else "None"

    signal_text = "; ".join(merged_signals[:4]) if merged_signals else "implementation details for this subsystem"

    summary = (
        f"This file likely {purpose}. "
        f"It belongs to the {', '.join(merged_tags) or 'general'} subsystem and contains {signal_text}. "
        f"Key functions: {functions}. "
        f"Key classes: {classes}."
    )

    if doc:
        summary += f" Docstring signal: {doc}."

    return summary


def summarize_file(file_path: str, code: str, info: Dict, cache: Dict) -> str:
    file_hash = compute_hash(code)
    cache_key = f"{SUMMARY_VERSION}:{file_path}:{file_hash}"

    if cache_key in cache:
        return cache[cache_key]

    if is_important(file_path):
        chunks = chunk_code(code)
        selected_chunks = chunks[:MAX_CHUNKS]

        chunk_summaries = [
            summarize_chunk_with_prompt(file_path, chunk, info)
            for chunk in selected_chunks
        ]

        final_summary = merge_chunk_summaries(file_path, info, chunk_summaries)
    else:
        final_summary = (
            f"This file likely {guess_file_purpose(file_path)}. "
            f"It contains functions {', '.join(info['functions'][:5]) or 'None'} "
            f"and classes {', '.join(info['classes'][:5]) or 'None'}."
        )

    cache[cache_key] = final_summary
    return final_summary


# =========================
# DEDUPLICATION
# =========================

def normalize_summary(text: str) -> str:
    return " ".join(text.lower().split())


def is_duplicate(new_summary: str, existing_summaries: List[str]) -> bool:
    new_norm = normalize_summary(new_summary)
    new_prefix = new_norm[:160]

    for s in existing_summaries:
        s_norm = normalize_summary(s)
        if new_prefix and new_prefix in s_norm:
            return True
        if s_norm[:160] == new_prefix:
            return True

    return False


def fact_signature(file_path: str, info: Dict, summary: str) -> str:
    key = {
        "purpose": guess_file_purpose(file_path),
        "tags": infer_tags(file_path),
        "functions": info["functions"][:8],
        "classes": info["classes"][:8],
        "summary_prefix": normalize_summary(summary)[:180],
    }
    return json.dumps(key, sort_keys=True)


# =========================
# FACT BUILDING
# =========================

def build_fact(file_path: str, info: Dict, idx: int, summary: str):
    return {
        "id": f"auto_{idx}",
        "type": "code",
        "file": file_path,
        "tags": infer_tags(file_path),
        "summary": summary,
        "functions": info["functions"][:15],
        "classes": info["classes"][:15],
        "confidence": 0.9 if is_important(file_path) else 0.7
    }


# =========================
# REPO SCAN
# =========================

def scan_repo():
    facts = []
    cache = load_cache()
    existing_summaries = []
    seen_signatures = set()
    idx = 0

    for root, dirs, files in os.walk(REPO_PATH):
        if should_skip_root(root):
            continue

        for file in files:
            full_path = os.path.join(root, file)

            if file.endswith(".py"):
                info = extract_python_info(full_path)
                if not info:
                    continue

                code = info["code"]

                if should_skip_file(full_path, code):
                    continue

                if is_low_value_file(full_path, info):
                    continue

                summary = summarize_file(full_path, code, info, cache)

                if is_duplicate(summary, existing_summaries):
                    continue

                sig = fact_signature(full_path, info, summary)
                if sig in seen_signatures:
                    continue

                seen_signatures.add(sig)

                fact = build_fact(full_path, info, idx, summary)
                facts.append(fact)
                existing_summaries.append(summary)
                idx += 1

            elif file.endswith(".md"):
                text = safe_read_file(full_path)
                if not text or should_skip_file(full_path, text):
                    continue

                if not is_important(full_path):
                    continue

                file_hash = compute_hash(text)
                cache_key = f"{SUMMARY_VERSION}:{full_path}:{file_hash}"

                if cache_key in cache:
                    summary = cache[cache_key]
                else:
                    summary = (
                        f"This documentation file likely contains instructions, usage notes, "
                        f"or design details related to {guess_file_purpose(full_path)}."
                    )
                    cache[cache_key] = summary

                if is_duplicate(summary, existing_summaries):
                    continue

                facts.append({
                    "id": f"auto_{idx}",
                    "type": "doc",
                    "file": full_path,
                    "tags": infer_tags(full_path),
                    "summary": summary,
                    "confidence": 0.75
                })
                existing_summaries.append(summary)
                idx += 1

    save_cache(cache)
    return facts


# =========================
# SAVE
# =========================

def save_facts(facts):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)

    print(f"✅ Generated {len(facts)} facts")
    print(f"📁 Saved to {OUTPUT_PATH}")
    print(f"💾 Cache saved to {CACHE_PATH}")


if __name__ == "__main__":
    facts = scan_repo()
    save_facts(facts)