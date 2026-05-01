#!/usr/bin/env python3
"""
Drop-in replacement for learning/extract_facts.py.

Goal:
- Replace shallow file-level facts with source-grounded, symbol-level facts.
- Keep the same default output path: memory/extracted_facts/auto_facts.json
- Keep the same cache path: memory/cache/summaries.json
- Allow repo path via --repo or FASTVIDEO_REPO.

Usage:
    python learning/extract_facts.py --repo /path/to/FastVideo

or:
    FASTVIDEO_REPO=/path/to/FastVideo python learning/extract_facts.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =========================
# CONFIG
# =========================

SUMMARY_VERSION = "v4_1_precise_tags"
OUTPUT_PATH = Path("memory/extracted_facts/auto_facts.json")
CACHE_PATH = Path("memory/cache/summaries.json")
MANIFEST_PATH = Path("memory/extracted_facts/manifest.json")

CODE_EXTENSIONS = {".py"}
DOC_EXTENSIONS = {".md", ".rst", ".txt"}
CONFIG_EXTENSIONS = {".toml", ".yaml", ".yml", ".json"}

SKIP_DIRS = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".venv", "venv", "env", "node_modules", "build", "dist",
    "site-packages", ".idea", ".vscode",
}

MAX_EXCERPT_LINES = 90
MAX_EXCERPT_CHARS = 6000
MAX_DOC_CHUNKS_PER_FILE = 8
MAX_DOC_CHUNK_LINES = 90
MIN_TEXT_LENGTH = 40

# Do not use this to exclude everything else. It only boosts confidence.
PRIORITY_PATH_HINTS = [
    "fastvideo/pipelines", "fastvideo/models", "fastvideo/training", "fastvideo/train",
    "fastvideo/attention", "fastvideo/dataset", "fastvideo/datasets", "fastvideo/entrypoints",
    "fastvideo/worker", "fastvideo/workflows", "fastvideo/envs", "fastvideo/utils",
    "fastvideo_kernel", "fastvideo-kernel", "comfyui/video_generator",
    "examples", "scripts", "docs", "tests", "benchmarks",
]

TAG_RULES: Dict[str, List[str]] = {
    "training": [
        "train", "trainer", "training", "optimizer", "scheduler", "loss",
        "backward", "gradient", "epoch", "validation",
    ],
    "inference": [
        "inference", "infer", "generate", "generation", "sample", "sampling",
        "prompt", "negative_prompt", "num_inference_steps",
    ],
    "pipeline": ["pipeline", "pipelines", "stage", "stages", "workflow", "orchestrat"],
    "model": [
        "model", "models", "dit", "transformer", "forward", "nn.module",
        "pretrained", "from_pretrained", "state_dict",
    ],
    "attention": [
        "attention", "attn", "flash_attn", "flashattention", "block_sparse",
        "sparse_attn", "qkv", "query", "key", "value", "triton",
    ],
    "vae": ["vae", "latent", "latents", "encode", "decode", "decoder", "encoder"],
    "dataset": ["dataset", "datasets", "dataloader", "collate", "__getitem__", "sampler"],
    "distributed": [
        "distributed", "torch.distributed", "accelerate", "deepspeed", "world_size",
        "rank", "local_rank", "worker", "process_group", "barrier",
    ],
    "checkpointing": [
        "checkpoint", "resume", "save_pretrained", "load_state_dict", "state_dict",
        "safetensors", "ckpt", "save_checkpoint",
    ],
    "config": [
        "config", "configs", "args", "arguments", "argumentparser", "dataclass",
        "input_types", "validate_inputs", "set_args", "yaml", "toml",
    ],
    "comfyui": ["comfyui", "input_types", "return_types", "node", "category", "function"],
    "entrypoint": ["entrypoint", "entrypoints", "main", "cli", "command", "argparse"],
    "kernel": ["kernel", "triton", "cuda", "sm90", "block_sparse", "torch.library"],
    "logging": ["logger", "logging", "log", "warning", "debug", "info"],
    "testing": ["test", "pytest", "assert", "fixture", "unittest"],
    "docs": ["readme", "installation", "quickstart", "usage", "tutorial", "guide"],
    "platform": ["env", "environment", "cuda", "gpu", "cpu", "version", "collect_env"],
    "video": ["video", "frames", "fps", "height", "width", "decode", "encode"],
}


# =========================
# DATA MODELS
# =========================

@dataclass
class SymbolInfo:
    kind: str
    symbol: Optional[str]
    signature: Optional[str]
    docstring: Optional[str]
    source_excerpt: str
    line_start: int
    line_end: int
    decorators: List[str] = field(default_factory=list)
    child_symbols: List[str] = field(default_factory=list)


@dataclass
class MemoryFact:
    id: str
    type: str
    file: str
    tags: List[str]
    summary: str
    confidence: float

    # Rich metadata for better RAG.
    symbol: Optional[str] = None
    signature: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    docstring: Optional[str] = None
    responsibilities: List[str] = field(default_factory=list)
    answers_questions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    source_excerpt: str = ""
    retrieval_text: str = ""
    content_hash: str = ""
    summary_version: str = SUMMARY_VERSION


# =========================
# FILE HELPERS
# =========================

def safe_read_file(path: Path) -> Optional[str]:
    for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def short_hash(text: str, n: int = 14) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def normalize_path(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def rel_path(path: Path, repo_root: Path) -> str:
    try:
        return normalize_path(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return normalize_path(path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n... [truncated]"


# =========================
# REPO SCAN
# =========================

def should_skip_dir(dirname: str) -> bool:
    return dirname in SKIP_DIRS or (dirname.startswith(".") and dirname != ".github")


def iter_repo_files(
    repo_root: Path,
    include_tests: bool = True,
    include_examples: bool = True,
    include_docs: bool = True,
    include_configs: bool = True,
) -> Iterable[Path]:
    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]

        root_rel = normalize_path(root_path.relative_to(repo_root)) if root_path != repo_root else ""
        root_parts = set(root_rel.lower().split("/"))

        if not include_tests and "tests" in root_parts:
            continue
        if not include_examples and "examples" in root_parts:
            continue
        if not include_docs and "docs" in root_parts:
            continue

        for filename in files:
            path = root_path / filename
            suffix = path.suffix.lower()
            if suffix in CODE_EXTENSIONS:
                yield path
            elif include_docs and suffix in DOC_EXTENSIONS:
                yield path
            elif include_configs and suffix in CONFIG_EXTENSIONS:
                yield path


def should_skip_text(text: Optional[str]) -> bool:
    return text is None or len(text.strip()) < MIN_TEXT_LENGTH


def path_importance_score(rel: str) -> float:
    lower = rel.lower()
    score = 0.0
    for hint in PRIORITY_PATH_HINTS:
        if hint in lower:
            score += 0.15
    for word in [
        "pipeline", "stage", "model", "train", "trainer", "callback", "optimizer",
        "scheduler", "worker", "attention", "attn", "denoising", "encoding",
        "decoding", "encoder", "decoder", "dataset", "dataloader", "vae",
        "inference", "generate", "generation", "config", "entrypoint", "checkpoint",
        "distributed", "comfyui", "kernel",
    ]:
        if word in lower:
            score += 0.04
    if lower.endswith("__init__.py"):
        score -= 0.2
    return max(0.0, min(1.0, score))


# =========================
# AST EXTRACTION
# =========================

def parse_python(code: str, path: Path) -> Optional[ast.Module]:
    try:
        return ast.parse(code, filename=str(path))
    except SyntaxError:
        return None


def ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = ast_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return ast_name(node.func)
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def annotation_to_str(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def decorators(node: ast.AST) -> List[str]:
    vals = []
    for dec in getattr(node, "decorator_list", []):
        name = ast_name(dec)
        if name:
            vals.append(name)
    return vals


def extract_imports(tree: ast.AST) -> List[str]:
    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
    return sorted(set(imports))


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    parts: List[str] = []
    args = node.args

    defaults = list(args.defaults)
    default_offset = len(args.args) - len(defaults)

    def fmt_arg(arg: ast.arg, default_node: Optional[ast.AST] = None) -> str:
        s = arg.arg
        ann = annotation_to_str(arg.annotation)
        if ann:
            s += f": {ann}"
        if default_node is not None:
            try:
                s += f" = {ast.unparse(default_node)}"
            except Exception:
                s += " = ..."
        return s

    for arg in getattr(args, "posonlyargs", []):
        parts.append(fmt_arg(arg))
    if getattr(args, "posonlyargs", []):
        parts.append("/")

    for i, arg in enumerate(args.args):
        default_node = defaults[i - default_offset] if i >= default_offset else None
        parts.append(fmt_arg(arg, default_node))

    if args.vararg:
        parts.append("*" + fmt_arg(args.vararg))
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(fmt_arg(arg, default_node))

    if args.kwarg:
        parts.append("**" + fmt_arg(args.kwarg))

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    sig = f"{prefix} {node.name}({', '.join(parts)})"
    ret = annotation_to_str(node.returns)
    if ret:
        sig += f" -> {ret}"
    return sig


def class_signature(node: ast.ClassDef) -> str:
    bases = [ast_name(base) for base in node.bases if ast_name(base)]
    if bases:
        return f"class {node.name}({', '.join(bases)})"
    return f"class {node.name}"


def source_excerpt(code: str, node: ast.AST, max_lines: int = MAX_EXCERPT_LINES) -> Tuple[str, int, int]:
    lines = code.splitlines()
    start = int(getattr(node, "lineno", 1) or 1)
    end = int(getattr(node, "end_lineno", start) or start)
    end = min(end, start + max_lines - 1)
    excerpt = "\n".join(lines[start - 1 : end])
    return truncate(excerpt, MAX_EXCERPT_CHARS), start, end


def module_excerpt(code: str, max_lines: int = 80) -> Tuple[str, int, int]:
    lines = code.splitlines()
    end = min(len(lines), max_lines)
    return truncate("\n".join(lines[:end]), MAX_EXCERPT_CHARS), 1, end


def top_classes(tree: ast.Module) -> List[str]:
    return [n.name for n in tree.body if isinstance(n, ast.ClassDef)]


def top_functions(tree: ast.Module) -> List[str]:
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def class_methods(node: ast.ClassDef) -> List[str]:
    return [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def extract_symbols(code: str, tree: ast.Module) -> List[SymbolInfo]:
    symbols: List[SymbolInfo] = []
    m_excerpt, m_start, m_end = module_excerpt(code)
    symbols.append(
        SymbolInfo(
            kind="module",
            symbol=None,
            signature=None,
            docstring=ast.get_docstring(tree),
            source_excerpt=m_excerpt,
            line_start=m_start,
            line_end=m_end,
            child_symbols=top_classes(tree) + top_functions(tree),
        )
    )

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            excerpt, start, end = source_excerpt(code, node)
            symbols.append(
                SymbolInfo(
                    kind="class",
                    symbol=node.name,
                    signature=class_signature(node),
                    docstring=ast.get_docstring(node),
                    source_excerpt=excerpt,
                    line_start=start,
                    line_end=end,
                    decorators=decorators(node),
                    child_symbols=class_methods(node),
                )
            )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_leaf = child.name
                    if method_leaf.startswith("__") and method_leaf.endswith("__") and method_leaf != "__init__":
                        continue
                    excerpt, start, end = source_excerpt(code, child)
                    symbols.append(
                        SymbolInfo(
                            kind="method",
                            symbol=f"{node.name}.{child.name}",
                            signature=function_signature(child),
                            docstring=ast.get_docstring(child),
                            source_excerpt=excerpt,
                            line_start=start,
                            line_end=end,
                            decorators=decorators(child),
                        )
                    )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            excerpt, start, end = source_excerpt(code, node)
            symbols.append(
                SymbolInfo(
                    kind="function",
                    symbol=node.name,
                    signature=function_signature(node),
                    docstring=ast.get_docstring(node),
                    source_excerpt=excerpt,
                    line_start=start,
                    line_end=end,
                    decorators=decorators(node),
                )
            )

    return symbols


# =========================
# FACT INFERENCE
# =========================


def path_tokens(rel: str) -> set[str]:
    """Tokenize a path without treating 'fastvideo' as the token 'video'."""
    rel = rel.lower()
    parts = re.split(r"[/._\-]+", rel)
    return {p for p in parts if p}


def word_tokens(text: str) -> set[str]:
    """Tokenize code/text into coarse words for exact-ish matching."""
    return {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or "")}


def has_any_word(tokens: set[str], words: List[str]) -> bool:
    return any(w in tokens for w in words)


def has_any_substring(text: str, needles: List[str]) -> bool:
    return any(n in text for n in needles)


def is_test_path(rel: str) -> bool:
    toks = path_tokens(rel)
    return "tests" in toks or "test" in toks


def is_doc_path(rel: str) -> bool:
    toks = path_tokens(rel)
    return "docs" in toks or rel.lower().endswith((".md", ".rst", ".txt"))


def is_example_path(rel: str) -> bool:
    return "examples" in path_tokens(rel)


def infer_tags(
    rel: str,
    text: str = "",
    imports: Optional[List[str]] = None,
    symbol: Optional[str] = None,
    signature: Optional[str] = None,
    decs: Optional[List[str]] = None,
) -> List[str]:
    """Infer tags conservatively.

    v4 was useful but over-tagged short methods because every method inherited
    module imports and substring matches. For example, a method in fastvideo/*
    got the 'video' tag because 'fastvideo' contains 'video'; methods in
    platform files inherited 'attention' from AttentionBackendEnum imports.

    This version separates local evidence from file-level evidence:
      - path/file name gives broad subsystem tags like platform/comfyui/kernel
      - local symbol/signature/source gives method-specific tags
      - imports are only used strongly for module/class facts, or when local
        source also points at that concept
    """
    rel_lower = rel.lower()
    toks = path_tokens(rel)
    local_text = "\n".join([
        symbol or "",
        signature or "",
        " ".join(decs or []),
        text[:12000],
    ]).lower()
    local_tokens = word_tokens(local_text)
    import_text = " ".join(imports or []).lower()
    tags: List[str] = []

    def add(tag: str) -> None:
        if tag not in tags:
            tags.append(tag)

    # Path-level subsystem tags. These are safe because they come from directory/file names.
    if "platforms" in toks or "platform" in toks or rel_lower.endswith("platform.py"):
        add("platform")
    if "comfyui" in toks:
        add("comfyui")
    if "kernel" in toks or "fastvideo_kernel" in rel_lower or "fastvideo-kernel" in rel_lower:
        add("kernel")
    if "attention" in toks or "attn" in toks:
        add("attention")
    if "pipelines" in toks or "pipeline" in toks or "stages" in toks or "stage" in toks or "workflows" in toks:
        add("pipeline")
    if "train" in toks or "training" in toks or "trainer" in toks:
        add("training")
    if "dataset" in toks or "datasets" in toks or "dataloader" in toks:
        add("dataset")
    if "models" in toks or "model" in toks or "dit" in toks:
        add("model")
    if "vae" in toks:
        add("vae")
    if "configs" in toks or "config" in toks or rel_lower.endswith((".toml", ".yaml", ".yml", ".json")):
        add("config")
    if is_test_path(rel):
        add("testing")
    if is_doc_path(rel):
        add("docs")
    if is_example_path(rel):
        add("example")

    # Local evidence tags. Use exact tokens where possible to avoid fastvideo -> video.
    if has_any_word(local_tokens, ["train", "trainer", "training", "optimizer", "scheduler", "loss", "backward", "gradient", "epoch", "validation"]):
        add("training")
    if has_any_word(local_tokens, ["inference", "infer", "generate", "generation", "sample", "sampling", "prompt"]):
        add("inference")
    if has_any_word(local_tokens, ["pipeline", "pipelines", "stage", "stages", "workflow"]):
        add("pipeline")
    if has_any_word(local_tokens, ["model", "models", "dit", "transformer", "forward", "pretrained", "state_dict"]):
        add("model")
    if has_any_word(local_tokens, ["attention", "attn", "flash_attn", "flashattention", "block_sparse", "qkv", "query", "key", "value", "triton"]):
        add("attention")
    if has_any_word(local_tokens, ["vae", "latent", "latents", "encode", "decode", "decoder", "encoder"]):
        add("vae")
    if has_any_word(local_tokens, ["dataset", "datasets", "dataloader", "collate", "sampler", "__getitem__"]):
        add("dataset")
    if has_any_substring(local_text, ["torch.distributed", "processgroup", "prefixstore", "world_size", "local_rank", "process_group"]):
        add("distributed")
    if has_any_word(local_tokens, ["checkpoint", "resume", "safetensors", "ckpt"]) or has_any_substring(local_text, ["save_pretrained", "load_state_dict", "save_checkpoint"]):
        add("checkpointing")
    if has_any_word(local_tokens, ["config", "args", "arguments", "argumentparser", "dataclass"]) or has_any_substring(local_text, ["input_types", "validate_inputs", "set_args"]):
        add("config")
    if "logger" in local_tokens or "logging" in local_tokens or "warning" in local_tokens:
        add("logging")
    if has_any_word(local_tokens, ["cuda", "gpu", "cpu", "npu", "rocm", "device", "environment", "version"]):
        add("platform")
    if has_any_word(local_tokens, ["video", "frames", "fps", "height", "width"]):
        add("video")

    # Import-level tags: only for module/class facts, or when local text also uses related names.
    # This prevents tiny methods from inheriting every module-wide concern.
    broad_import_scope = symbol is None or (symbol is not None and "." not in symbol)
    if broad_import_scope:
        if has_any_substring(import_text, ["torch.distributed", "processgroup", "accelerate", "deepspeed"]):
            add("distributed")
        if has_any_substring(import_text, ["attentionbackend", "fastvideo.attention", "flash_attn"]):
            add("attention")
        if has_any_substring(import_text, ["logger", "logging"]):
            add("logging")

    # Strict testing/docs. Do not tag production code as testing just because it has assert.
    if not is_test_path(rel) and "testing" in tags:
        tags.remove("testing")
    if not is_doc_path(rel) and "docs" in tags:
        # Keep docs only for real doc files/dirs, not because docstring contains usage/guide words.
        tags.remove("docs")

    return tags

def file_purpose(rel: str, tags: List[str]) -> str:
    if "comfyui" in tags:
        return "connects FastVideo functionality to ComfyUI node inputs, validation, and execution"
    if "pipeline" in tags:
        return "defines pipeline or stage orchestration logic"
    if "training" in tags:
        return "implements training, optimization, validation, or checkpoint logic"
    if "inference" in tags:
        return "implements inference-time generation or sampling behavior"
    if "attention" in tags:
        return "implements attention-related tensor operations or kernels"
    if "dataset" in tags:
        return "handles dataset loading, preprocessing, or batching"
    if "config" in tags:
        return "defines configuration, arguments, or validation behavior"
    if "model" in tags:
        return "implements model architecture or model loading behavior"
    if "docs" in tags:
        return "documents setup, usage, design, or troubleshooting information"
    if "testing" in tags:
        return "tests expected behavior or documents usage through assertions"
    if "example" in tags:
        return "demonstrates how to use part of the repository"
    if "collect_env" in rel.lower():
        return "collects runtime environment and dependency diagnostics"
    return "implements repository functionality or utility behavior"



def responsibilities(rel: str, kind: str, symbol: Optional[str], code: str, tags: List[str]) -> List[str]:
    """Infer responsibilities with local-source priority.

    Tags are useful retrieval labels, but responsibilities should describe what
    this symbol actually does. For short methods, use the method name and source
    before broad module-level labels.
    """
    text = "\n".join([rel, kind, symbol or "", code[:12000]]).lower()
    sym = (symbol or "").lower()
    out: List[str] = []

    def add(condition: bool, value: str) -> None:
        if condition and value not in out:
            out.append(value)

    # Name/source-specific responsibilities first.
    add("mem_get_info" in sym or "memory" in sym or "mem_" in sym,
        "queries device memory information or memory usage")
    add("clear" in sym and "memory" in sym,
        "clears device memory caches and resets memory statistics")
    add("get_torch_device" in sym,
        "returns the torch device module used by this platform")
    add("get_device_name" in sym,
        "returns the device name for this platform")
    add("get_device_total_memory" in sym,
        "returns total memory for a selected device")
    add("get_device_capability" in sym,
        "returns hardware/device capability information")
    add("communicator" in sym,
        "returns the distributed device communicator class for this platform")
    add("attn_backend" in sym or "attentionbackend" in text or "fastvideo_attention_backend" in text,
        "selects or reports the attention backend supported by this platform")
    add("processgroup" in text or "prefixstore" in text or "_register_backend" in text,
        "initializes or registers a distributed process group backend")
    add("input_types" in text or "validate_inputs" in text,
        "declares or validates UI/configuration inputs")
    add("set_args" in sym or "set_args" in text,
        "copies configuration values into runtime arguments")
    add("load" in sym or "from_pretrained" in text,
        "loads resources such as models, inputs, outputs, or checkpoints")
    add("save" in sym or "write" in text,
        "writes files, generated outputs, logs, or checkpoints")
    add("forward" in sym,
        "defines forward-pass or tensor transformation behavior")
    add("backward" in sym,
        "defines backward-pass or gradient behavior")
    add("logger" in text or "logging" in text or "logger." in text,
        "reports runtime status, warnings, or diagnostics")
    add("raise " in text or "exception" in text,
        "handles error cases or unsupported configurations")

    # Broader fallback only if we don't already have enough local evidence.
    if len(out) < 2:
        add("config" in tags, "defines or transfers configuration values used by runtime components")
        add("comfyui" in tags, "exposes ComfyUI node metadata, input validation, or UI-facing execution hooks")
        add("inference" in tags, "runs or supports inference-time generation")
        add("training" in tags, "supports training, optimization, validation, or checkpointing")
        add("pipeline" in tags, "organizes execution flow across pipeline stages or workflows")
        add("attention" in tags and ("attention" in text or "attn" in text or "backend" in text),
            "implements or selects attention computation support")
        add("dataset" in tags, "loads, transforms, batches, or samples data")
        add("distributed" in tags and ("distributed" in text or "processgroup" in text or "communicator" in text or "rank" in text),
            "handles distributed, worker, rank, or device-aware execution")
        add("checkpointing" in tags, "saves, loads, or resumes model/runtime state")
        add("vae" in tags, "encodes or decodes latent/video representations")
        add("model" in tags, "defines model structure, model loading, or forward-pass behavior")
        add("testing" in tags, "checks expected behavior and serves as executable usage documentation")
        add("example" in tags, "shows a concrete usage pattern for users or developers")
        add("docs" in tags, "explains setup, usage, design decisions, or troubleshooting steps")
        add("platform" in tags, "abstracts device/platform-specific runtime behavior")

    if not out:
        out.append(file_purpose(rel, tags))
    return out[:6]

def likely_questions(rel: str, kind: str, symbol: Optional[str], tags: List[str]) -> List[str]:
    name = symbol or rel
    qs: List[str] = []

    def add(q: str) -> None:
        if q not in qs:
            qs.append(q)

    add(f"What does {name} do?")
    add(f"Where is {name} implemented?")
    if kind == "module":
        add(f"What is the purpose of {rel}?")
    else:
        add(f"How is {name} used?")

    if "training" in tags:
        add("How does FastVideo handle training or optimization?")
    if "inference" in tags:
        add("How does FastVideo run inference or generation?")
    if "pipeline" in tags:
        add("How is the FastVideo pipeline organized?")
    if "attention" in tags:
        add("Where is attention implemented?")
    if "dataset" in tags:
        add("How does FastVideo load or batch data?")
    if "config" in tags:
        add("Where are configuration arguments defined or validated?")
    if "comfyui" in tags:
        add("How does the ComfyUI integration connect to FastVideo?")
    if "distributed" in tags:
        add("How does FastVideo support distributed or worker execution?")
    if "checkpointing" in tags:
        add("Where are checkpoints loaded, saved, or resumed?")
    if "testing" in tags:
        add("What behavior is covered by tests?")
    if "example" in tags:
        add("What is a minimal usage example?")
    if "docs" in tags:
        add("Where are setup or usage instructions documented?")

    return qs[:8]


def make_summary(
    rel: str,
    kind: str,
    symbol: Optional[str],
    signature: Optional[str],
    docstring: Optional[str],
    tags: List[str],
    resp: List[str],
    child_symbols: Optional[List[str]] = None,
) -> str:
    name = symbol or rel
    parts = []
    if kind == "module":
        parts.append(f"{rel} {file_purpose(rel, tags)}.")
        if child_symbols:
            parts.append(f"It exposes key symbols such as {', '.join(child_symbols[:10])}.")
    else:
        parts.append(f"{name} is a {kind} in {rel}.")
        if signature:
            parts.append(f"Signature: {signature}.")
        if resp:
            parts.append("It " + "; ".join(resp[:3]) + ".")

    if docstring:
        doc = compact(docstring)
        if doc:
            parts.append(f"Docstring: {truncate(doc, 260)}")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}.")
    return " ".join(parts)


def confidence(kind: str, tags: List[str], docstring: Optional[str], signature: Optional[str], excerpt: str, rel: str) -> float:
    score = 0.40
    if kind in {"module", "class", "function", "method", "doc", "config"}:
        score += 0.12
    if signature:
        score += 0.10
    if docstring:
        score += 0.10
    if excerpt and len(excerpt) > 80:
        score += 0.10
    if tags:
        score += min(0.12, 0.03 * len(tags))
    score += min(0.10, path_importance_score(rel) * 0.10)
    if rel.endswith("__init__.py"):
        score -= 0.08
    return round(max(0.20, min(0.95, score)), 2)


def make_fact_id(rel: str, kind: str, symbol: Optional[str], line_start: Optional[int]) -> str:
    return "auto_" + short_hash(f"{rel}:{kind}:{symbol or '<module>'}:{line_start or 0}")


def build_retrieval_text(fact: MemoryFact) -> str:
    return "\n".join([
        fact.summary,
        f"file: {fact.file}",
        f"symbol: {fact.symbol or ''}",
        f"signature: {fact.signature or ''}",
        "tags: " + ", ".join(fact.tags),
        "responsibilities: " + "; ".join(fact.responsibilities),
        "questions: " + "; ".join(fact.answers_questions),
        "imports: " + ", ".join(fact.imports[:20]),
        "source: " + fact.source_excerpt[:1800],
    ])

def build_curated_facts() -> List[MemoryFact]:
    """
    Manually curated high-value facts that should always be included in auto_facts.json.

    These are not extracted from source code automatically, but they are important
    onboarding/review knowledge for the agent.
    """

    checklist = """Checklist for New Model Support

DiT:
- Is DistributedAttention and LocalAttention imported and used?
- Does your DiT support sequence parallelism with optional padding?
- Do not import any models, layers, config, etc. from the transformers library or diffusers library.
- Raise a flag if diffusers or any other model-specific Python package is imported and used within the DiT modeling file.
- If the DiT or text encoder contains a sub-model directly imported from transformers/diffusers, copy and paste the source code of that sub-model into your model file.
- Use as many layers from fastvideo/layers as possible, including but not limited to ReplicatedLinear, RoPE, fused layernorm, and scale shift.
- CFG is usually used for non-distilled checkpoints but not used for step-distilled checkpoints. Raise a flag if CFG completely does not exist.
- Inherit the transformer model class from BaseDiT or CachableDiT.

Text Encoders:
- Should use FastVideo’s linear layers: ReplicatedLinear, ColumnParallelLinear, or RowParallelLinear for attention projections.
- Should use LocalAttention, but not DistributedAttention.

Pipelines:
- Reuse as many stages from fastvideo/pipelines/stages as possible.
- If no stages can be used and it is difficult to modify the existing stages, create a dedicated stage for the new model.

Configs:
- There should be a model-specific file and dataclass definition under configs/pipelines, configs/sample, and configs/models.
- configs/sample is used for generation-time parameters such as resolution, num_frames, num_inference_steps, and cfg_scale. These default values should match the reference repository’s default values.
- configs/pipelines is used for initialization parameters such as timestep shift, dtype, offloading flags, and parallelism.
- configs/models is used for model parameters for DiTs, VAEs, etc. These values should match the config.json under the corresponding model folder.
"""

    fact = MemoryFact(
        id="curated_new_model_support_checklist",
        type="curated_checklist",
        file="curated/new_model_support_checklist.md",
        symbol="NewModelSupportChecklist",
        tags=[
            "new_model_support",
            "model",
            "dit",
            "text_encoder",
            "pipeline",
            "config",
            "attention",
            "review_checklist",
        ],
        summary=(
            "Checklist for adding or reviewing support for a new model in FastVideo. "
            "Covers DiT requirements, text encoder requirements, pipeline reuse, and required config files."
        ),
        confidence=0.97,
        responsibilities=[
            "guides new model integration reviews",
            "checks whether DiT models use FastVideo attention and layer abstractions",
            "checks whether text encoders use FastVideo parallel linear layers and LocalAttention",
            "checks whether pipelines reuse existing stages or define dedicated model-specific stages",
            "checks whether configs exist under configs/pipelines, configs/sample, and configs/models",
        ],
        answers_questions=[
            "How do you add a new model?",
            "What should I check when adding a new DiT model?",
            "What are the requirements for new text encoders?",
            "Should text encoders use DistributedAttention?",
            "How should pipelines be added for a new model?",
            "What config files are needed for a new model?",
            "What should be reviewed for new model support?",
        ],
        source_excerpt=checklist,
        content_hash=compute_hash(checklist),
    )

    fact.retrieval_text = "\n".join([
        fact.summary,
        "file: " + fact.file,
        "symbol: " + str(fact.symbol or ""),
        "tags: " + ", ".join(fact.tags),
        "responsibilities: " + "; ".join(fact.responsibilities),
        "questions: " + "; ".join(fact.answers_questions),
        "source: " + checklist,
    ])

    return [fact]
# =========================
# PYTHON FACTS
# =========================

def extract_python_facts(path: Path, repo_root: Path, cache: Dict[str, Any]) -> List[MemoryFact]:
    code = safe_read_file(path)
    if should_skip_text(code):
        return []
    assert code is not None

    rel = rel_path(path, repo_root)
    content_hash = compute_hash(code)
    cache_key = f"{SUMMARY_VERSION}:py:{rel}:{content_hash}"

    if cache_key in cache:
        try:
            return [MemoryFact(**item) for item in cache[cache_key]]
        except Exception:
            pass

    tree = parse_python(code, path)
    if tree is None:
        return []

    imports = extract_imports(tree)
    module_classes = top_classes(tree)
    module_functions = top_functions(tree)
    symbols = extract_symbols(code, tree)
    facts: List[MemoryFact] = []

    for sym in symbols:
        tags = infer_tags(
            rel,
            text=sym.source_excerpt,
            imports=imports,
            symbol=sym.symbol,
            signature=sym.signature,
            decs=sym.decorators,
        )

        if sym.kind == "module" and not tags and not sym.docstring and rel.endswith("__init__.py"):
            continue

        resp = responsibilities(rel, sym.kind, sym.symbol, sym.source_excerpt, tags)
        qs = likely_questions(rel, sym.kind, sym.symbol, tags)
        summary = make_summary(
            rel=rel,
            kind=sym.kind,
            symbol=sym.symbol,
            signature=sym.signature,
            docstring=sym.docstring,
            tags=tags,
            resp=resp,
            child_symbols=sym.child_symbols,
        )

        fact = MemoryFact(
            id=make_fact_id(rel, sym.kind, sym.symbol, sym.line_start),
            type=sym.kind,
            file=rel,
            symbol=sym.symbol,
            signature=sym.signature,
            line_start=sym.line_start,
            line_end=sym.line_end,
            docstring=truncate(compact(sym.docstring or ""), 1200) or None,
            responsibilities=resp,
            answers_questions=qs,
            imports=imports[:50],
            decorators=sym.decorators,
            functions=module_functions[:50],
            classes=module_classes[:50],
            source_excerpt=sym.source_excerpt,
            tags=tags,
            summary=summary,
            confidence=confidence(sym.kind, tags, sym.docstring, sym.signature, sym.source_excerpt, rel),
            content_hash=content_hash,
        )
        fact.retrieval_text = build_retrieval_text(fact)
        facts.append(fact)

    cache[cache_key] = [asdict(f) for f in facts]
    return facts


# =========================
# DOC / CONFIG FACTS
# =========================

def split_text_chunks(text: str, max_lines: int = MAX_DOC_CHUNK_LINES) -> List[Tuple[str, int, int, Optional[str]]]:
    lines = text.splitlines()
    chunks: List[Tuple[str, int, int, Optional[str]]] = []
    buf: List[str] = []
    start = 1
    heading: Optional[str] = None

    def flush(end: int) -> None:
        nonlocal buf, start, heading
        chunk = "\n".join(buf).strip()
        if len(chunk) >= MIN_TEXT_LENGTH:
            chunks.append((truncate(chunk, MAX_EXCERPT_CHARS), start, end, heading))
        buf = []

    for i, line in enumerate(lines, start=1):
        is_heading = bool(re.match(r"^#{1,6}\s+", line.strip()))
        if is_heading and buf:
            flush(i - 1)
            start = i
            heading = line.strip("# ").strip()
        if not buf:
            start = i
            if is_heading:
                heading = line.strip("# ").strip()
        buf.append(line)
        if len(buf) >= max_lines:
            flush(i)
            start = i + 1
            heading = None
    if buf:
        flush(len(lines))

    return chunks[:MAX_DOC_CHUNKS_PER_FILE]


def extract_doc_config_facts(path: Path, repo_root: Path, cache: Dict[str, Any]) -> List[MemoryFact]:
    text = safe_read_file(path)
    if should_skip_text(text):
        return []
    assert text is not None

    rel = rel_path(path, repo_root)
    suffix = path.suffix.lower()
    content_hash = compute_hash(text)
    cache_key = f"{SUMMARY_VERSION}:doc:{rel}:{content_hash}"

    if cache_key in cache:
        try:
            return [MemoryFact(**item) for item in cache[cache_key]]
        except Exception:
            pass

    if suffix in DOC_EXTENSIONS:
        chunks = split_text_chunks(text)
        kind = "doc"
    else:
        lines = text.splitlines()
        end = min(len(lines), MAX_DOC_CHUNK_LINES)
        chunks = [(truncate("\n".join(lines[:end]), MAX_EXCERPT_CHARS), 1, end, None)]
        kind = "config"

    facts: List[MemoryFact] = []

    for i, (chunk, start, end, heading) in enumerate(chunks):
        tags = infer_tags(rel, text=chunk, symbol=heading)
        if kind == "doc" and "docs" not in tags:
            tags.append("docs")
        if kind == "config" and "config" not in tags:
            tags.append("config")

        resp = responsibilities(rel, kind, heading, chunk, tags)
        qs = likely_questions(rel, kind, heading, tags)
        first = truncate(compact(chunk).split(". ")[0], 260)
        title = f" section '{heading}'" if heading else ""
        summary = f"{rel}{title} {file_purpose(rel, tags)}. Key content: {first}. Tags: {', '.join(tags)}."

        fact = MemoryFact(
            id=make_fact_id(rel, kind, heading or f"chunk_{i}", start),
            type=kind,
            file=rel,
            symbol=heading,
            line_start=start,
            line_end=end,
            responsibilities=resp,
            answers_questions=qs,
            tags=tags,
            summary=summary,
            source_excerpt=chunk,
            confidence=confidence(kind, tags, None, None, chunk, rel),
            content_hash=content_hash,
        )
        fact.retrieval_text = build_retrieval_text(fact)
        facts.append(fact)

    cache[cache_key] = [asdict(f) for f in facts]
    return facts


# =========================
# OUTPUT
# =========================

def fact_key(f: MemoryFact) -> str:
    return f"{f.file}:{f.type}:{f.symbol or ''}:{f.line_start or 0}:{short_hash(f.summary, 8)}"


def dedupe_facts(facts: List[MemoryFact]) -> List[MemoryFact]:
    seen = set()
    out = []
    for f in facts:
        key = fact_key(f)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def sort_facts(facts: List[MemoryFact]) -> List[MemoryFact]:
    order = {"module": 0, "class": 1, "function": 2, "method": 3, "doc": 4, "config": 5}
    return sorted(facts, key=lambda f: (f.file, order.get(f.type, 9), f.line_start or 0, f.symbol or ""))


def save_manifest(path: Path, repo_root: Path, facts: List[MemoryFact], scanned: int, skipped: int) -> None:
    by_type: Dict[str, int] = {}
    by_tag: Dict[str, int] = {}
    for f in facts:
        by_type[f.type] = by_type.get(f.type, 0) + 1
        for t in f.tags:
            by_tag[t] = by_tag.get(t, 0) + 1

    manifest = {
        "summary_version": SUMMARY_VERSION,
        "repo_root": normalize_path(repo_root.resolve()),
        "num_facts": len(facts),
        "num_scanned_files": scanned,
        "num_skipped_files": skipped,
        "facts_by_type": dict(sorted(by_type.items())),
        "facts_by_tag": dict(sorted(by_tag.items())),
    }
    write_json(path, manifest)


def scan_repo(
    repo_root: Path,
    output_path: Path = OUTPUT_PATH,
    cache_path: Path = CACHE_PATH,
    manifest_path: Path = MANIFEST_PATH,
    include_tests: bool = True,
    include_examples: bool = True,
    include_docs: bool = True,
    include_configs: bool = True,
    min_confidence: float = 0.0,
) -> List[MemoryFact]:
    repo_root = repo_root.expanduser().resolve()
    cache: Dict[str, Any] = load_json(cache_path, {})

    facts: List[MemoryFact] = []
    scanned = 0
    skipped = 0

    for path in iter_repo_files(
        repo_root,
        include_tests=include_tests,
        include_examples=include_examples,
        include_docs=include_docs,
        include_configs=include_configs,
    ):
        scanned += 1
        suffix = path.suffix.lower()
        try:
            if suffix in CODE_EXTENSIONS:
                new_facts = extract_python_facts(path, repo_root, cache)
            elif suffix in DOC_EXTENSIONS or suffix in CONFIG_EXTENSIONS:
                new_facts = extract_doc_config_facts(path, repo_root, cache)
            else:
                new_facts = []
        except Exception as exc:
            print(f"⚠️  Skipped {path}: {exc}")
            skipped += 1
            continue

        if not new_facts:
            skipped += 1
        facts.extend(new_facts)

    # Add manually curated high-value facts into the same auto_facts.json.
    facts.extend(build_curated_facts())

    facts = dedupe_facts(facts)
    facts = [f for f in facts if f.confidence >= min_confidence]
    facts = sort_facts(facts)
    write_json(output_path, [asdict(f) for f in facts])
    write_json(cache_path, cache)
    save_manifest(manifest_path, repo_root, facts, scanned, skipped)
    return facts


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured facts from the FastVideo repository.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ["FASTVIDEO_REPO"]) if os.environ.get("FASTVIDEO_REPO") else None,
        help="Path to FastVideo repo. You can also set FASTVIDEO_REPO=/path/to/FastVideo.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--exclude-tests", action="store_true")
    parser.add_argument("--exclude-examples", action="store_true")
    parser.add_argument("--exclude-docs", action="store_true")
    parser.add_argument("--exclude-configs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repo is None:
        raise SystemExit("❌ Missing repo path. Use --repo /path/to/FastVideo or set FASTVIDEO_REPO.")

    repo_root = args.repo.expanduser().resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise SystemExit(f"❌ Invalid repo path: {repo_root}")

    facts = scan_repo(
        repo_root=repo_root,
        output_path=args.output,
        cache_path=args.cache,
        manifest_path=args.manifest,
        include_tests=not args.exclude_tests,
        include_examples=not args.exclude_examples,
        include_docs=not args.exclude_docs,
        include_configs=not args.exclude_configs,
        min_confidence=args.min_confidence,
    )

    print(f"✅ Generated {len(facts)} structured facts")
    print(f"📁 Saved facts to {args.output}")
    print(f"💾 Saved cache to {args.cache}")
    print(f"🧭 Saved manifest to {args.manifest}")
    print("\nPreview:")
    for f in facts[:8]:
        loc = f":{f.line_start}-{f.line_end}" if f.line_start else ""
        sym = f" :: {f.symbol}" if f.symbol else ""
        print(f"- [{f.type}] {f.file}{loc}{sym}")
        print(f"  {f.summary[:220]}{'...' if len(f.summary) > 220 else ''}")


if __name__ == "__main__":
    main()
