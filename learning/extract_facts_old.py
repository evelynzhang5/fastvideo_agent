#!/usr/bin/env python3
"""
Structured fact extractor for the FastVideo onboarding agent.

This script replaces the old file-level summarizer with a source-grounded,
symbol-level memory generator. It creates richer facts for RAG:

- module-level facts
- class-level facts
- top-level function facts
- method-level facts
- documentation/config/example facts

Each fact includes file path, symbol, signature, line numbers, source excerpt,
tags, responsibilities, likely questions, imports, confidence, and retrieval_text.

Usage:
    python learning/extract_facts.py \
        --repo /Users/evelynzhang/Documents/GitHub/fastvideo_agent/FastVideo

Optional:
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# =============================================================================
# Defaults
# =============================================================================

SUMMARY_VERSION = "v4_symbol_evidence"

DEFAULT_OUTPUT_PATH = Path("memory/extracted_facts/auto_facts.json")
DEFAULT_CACHE_PATH = Path("memory/cache/summaries.json")
DEFAULT_MANIFEST_PATH = Path("memory/extracted_facts/manifest.json")

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "build",
    "dist",
    "site-packages",
    ".idea",
    ".vscode",
}

CODE_EXTENSIONS = {".py"}
DOC_EXTENSIONS = {".md", ".rst", ".txt"}
CONFIG_EXTENSIONS = {".toml", ".yaml", ".yml", ".json"}

# These are not used to exclude everything else. They only boost importance.
PRIORITY_PATH_HINTS = [
    "fastvideo/pipelines",
    "fastvideo/models",
    "fastvideo/training",
    "fastvideo/train",
    "fastvideo/attention",
    "fastvideo/dataset",
    "fastvideo/datasets",
    "fastvideo/entrypoints",
    "fastvideo/worker",
    "fastvideo/workflows",
    "fastvideo/envs",
    "fastvideo/utils",
    "fastvideo_kernel",
    "fastvideo-kernel",
    "comfyui/video_generator",
    "examples",
    "scripts",
    "docs",
    "tests",
    "benchmarks",
]

LOW_VALUE_FILENAME_PATTERNS = {
    ".DS_Store",
    "Thumbs.db",
}

MAX_EXCERPT_LINES = 90
MAX_EXCERPT_CHARS = 6000
MAX_DOC_CHUNK_LINES = 90
MAX_DOC_FACTS_PER_FILE = 8
MIN_TEXT_LENGTH = 40


# =============================================================================
# Data models
# =============================================================================

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
    summary: str
    tags: List[str]
    confidence: float
    symbol: Optional[str] = None
    signature: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    docstring: Optional[str] = None
    responsibilities: List[str] = field(default_factory=list)
    answers_questions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    source_excerpt: str = ""
    retrieval_text: str = ""
    content_hash: str = ""
    summary_version: str = SUMMARY_VERSION


# =============================================================================
# Basic helpers
# =============================================================================

def safe_read_file(path: Path) -> Optional[str]:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def normalize_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def rel_path(path: Path, repo_root: Path) -> str:
    try:
        return normalize_path(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return normalize_path(path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n... [truncated]"


def split_lines_with_numbers(text: str) -> List[Tuple[int, str]]:
    return list(enumerate(text.splitlines(), start=1))


# =============================================================================
# Repository scanning and filtering
# =============================================================================

def should_skip_dir(dirname: str) -> bool:
    return dirname in SKIP_DIRS or dirname.startswith(".") and dirname not in {".github"}


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

        root_norm = normalize_path(root_path.relative_to(repo_root)) if root_path != repo_root else ""
        root_norm_lower = root_norm.lower()

        if not include_tests and ("tests" in root_norm_lower.split("/")):
            continue
        if not include_examples and ("examples" in root_norm_lower.split("/")):
            continue
        if not include_docs and ("docs" in root_norm_lower.split("/")):
            continue

        for filename in files:
            if filename in LOW_VALUE_FILENAME_PATTERNS:
                continue

            path = root_path / filename
            suffix = path.suffix.lower()

            if suffix in CODE_EXTENSIONS:
                yield path
            elif include_docs and suffix in DOC_EXTENSIONS:
                yield path
            elif include_configs and suffix in CONFIG_EXTENSIONS:
                yield path


def path_importance_score(rel: str) -> float:
    lower = rel.lower()
    score = 0.0

    for hint in PRIORITY_PATH_HINTS:
        if hint.lower() in lower:
            score += 0.15

    important_words = [
        "pipeline", "stage", "model", "train", "trainer", "callback",
        "optimizer", "scheduler", "worker", "attention", "attn", "denoising",
        "encoding", "decoding", "encoder", "decoder", "dataset", "dataloader",
        "vae", "inference", "generate", "generation", "config", "entrypoint",
        "checkpoint", "distributed", "comfyui", "kernel",
    ]
    for word in important_words:
        if word in lower:
            score += 0.04

    if lower.endswith("__init__.py"):
        score -= 0.20

    return max(0.0, min(1.0, score))


def should_skip_text_file(path: Path, text: Optional[str]) -> bool:
    if text is None:
        return True
    if len(text.strip()) < MIN_TEXT_LENGTH:
        return True
    return False


# =============================================================================
# AST helpers
# =============================================================================

def parse_python(code: str, path: Path) -> Optional[ast.Module]:
    try:
        return ast.parse(code, filename=str(path))
    except SyntaxError:
        return None


def extract_imports(tree: ast.AST) -> List[str]:
    imports: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if module:
                    imports.append(f"{module}.{alias.name}")
                else:
                    imports.append(alias.name)

    return sorted(set(imports))


def get_decorator_names(node: ast.AST) -> List[str]:
    decorators = []
    for dec in getattr(node, "decorator_list", []):
        decorators.append(ast_expr_to_name(dec))
    return [d for d in decorators if d]


def ast_expr_to_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = ast_expr_to_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return ast_expr_to_name(node.func)
    if isinstance(node, ast.Subscript):
        return ast_expr_to_name(node.value)
    if isinstance(node, ast.Constant):
        return repr(node.value)
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


def get_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    parts: List[str] = []

    posonly = getattr(node.args, "posonlyargs", [])
    regular = node.args.args
    defaults = list(node.args.defaults)
    default_offset = len(regular) - len(defaults)

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

    for i, arg in enumerate(posonly):
        parts.append(fmt_arg(arg))
    if posonly:
        parts.append("/")

    for i, arg in enumerate(regular):
        default_node = None
        if i >= default_offset:
            default_node = defaults[i - default_offset]
        parts.append(fmt_arg(arg, default_node))

    if node.args.vararg:
        vararg = "*" + fmt_arg(node.args.vararg)
        parts.append(vararg)
    elif node.args.kwonlyargs:
        parts.append("*")

    for arg, default_node in zip(node.args.kwonlyargs, node.args.kw_defaults):
        parts.append(fmt_arg(arg, default_node))

    if node.args.kwarg:
        parts.append("**" + fmt_arg(node.args.kwarg))

    returns = annotation_to_str(node.returns)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    sig = f"{prefix} {node.name}({', '.join(parts)})"
    if returns:
        sig += f" -> {returns}"
    return sig


def get_class_signature(node: ast.ClassDef) -> str:
    bases = []
    for base in node.bases:
        name = ast_expr_to_name(base)
        if name:
            bases.append(name)
    if bases:
        return f"class {node.name}({', '.join(bases)})"
    return f"class {node.name}"


def get_source_excerpt(code: str, node: ast.AST, max_lines: int = MAX_EXCERPT_LINES) -> Tuple[str, int, int]:
    lines = code.splitlines()
    start = int(getattr(node, "lineno", 1) or 1)
    end = int(getattr(node, "end_lineno", start) or start)
    end = min(end, start + max_lines - 1)
    excerpt = "\n".join(lines[start - 1 : end])
    excerpt = truncate_text(excerpt, MAX_EXCERPT_CHARS)
    return excerpt, start, end


def module_top_excerpt(code: str, max_lines: int = 80) -> Tuple[str, int, int]:
    lines = code.splitlines()
    end = min(len(lines), max_lines)
    excerpt = "\n".join(lines[:end])
    return truncate_text(excerpt, MAX_EXCERPT_CHARS), 1, end


def direct_child_functions(class_node: ast.ClassDef) -> List[str]:
    names = []
    for child in class_node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(child.name)
    return names


def direct_child_classes(tree: ast.Module) -> List[str]:
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


def direct_child_top_functions(tree: ast.Module) -> List[str]:
    return [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def extract_symbols_from_python(code: str, tree: ast.Module) -> List[SymbolInfo]:
    symbols: List[SymbolInfo] = []

    # Module symbol
    module_doc = ast.get_docstring(tree)
    module_excerpt, module_start, module_end = module_top_excerpt(code)
    symbols.append(
        SymbolInfo(
            kind="module",
            symbol=None,
            signature=None,
            docstring=module_doc,
            source_excerpt=module_excerpt,
            line_start=module_start,
            line_end=module_end,
            child_symbols=direct_child_classes(tree) + direct_child_top_functions(tree),
        )
    )

    # Top-level classes and their direct methods
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            excerpt, start, end = get_source_excerpt(code, node)
            class_name = node.name
            symbols.append(
                SymbolInfo(
                    kind="class",
                    symbol=class_name,
                    signature=get_class_signature(node),
                    docstring=ast.get_docstring(node),
                    source_excerpt=excerpt,
                    line_start=start,
                    line_end=end,
                    decorators=get_decorator_names(node),
                    child_symbols=direct_child_functions(node),
                )
            )

            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    excerpt, start, end = get_source_excerpt(code, child)
                    method_name = f"{class_name}.{child.name}"
                    symbols.append(
                        SymbolInfo(
                            kind="method",
                            symbol=method_name,
                            signature=get_function_signature(child),
                            docstring=ast.get_docstring(child),
                            source_excerpt=excerpt,
                            line_start=start,
                            line_end=end,
                            decorators=get_decorator_names(child),
                        )
                    )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            excerpt, start, end = get_source_excerpt(code, node)
            symbols.append(
                SymbolInfo(
                    kind="function",
                    symbol=node.name,
                    signature=get_function_signature(node),
                    docstring=ast.get_docstring(node),
                    source_excerpt=excerpt,
                    line_start=start,
                    line_end=end,
                    decorators=get_decorator_names(node),
                )
            )

    return symbols


# =============================================================================
# Tagging and concept inference
# =============================================================================

TAG_RULES: Dict[str, Sequence[str]] = {
    "training": [
        "train", "trainer", "training", "optimizer", "scheduler", "loss", "backward",
        "gradient", "checkpoint", "epoch", "step", "validation",
    ],
    "inference": [
        "inference", "infer", "generate", "generation", "pipeline", "sample", "sampling",
        "prompt", "negative_prompt", "num_inference_steps",
    ],
    "pipeline": ["pipeline", "pipelines", "stage", "stages", "workflow", "orchestrat"],
    "model": [
        "model", "models", "dit", "transformer", "forward", "module", "nn.module",
        "pretrained", "from_pretrained", "state_dict",
    ],
    "attention": [
        "attention", "attn", "flash_attn", "flashattention", "block_sparse", "sparse_attn",
        "qkv", "query", "key", "value", "triton",
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


def infer_tags(
    rel: str,
    code_or_text: str = "",
    imports: Optional[List[str]] = None,
    symbol: Optional[str] = None,
    signature: Optional[str] = None,
    decorators: Optional[List[str]] = None,
) -> List[str]:
    haystack = "\n".join(
        [
            rel.lower(),
            (symbol or "").lower(),
            (signature or "").lower(),
            " ".join(imports or []).lower(),
            " ".join(decorators or []).lower(),
            code_or_text[:12000].lower(),
        ]
    )

    tags: List[str] = []
    for tag, keywords in TAG_RULES.items():
        if any(k in haystack for k in keywords):
            tags.append(tag)

    # Path-specific backup rules.
    rel_lower = rel.lower()
    if "/tests/" in f"/{rel_lower}" or rel_lower.startswith("tests/"):
        if "testing" not in tags:
            tags.append("testing")
    if "/examples/" in f"/{rel_lower}" or rel_lower.startswith("examples/"):
        if "example" not in tags:
            tags.append("example")
    if "/docs/" in f"/{rel_lower}" or rel_lower.startswith("docs/"):
        if "docs" not in tags:
            tags.append("docs")

    return tags


def infer_file_purpose(rel: str, tags: List[str]) -> str:
    lower = rel.lower()

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
    if "collect_env" in lower:
        return "collects runtime environment and dependency diagnostics"

    return "implements repository functionality or utility behavior"


def infer_responsibilities(
    rel: str,
    kind: str,
    symbol: Optional[str],
    code: str,
    tags: List[str],
) -> List[str]:
    text = "\n".join([rel, kind, symbol or "", code[:12000]]).lower()
    responsibilities: List[str] = []

    def add(condition: bool, value: str) -> None:
        if condition and value not in responsibilities:
            responsibilities.append(value)

    add("config" in tags, "defines or transfers configuration values used by runtime components")
    add("comfyui" in tags, "exposes node metadata, input validation, or UI-facing execution hooks")
    add("inference" in tags, "runs or supports inference-time generation")
    add("training" in tags, "supports training, optimization, validation, or checkpointing")
    add("pipeline" in tags, "organizes execution flow across pipeline stages or workflows")
    add("attention" in tags, "implements attention computation or attention-kernel support")
    add("dataset" in tags, "loads, transforms, batches, or samples data")
    add("distributed" in tags, "handles multi-process, worker, rank, or device-aware execution")
    add("checkpointing" in tags, "saves, loads, or resumes model/runtime state")
    add("vae" in tags, "encodes or decodes latent/video representations")
    add("model" in tags, "defines model structure, model loading, or forward-pass behavior")
    add("testing" in tags, "checks expected behavior and serves as executable usage documentation")
    add("example" in tags, "shows a concrete usage pattern for users or developers")
    add("docs" in tags, "explains setup, usage, design decisions, or troubleshooting steps")

    add("load" in text or "from_pretrained" in text, "loads resources such as models, inputs, outputs, or checkpoints")
    add("save" in text or "write" in text, "writes files, generated outputs, logs, or checkpoints")
    add("validate" in text or "input_types" in text, "validates inputs or declares accepted parameters")
    add("forward" in text, "defines forward-pass or tensor transformation behavior")
    add("backward" in text, "defines backward-pass or gradient behavior")
    add("argparse" in text or "parse_args" in text, "provides a command-line interface or parses CLI arguments")
    add("logger" in text or "logging" in text, "reports runtime status, warnings, or diagnostics")
    add("exception" in text or "raise" in text, "handles error cases or interruption behavior")

    if not responsibilities:
        purpose = infer_file_purpose(rel, tags)
        responsibilities.append(purpose)

    return responsibilities[:7]


def infer_questions(
    rel: str,
    kind: str,
    symbol: Optional[str],
    tags: List[str],
    responsibilities: List[str],
) -> List[str]:
    display = symbol or rel
    questions: List[str] = []

    def add(q: str) -> None:
        if q not in questions:
            questions.append(q)

    add(f"What does {display} do?")
    add(f"Where is {display} implemented?")

    if kind == "module":
        add(f"What is the purpose of {rel}?")
    elif kind in {"class", "function", "method"}:
        add(f"How is {display} used?")

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

    return questions[:8]


def make_summary(
    rel: str,
    kind: str,
    symbol: Optional[str],
    signature: Optional[str],
    docstring: Optional[str],
    tags: List[str],
    responsibilities: List[str],
    child_symbols: Optional[List[str]] = None,
) -> str:
    name = symbol or rel
    purpose = infer_file_purpose(rel, tags)

    parts: List[str] = []
    if kind == "module":
        parts.append(f"{rel} {purpose}.")
        if child_symbols:
            parts.append(f"It exposes key symbols such as {', '.join(child_symbols[:10])}.")
    else:
        parts.append(f"{name} is a {kind} in {rel}.")
        if signature:
            parts.append(f"Signature: {signature}.")
        if responsibilities:
            parts.append("It " + "; ".join(responsibilities[:3]) + ".")

    if docstring:
        doc = compact_whitespace(docstring)
        if doc:
            parts.append(f"Docstring: {truncate_text(doc, 260)}")

    if tags:
        parts.append(f"Tags: {', '.join(tags)}.")

    return " ".join(parts)


def compute_confidence(
    kind: str,
    tags: List[str],
    docstring: Optional[str],
    signature: Optional[str],
    source_excerpt: str,
    rel: str,
) -> float:
    score = 0.40

    if kind in {"module", "class", "function", "method"}:
        score += 0.12
    if signature:
        score += 0.10
    if docstring:
        score += 0.10
    if source_excerpt and len(source_excerpt) > 80:
        score += 0.10
    if tags:
        score += min(0.12, 0.03 * len(tags))
    score += min(0.10, path_importance_score(rel) * 0.10)

    if rel.endswith("__init__.py"):
        score -= 0.08

    return round(max(0.20, min(0.95, score)), 2)


def build_retrieval_text(fact: MemoryFact) -> str:
    fields = [
        fact.summary,
        f"file: {fact.file}",
        f"symbol: {fact.symbol or ''}",
        f"signature: {fact.signature or ''}",
        "tags: " + ", ".join(fact.tags),
        "responsibilities: " + "; ".join(fact.responsibilities),
        "questions: " + "; ".join(fact.answers_questions),
        "imports: " + ", ".join(fact.imports[:20]),
        "source: " + fact.source_excerpt[:1800],
    ]
    return "\n".join([f for f in fields if f and f.strip()])


def make_fact_id(rel: str, kind: str, symbol: Optional[str], line_start: Optional[int]) -> str:
    raw = f"{rel}:{kind}:{symbol or '<module>'}:{line_start or 0}"
    return "auto_" + short_hash(raw, 14)


# =============================================================================
# Python fact extraction
# =============================================================================

def extract_python_facts(path: Path, repo_root: Path, cache: Dict[str, Any]) -> List[MemoryFact]:
    code = safe_read_file(path)
    if should_skip_text_file(path, code):
        return []

    assert code is not None
    rel = rel_path(path, repo_root)
    content_hash = compute_hash(code)
    cache_key = f"{SUMMARY_VERSION}:py:{rel}:{content_hash}"

    if cache_key in cache:
        cached = cache[cache_key]
        try:
            return [MemoryFact(**item) for item in cached]
        except Exception:
            pass

    tree = parse_python(code, path)
    if tree is None:
        return []

    imports = extract_imports(tree)
    symbols = extract_symbols_from_python(code, tree)
    module_classes = direct_child_classes(tree)
    module_functions = direct_child_top_functions(tree)

    facts: List[MemoryFact] = []

    for sym in symbols:
        # Avoid too many weak dunder method facts. Keep __init__, but skip common noise.
        if sym.kind == "method" and sym.symbol:
            method_leaf = sym.symbol.split(".")[-1]
            if method_leaf.startswith("__") and method_leaf.endswith("__") and method_leaf != "__init__":
                continue

        tags = infer_tags(
            rel,
            code_or_text=sym.source_excerpt,
            imports=imports,
            symbol=sym.symbol,
            signature=sym.signature,
            decorators=sym.decorators,
        )

        # Skip truly low-value module facts, but keep symbol-level facts.
        if sym.kind == "module" and not tags and not sym.docstring and path.name == "__init__.py":
            continue

        responsibilities = infer_responsibilities(
            rel=rel,
            kind=sym.kind,
            symbol=sym.symbol,
            code=sym.source_excerpt,
            tags=tags,
        )
        questions = infer_questions(rel, sym.kind, sym.symbol, tags, responsibilities)
        summary = make_summary(
            rel=rel,
            kind=sym.kind,
            symbol=sym.symbol,
            signature=sym.signature,
            docstring=sym.docstring,
            tags=tags,
            responsibilities=responsibilities,
            child_symbols=sym.child_symbols,
        )
        confidence = compute_confidence(
            kind=sym.kind,
            tags=tags,
            docstring=sym.docstring,
            signature=sym.signature,
            source_excerpt=sym.source_excerpt,
            rel=rel,
        )

        fact = MemoryFact(
            id=make_fact_id(rel, sym.kind, sym.symbol, sym.line_start),
            type=sym.kind,
            file=rel,
            symbol=sym.symbol,
            signature=sym.signature,
            line_start=sym.line_start,
            line_end=sym.line_end,
            docstring=truncate_text(compact_whitespace(sym.docstring or ""), 1200) or None,
            responsibilities=responsibilities,
            answers_questions=questions,
            imports=imports[:50],
            decorators=sym.decorators,
            classes=module_classes[:50],
            functions=module_functions[:50],
            tags=tags,
            summary=summary,
            source_excerpt=sym.source_excerpt,
            confidence=confidence,
            content_hash=content_hash,
        )
        fact.retrieval_text = build_retrieval_text(fact)
        facts.append(fact)

    cache[cache_key] = [asdict(f) for f in facts]
    return facts


# =============================================================================
# Documentation/config fact extraction
# =============================================================================

def split_markdown_chunks(text: str, max_lines: int = MAX_DOC_CHUNK_LINES) -> List[Tuple[str, int, int, Optional[str]]]:
    lines = text.splitlines()
    chunks: List[Tuple[str, int, int, Optional[str]]] = []

    current_start = 1
    current_heading: Optional[str] = None
    buffer: List[str] = []

    def flush(end_line: int) -> None:
        nonlocal buffer, current_start, current_heading
        chunk_text = "\n".join(buffer).strip()
        if len(chunk_text) >= MIN_TEXT_LENGTH:
            chunks.append((truncate_text(chunk_text, MAX_EXCERPT_CHARS), current_start, end_line, current_heading))
        buffer = []

    for idx, line in enumerate(lines, start=1):
        is_heading = bool(re.match(r"^#{1,6}\s+", line.strip()))

        if is_heading and buffer:
            flush(idx - 1)
            current_start = idx
            current_heading = line.strip("# ").strip()

        if not buffer:
            current_start = idx
            if is_heading:
                current_heading = line.strip("# ").strip()

        buffer.append(line)

        if len(buffer) >= max_lines:
            flush(idx)
            current_start = idx + 1
            current_heading = None

    if buffer:
        flush(len(lines))

    return chunks[:MAX_DOC_FACTS_PER_FILE]


def summarize_doc_chunk(rel: str, heading: Optional[str], text: str, tags: List[str]) -> str:
    title = f" section '{heading}'" if heading else ""
    purpose = infer_file_purpose(rel, tags)
    first_sentence = compact_whitespace(text).split(". ")[0]
    first_sentence = truncate_text(first_sentence, 260)
    return f"{rel}{title} {purpose}. Key content: {first_sentence}. Tags: {', '.join(tags) or 'general'}."


def extract_doc_or_config_facts(path: Path, repo_root: Path, cache: Dict[str, Any]) -> List[MemoryFact]:
    text = safe_read_file(path)
    if should_skip_text_file(path, text):
        return []

    assert text is not None
    rel = rel_path(path, repo_root)
    suffix = path.suffix.lower()
    content_hash = compute_hash(text)
    cache_key = f"{SUMMARY_VERSION}:doc:{rel}:{content_hash}"

    if cache_key in cache:
        cached = cache[cache_key]
        try:
            return [MemoryFact(**item) for item in cached]
        except Exception:
            pass

    facts: List[MemoryFact] = []

    if suffix in DOC_EXTENSIONS:
        chunks = split_markdown_chunks(text)
    else:
        lines = text.splitlines()
        end = min(len(lines), MAX_DOC_CHUNK_LINES)
        chunks = [(truncate_text("\n".join(lines[:end]), MAX_EXCERPT_CHARS), 1, end, None)]

    for i, (chunk_text, start, end, heading) in enumerate(chunks):
        kind = "doc" if suffix in DOC_EXTENSIONS else "config"
        symbol = heading if heading else None
        tags = infer_tags(rel, code_or_text=chunk_text, imports=[], symbol=symbol)
        if suffix in CONFIG_EXTENSIONS and "config" not in tags:
            tags.append("config")
        if suffix in DOC_EXTENSIONS and "docs" not in tags:
            tags.append("docs")

        responsibilities = infer_responsibilities(rel, kind, symbol, chunk_text, tags)
        questions = infer_questions(rel, kind, symbol, tags, responsibilities)
        summary = summarize_doc_chunk(rel, heading, chunk_text, tags)
        confidence = compute_confidence(kind, tags, None, None, chunk_text, rel)

        fact = MemoryFact(
            id=make_fact_id(rel, kind, symbol or f"chunk_{i}", start),
            type=kind,
            file=rel,
            symbol=symbol,
            signature=None,
            line_start=start,
            line_end=end,
            docstring=None,
            responsibilities=responsibilities,
            answers_questions=questions,
            imports=[],
            decorators=[],
            classes=[],
            functions=[],
            tags=tags,
            summary=summary,
            source_excerpt=chunk_text,
            confidence=confidence,
            content_hash=content_hash,
        )
        fact.retrieval_text = build_retrieval_text(fact)
        facts.append(fact)

    cache[cache_key] = [asdict(f) for f in facts]
    return facts


# =============================================================================
# Deduplication and output
# =============================================================================

def fact_dedupe_key(fact: MemoryFact) -> str:
    return ":".join(
        [
            fact.file,
            fact.type,
            fact.symbol or "",
            str(fact.line_start or 0),
            short_hash(fact.summary, 8),
        ]
    )


def dedupe_facts(facts: List[MemoryFact]) -> List[MemoryFact]:
    seen = set()
    output: List[MemoryFact] = []

    for fact in facts:
        key = fact_dedupe_key(fact)
        if key in seen:
            continue
        seen.add(key)
        output.append(fact)

    return output


def sort_facts(facts: List[MemoryFact]) -> List[MemoryFact]:
    type_order = {
        "module": 0,
        "class": 1,
        "function": 2,
        "method": 3,
        "doc": 4,
        "config": 5,
    }
    return sorted(facts, key=lambda f: (f.file, type_order.get(f.type, 9), f.line_start or 0, f.symbol or ""))


def save_facts(facts: List[MemoryFact], output_path: Path) -> None:
    write_json(output_path, [asdict(fact) for fact in facts])


def save_manifest(
    manifest_path: Path,
    repo_root: Path,
    facts: List[MemoryFact],
    scanned_files: int,
    skipped_files: int,
    cache_hits_possible: int,
) -> None:
    by_type: Dict[str, int] = {}
    by_tag: Dict[str, int] = {}

    for fact in facts:
        by_type[fact.type] = by_type.get(fact.type, 0) + 1
        for tag in fact.tags:
            by_tag[tag] = by_tag.get(tag, 0) + 1

    manifest = {
        "summary_version": SUMMARY_VERSION,
        "repo_root": normalize_path(repo_root.resolve()),
        "num_facts": len(facts),
        "num_scanned_files": scanned_files,
        "num_skipped_files": skipped_files,
        "cache_entries_checked": cache_hits_possible,
        "facts_by_type": dict(sorted(by_type.items())),
        "facts_by_tag": dict(sorted(by_tag.items())),
    }
    write_json(manifest_path, manifest)


# =============================================================================
# Main scan
# =============================================================================

def scan_repo(
    repo_root: Path,
    output_path: Path,
    cache_path: Path,
    manifest_path: Path,
    include_tests: bool,
    include_examples: bool,
    include_docs: bool,
    include_configs: bool,
    min_confidence: float,
) -> List[MemoryFact]:
    repo_root = repo_root.expanduser().resolve()
    cache: Dict[str, Any] = load_json(cache_path, {})

    facts: List[MemoryFact] = []
    scanned_files = 0
    skipped_files = 0

    for path in iter_repo_files(
        repo_root,
        include_tests=include_tests,
        include_examples=include_examples,
        include_docs=include_docs,
        include_configs=include_configs,
    ):
        scanned_files += 1
        suffix = path.suffix.lower()

        try:
            if suffix in CODE_EXTENSIONS:
                file_facts = extract_python_facts(path, repo_root, cache)
            elif suffix in DOC_EXTENSIONS or suffix in CONFIG_EXTENSIONS:
                file_facts = extract_doc_or_config_facts(path, repo_root, cache)
            else:
                file_facts = []
        except Exception as exc:
            skipped_files += 1
            print(f"⚠️  Skipped {path}: {exc}")
            continue

        if not file_facts:
            skipped_files += 1
            continue

        facts.extend(file_facts)

    facts = dedupe_facts(facts)
    facts = [f for f in facts if f.confidence >= min_confidence]
    facts = sort_facts(facts)

    save_facts(facts, output_path)
    write_json(cache_path, cache)
    save_manifest(
        manifest_path=manifest_path,
        repo_root=repo_root,
        facts=facts,
        scanned_files=scanned_files,
        skipped_files=skipped_files,
        cache_hits_possible=len(cache),
    )

    return facts


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build structured source-grounded facts for the FastVideo onboarding agent."
    )

    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("FASTVIDEO_REPO", "")) if os.environ.get("FASTVIDEO_REPO") else None,
        help="Path to the FastVideo repository. Can also be set with FASTVIDEO_REPO.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Cache JSON path. Default: {DEFAULT_CACHE_PATH}",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Manifest JSON path. Default: {DEFAULT_MANIFEST_PATH}",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Drop facts below this confidence. Default keeps all facts.",
    )

    parser.add_argument("--exclude-tests", action="store_true", help="Do not index tests.")
    parser.add_argument("--exclude-examples", action="store_true", help="Do not index examples.")
    parser.add_argument("--exclude-docs", action="store_true", help="Do not index docs/Markdown/RST/TXT.")
    parser.add_argument("--exclude-configs", action="store_true", help="Do not index TOML/YAML/JSON config files.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.repo is None:
        raise SystemExit(
            "❌ Missing repo path. Use --repo /path/to/FastVideo or set FASTVIDEO_REPO=/path/to/FastVideo"
        )

    repo_root = args.repo.expanduser().resolve()
    if not repo_root.exists():
        raise SystemExit(f"❌ Repo path does not exist: {repo_root}")
    if not repo_root.is_dir():
        raise SystemExit(f"❌ Repo path is not a directory: {repo_root}")

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

    # Small preview so you can quickly see whether extraction is useful.
    print("\nPreview:")
    for fact in facts[:5]:
        loc = f":{fact.line_start}-{fact.line_end}" if fact.line_start else ""
        symbol = f" :: {fact.symbol}" if fact.symbol else ""
        print(f"- [{fact.type}] {fact.file}{loc}{symbol}")
        print(f"  {fact.summary[:220]}{'...' if len(fact.summary) > 220 else ''}")


if __name__ == "__main__":
    main()
