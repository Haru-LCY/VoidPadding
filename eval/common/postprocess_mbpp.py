#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import os
import re
import signal
from pathlib import Path
from typing import Any


EXEC_TIMEOUT_SECONDS = int(os.environ.get("MBPP_EXEC_TIMEOUT_SECONDS", "10"))
MAX_EXTRACT_LINES = int(os.environ.get("MBPP_MAX_EXTRACT_LINES", "100"))
SANITIZE_TIMEOUT_SECONDS = float(os.environ.get("MBPP_SANITIZE_TIMEOUT_SECONDS", "0"))
MAX_SYNTAX_CHARS = int(os.environ.get("MBPP_MAX_SYNTAX_CHARS", "0"))
PREFIX_ONLY_EXTRACTION = os.environ.get("MBPP_PREFIX_ONLY_EXTRACTION", "0") == "1"


class SanitizeTimeoutError(TimeoutError):
    pass


def _handle_sanitize_timeout(signum: int, frame: Any) -> None:
    raise SanitizeTimeoutError("sanitize_timeout")


def sanitize_with_timeout(text: str, entrypoint: str | None = None) -> str:
    if SANITIZE_TIMEOUT_SECONDS <= 0:
        return sanitize(text, entrypoint=entrypoint)

    old_handler = signal.signal(signal.SIGALRM, _handle_sanitize_timeout)
    signal.setitimer(signal.ITIMER_REAL, SANITIZE_TIMEOUT_SECONDS)
    try:
        return sanitize(text, entrypoint=entrypoint)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def refine_text(text: str) -> str:
    text = text.replace("\t", "    ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    return f"{text}\n" if text else ""


def syntax_check(code: str) -> bool:
    if MAX_SYNTAX_CHARS > 0 and len(code) > MAX_SYNTAX_CHARS:
        return False
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        return False


def extract_longest_valid_code(text: str) -> str:
    lines = text.splitlines()
    if len(lines) > MAX_EXTRACT_LINES:
        lines = lines[:MAX_EXTRACT_LINES]

    full_text = "\n".join(lines).strip()
    if full_text and syntax_check(full_text):
        return full_text

    if PREFIX_ONLY_EXTRACTION:
        for end in range(len(lines) - 1, 0, -1):
            snippet = "\n".join(lines[:end]).strip()
            if snippet and syntax_check(snippet):
                return snippet
        return ""

    # DAEDAL-style fallback: search valid line spans, preferring the longest
    # non-empty parseable snippet. Iterate longest spans first to avoid most
    # of the O(n^2) worst case on normal completions.
    for span_len in range(len(lines), 0, -1):
        for start in range(0, len(lines) - span_len + 1):
            snippet_lines = lines[start : start + span_len]
            if not any(line.strip() for line in snippet_lines):
                continue
            snippet = "\n".join(snippet_lines).strip()
            if syntax_check(snippet):
                return snippet
    return ""


def get_definition_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def get_deps(nodes: list[tuple[str, ast.AST]]) -> dict[str, set[str]]:
    name_to_deps: dict[str, set[str]] = {}
    for name, node in nodes:
        deps = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in ast.iter_child_nodes(current):
                if isinstance(child, ast.Name):
                    deps.add(child.id)
                elif isinstance(child, ast.Attribute):
                    continue
                else:
                    stack.append(child)
        name_to_deps[name] = deps
    return name_to_deps


def get_definition_dependency(entrypoint: str, call_graph: dict[str, set[str]]) -> set[str]:
    visited = set()
    to_visit = [entrypoint]
    while to_visit:
        current = to_visit.pop(0)
        if current in visited:
            continue
        visited.add(current)
        to_visit.extend(call_graph.get(current, set()) - visited)
    return visited


def sanitize(text: str, entrypoint: str | None = None) -> str:
    text = refine_text(text)
    if not text:
        return ""

    code = extract_longest_valid_code(text)
    if not code:
        return ""

    try:
        tree = ast.parse(code)
    except (SyntaxError, MemoryError):
        return ""

    imports: list[ast.AST] = []
    definitions: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            name = get_definition_name(node)
            if name:
                definitions[name] = node

    if entrypoint and entrypoint in definitions:
        name_to_deps = get_deps([(name, node) for name, node in definitions.items()])
        reachable = get_definition_dependency(entrypoint, name_to_deps)
    else:
        reachable = set(definitions)

    output = [ast.unparse(node) for node in imports]
    output.extend(ast.unparse(node) for name, node in definitions.items() if name in reachable)
    return "\n".join(output)


def _first_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return _first_text(value[0])
    return ""


def get_generation(sample: dict[str, Any]) -> str:
    for key in ("resps", "filtered_resps"):
        if key in sample:
            text = _first_text(sample[key])
            if text:
                return text
    return ""


def strip_completion_wrappers(text: str) -> str:
    matched_wrapper = False
    patterns = [
        r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]",
        r"BEGIN\s*'(.*)'\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*)'\s*DONE",
        r"BEGIN\s*'(.*)'\s*DONE",
        r"\[BEGIN\]\s*'(.*)\s*\[DONE\]",
        r"BEGIN\s*'(.*)\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*)\s*DONE",
        r"BEGIN\s*'(.*)\s*DONE",
        r"\[BEGIN\]\s*(.*)\s*\[DONE\]",
        r"BEGIN\s*(.*)\s*\[DONE\]",
        r"\[BEGIN\]\s*(.*)\s*DONE",
        r"BEGIN\s*(.*)\s*DONE",
        r"\[BEGIN\]\s*'(.*)",
        r"\[BEGIN\](.*)",
        r"'(.*)'\s*\[DONE\]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1)
            matched_wrapper = True
            break

    if re.search(r"\[?DONE\]?", text):
        text = re.split(r"\s*\[?DONE\]?", text, maxsplit=1)[0]

    text = text.replace("\\_", "_").strip()

    # Only strip wrapper quotes when the completion was actually wrapped. Do
    # not remove a trailing quote from direct code such as `return 'No'`.
    if matched_wrapper and text.startswith("'") and text.endswith("'") and len(text) >= 2:
        text = text[1:]
        text = text[:-1]
    return text.strip()


def extract_python_code(text: str) -> str:
    text = text.strip()
    fence_matches = re.findall(r"```(?:python|py)?[^\n`]*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if fence_matches:
        return strip_completion_wrappers(fence_matches[0])

    if text.strip().startswith("```"):
        _, _, text = text.partition("\n")
    if "```" in text:
        text = text.split("```", 1)[0]

    text = strip_completion_wrappers(text)
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("def ", "async def ", "class ", "import ", "from ")):
            return "\n".join(lines[idx:]).strip()
    return text.strip()


def find_entry_point(code_block: str) -> str | None:
    if not code_block:
        return None
    match = re.search(r"^\s*(?:async\s+)?def\s+([a-zA-Z_]\w*)", code_block, re.MULTILINE)
    return match.group(1) if match else None


def entrypoint_from_doc(doc: dict[str, Any]) -> str | None:
    tests = "\n".join(str(item) for item in doc.get("test_list", [])[:3])
    match = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", tests)
    return match.group(1) if match else None


def build_tests(sample: dict[str, Any]) -> str:
    target = sample.get("target")
    if isinstance(target, str) and target.strip():
        return target
    if isinstance(target, list):
        return "\n".join(str(item) for item in target if str(item).strip())

    doc = sample.get("doc") or {}
    parts = []
    setup = doc.get("test_setup_code", "")
    if setup:
        parts.append(str(setup))
    parts.extend(str(item) for item in doc.get("test_list", [])[:3])
    return "\n".join(parts)


def task_id_from_sample(sample: dict[str, Any]) -> Any:
    doc = sample.get("doc") or {}
    return doc.get("task_id", sample.get("task_id"))


def _run_candidate(code: str, tests: str, queue: mp.Queue) -> None:
    namespace: dict[str, Any] = {}
    try:
        exec(f"{code}\n{tests}", namespace)
    except BaseException as exc:
        queue.put((False, repr(exc)))
        return
    queue.put((True, None))


def passes_tests(code: str, tests: str, timeout: int) -> tuple[bool, str | None]:
    if not tests.strip():
        return False, "missing_tests"

    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_run_candidate, args=(code, tests, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return False, "timeout"
    if queue.empty():
        return False, "no_result"
    passed, error = queue.get()
    return bool(passed), error


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_samples(sample_file: Path, output_file: Path, summary_file: Path, limit: int | None, timeout: int) -> dict[str, Any]:
    samples = load_jsonl(sample_file, limit=limit)
    rows = []
    counters = {
        "failed_extraction": 0,
        "failed_syntax": 0,
        "failed_execution": 0,
    }
    raw_chars = 0
    cleaned_chars = 0
    passed = 0

    for sample in samples:
        raw_generation = get_generation(sample)
        raw_chars += len(raw_generation)
        extracted = extract_python_code(raw_generation)
        doc = sample.get("doc") or {}
        entrypoint = entrypoint_from_doc(doc) or find_entry_point(extracted)
        tests = build_tests(sample)
        try:
            cleaned = sanitize_with_timeout(extracted, entrypoint=entrypoint)
        except SanitizeTimeoutError:
            cleaned = ""
            counters["failed_syntax"] += 1
            rows.append(
                {
                    "task_id": task_id_from_sample(sample),
                    "entrypoint": entrypoint,
                    "pass_at_1": 0.0,
                    "reason": "sanitize_timeout",
                    "raw_completion": raw_generation,
                    "extracted_completion": extracted,
                    "cleaned_completion": cleaned,
                }
            )
            continue
        cleaned_chars += len(cleaned)

        reason = None
        pass_value = 0.0
        if not extracted:
            reason = "failed_extraction"
            counters["failed_extraction"] += 1
        elif not cleaned:
            reason = "failed_syntax"
            counters["failed_syntax"] += 1
        else:
            ok, error = passes_tests(cleaned, tests, timeout=timeout)
            if ok:
                pass_value = 1.0
                passed += 1
            else:
                reason = error or "failed_execution"
                counters["failed_execution"] += 1

        rows.append(
            {
                "task_id": task_id_from_sample(sample),
                "entrypoint": entrypoint,
                "pass_at_1": pass_value,
                "reason": reason,
                "raw_completion": raw_generation,
                "extracted_completion": extracted,
                "cleaned_completion": cleaned,
            }
        )

    total = len(samples)
    summary = {
        "input_file": str(sample_file),
        "output_file": str(output_file),
        "total": total,
        "passed": passed,
        "pass_at_1": passed / total if total else 0.0,
        "failed_extraction": counters["failed_extraction"],
        "failed_syntax": counters["failed_syntax"],
        "failed_execution": counters["failed_execution"],
        "avg_raw_chars": raw_chars / total if total else 0.0,
        "avg_cleaned_chars": cleaned_chars / total if total else 0.0,
        "timeout_seconds": timeout,
        "limit": limit,
    }

    write_jsonl(output_file, rows)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAEDAL-style cleaned MBPP pass@1 postprocess.")
    parser.add_argument("sample_file", type=Path, help="lm-eval samples_*.jsonl file")
    parser.add_argument("--output", type=Path, default=None, help="cleaned JSONL output path")
    parser.add_argument("--summary-output", type=Path, default=None, help="summary JSON output path")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N samples")
    parser.add_argument("--timeout", type=int, default=EXEC_TIMEOUT_SECONDS, help="seconds per sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_file = args.sample_file
    output_file = args.output or Path(f"{sample_file}.cleaned")
    summary_file = args.summary_output or Path(f"{sample_file}.cleaned.summary.json")
    summary = evaluate_samples(
        sample_file=sample_file,
        output_file=output_file,
        summary_file=summary_file,
        limit=args.limit,
        timeout=args.timeout,
    )
    print(
        "[mbpp-cleaned] "
        f"total={summary['total']} passed={summary['passed']} "
        f"pass@1={summary['pass_at_1']:.4f} "
        f"failed_extraction={summary['failed_extraction']} "
        f"failed_syntax={summary['failed_syntax']} "
        f"failed_execution={summary['failed_execution']}"
    )
    print(f"[mbpp-cleaned] output={output_file}")
    print(f"[mbpp-cleaned] summary={summary_file}")


if __name__ == "__main__":
    main()
