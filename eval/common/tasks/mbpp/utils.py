import re
import ast
import multiprocessing as mp
import os
from pathlib import Path
from typing import Union

from huggingface_hub import hf_hub_download
from lm_eval.api.task import ConfigurableTask

EXEC_TIMEOUT_SECONDS = 10
MBPP_DATASET = "Muennighoff/mbpp"


def _local_files_only() -> bool:
    return os.environ.get("HF_DATASETS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"


def resolve_mbpp_data_file() -> str:
    override = os.environ.get("MBPP_DATA_FILE")
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"MBPP_DATA_FILE does not exist: {path}")

    return hf_hub_download(
        repo_id=MBPP_DATASET,
        filename="data/mbpp.jsonl",
        repo_type="dataset",
        local_files_only=_local_files_only(),
    )


class MBPPTask(ConfigurableTask):
    def __init__(self, config=None):
        config = dict(config or {})
        config.pop("class", None)
        config["dataset_path"] = "json"
        config["dataset_kwargs"] = {"data_files": {"test": resolve_mbpp_data_file()}}
        super().__init__(config=config)


def _run_candidate(code: str, tests: str, queue: mp.Queue) -> None:
    namespace = {}
    try:
        exec(code + "\n" + tests, namespace)
    except BaseException as exc:
        queue.put((False, repr(exc)))
        return
    queue.put((True, None))


def _passes_tests(code: str, tests: str) -> bool:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_run_candidate, args=(code, tests, queue))
    process.start()
    process.join(EXEC_TIMEOUT_SECONDS)
    if process.is_alive():
        process.terminate()
        process.join()
        return False
    if queue.empty():
        return False
    passed, _ = queue.get()
    return bool(passed)


def pass_at_1(
    references: Union[str, list[str]],
    predictions: Union[str, list[str], list[list[str]]],
) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions, str):
        predictions = [[predictions]]
    elif predictions and isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]

    total = 0
    passed = 0
    for tests, candidates in zip(references, predictions):
        total += 1
        candidate = candidates[0] if candidates else ""
        if _passes_tests(candidate, tests):
            passed += 1
    return passed / total if total else 0.0


def extract_code_blocks(text: str) -> str:
    pattern = r"```[^\n`]*\n?(.*?)\n?```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    text = text.strip()
    if text.startswith("```"):
        _, _, text = text.partition("\n")
    if "```" in text:
        text = text.split("```", 1)[0]

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ", "import ", "from ")):
            return "\n".join(lines[idx:]).strip()
    return text.strip()


def _entrypoint_from_doc(doc: dict) -> str | None:
    tests = "\n".join(doc.get("test_list", [])[:3])
    match = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", tests)
    return match.group(1) if match else None


def _has_return_statement(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Return) for child in ast.walk(node))


def _definition_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def _node_deps(node: ast.AST) -> set[str]:
    deps = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            deps.add(child.id)
        elif isinstance(child, ast.Attribute):
            deps.add(child.attr)
    return deps


def _reachable_defs(entrypoint: str, definitions: dict[str, ast.AST]) -> set[str]:
    reachable = set()
    pending = [entrypoint]
    name_to_deps = {name: _node_deps(node) for name, node in definitions.items()}
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(name_to_deps.get(name, set()) - reachable)
    return reachable


def sanitize_code(text: str, entrypoint: str | None = None) -> str:
    text = text.replace("\t", "    ").replace("\r\n", "\n").replace("\r", "\n").strip()
    try:
        tree = ast.parse(text)
    except (SyntaxError, MemoryError):
        return text

    imports = []
    definitions: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            definitions[node.name] = node
        elif isinstance(node, ast.FunctionDef) and _has_return_statement(node):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            name = _definition_name(node)
            if name:
                definitions[name] = node

    keep = _reachable_defs(entrypoint, definitions) if entrypoint else set(definitions)
    output = [ast.unparse(node) for node in imports]
    output.extend(ast.unparse(node) for name, node in definitions.items() if name in keep)
    return "\n".join(output) if output else text


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    predictions = []
    for resp, doc in zip(resps, docs):
        entrypoint = _entrypoint_from_doc(doc)
        predictions.append([sanitize_code(extract_code_blocks(r), entrypoint) for r in resp])
    return predictions


def _build_tests(doc: dict) -> str:
    parts = []
    setup = doc.get("test_setup_code", "")
    if setup:
        parts.append(setup)
    parts.extend(doc["test_list"][:3])
    return "\n".join(parts)


def doc_to_target(doc: dict) -> str:
    if doc.get("is_fewshot"):
        return doc["code"] + "\n[DONE]"
    return _build_tests(doc)


def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]
