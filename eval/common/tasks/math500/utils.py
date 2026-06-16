import os
import re
from pathlib import Path
from typing import Any, Union

from huggingface_hub import hf_hub_download
from lm_eval.api.task import ConfigurableTask


MATH500_DATASET = "HuggingFaceH4/MATH-500"
DEFAULT_PROMPT_TEMPLATE = """Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{problem}"""


def _local_files_only() -> bool:
    return os.environ.get("HF_DATASETS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"


def resolve_math500_data_file() -> str:
    override = os.environ.get("MATH500_DATA_FILE")
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"MATH500_DATA_FILE does not exist: {path}")

    return hf_hub_download(
        repo_id=MATH500_DATASET,
        filename="test.jsonl",
        repo_type="dataset",
        local_files_only=_local_files_only(),
    )


class Math500Task(ConfigurableTask):
    def __init__(self, config=None):
        config = dict(config or {})
        config.pop("class", None)
        config["dataset_path"] = "json"
        config["dataset_kwargs"] = {"data_files": {"test": resolve_math500_data_file()}}
        super().__init__(config=config)


def doc_to_text(doc: dict[str, Any]) -> str:
    return DEFAULT_PROMPT_TEMPLATE.format(problem=doc["problem"])


def doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc["answer"])


def doc_to_fewshot_target(doc: dict[str, Any]) -> str:
    solution = str(doc.get("solution") or "").strip()
    final_answer = f"Therefore, the final answer is: $\\boxed{{{doc['answer']}}}$. I hope it is correct"
    return f"{solution}\n\n{final_answer}" if solution else final_answer


def _first_prediction(prediction: Union[str, list[str], list[list[str]]]) -> str:
    if isinstance(prediction, str):
        return prediction
    if not prediction:
        return ""
    first = prediction[0]
    if isinstance(first, list):
        return str(first[0]) if first else ""
    return str(first)


def _extract_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    pos = start + len(marker)
    depth = 1
    chars = []
    while pos < len(text):
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(ch)
        pos += 1
    return None


def _normalize_text(text: str) -> str:
    text = _extract_boxed(text) or text
    text = text.strip()
    text = re.sub(r"^Therefore,\s*the\s*final\s*answer\s*is:?\s*", "", text, flags=re.I)
    text = text.strip(" $.\n\t")
    text = text.replace(r"\left", "")
    text = text.replace(r"\right", "")
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace(r"\(", "")
    text = text.replace(r"\)", "")
    text = text.replace("\\,", "")
    text = text.replace("\\!", "")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _math_verify_equal(gold: str, pred: str) -> bool:
    try:
        from math_verify import parse, verify
    except Exception:
        return False

    try:
        parsed_gold = parse(gold)
        parsed_pred = parse(pred)
        return bool(verify(parsed_gold, parsed_pred))
    except Exception:
        return False


def math500_accuracy(
    references: Union[str, list[str]],
    predictions: Union[str, list[str], list[list[str]]],
) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions, str):
        predictions = [predictions]

    total = 0
    correct = 0
    for gold, pred in zip(references, predictions):
        total += 1
        gold_text = str(gold)
        pred_text = _first_prediction(pred)
        pred_answer = _extract_boxed(pred_text) or pred_text

        if _math_verify_equal(gold_text, pred_answer):
            correct += 1
        elif _normalize_text(gold_text) == _normalize_text(pred_text):
            correct += 1

    return correct / total if total else 0.0
