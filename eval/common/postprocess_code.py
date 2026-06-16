import os
import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("Usage: python eval/common/postprocess_code.py /path/to/samples.jsonl")

file_path = sys.argv[1]
cache_root = Path(os.environ.get("POSTPROCESS_CODE_CACHE", f"{file_path}.postprocess_cache"))
os.environ["HF_EVALUATE_CACHE"] = str(cache_root / "evaluate")
os.environ["HF_METRICS_CACHE"] = str(cache_root / "metrics")
cache_root.mkdir(parents=True, exist_ok=True)

COMMON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(COMMON_DIR))

import evaluate as hf_evaluate
from sanitize import sanitize

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
pass_at_k = hf_evaluate.load("code_eval")

def pass_at_1(references, predictions):
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]

import json
import re

        
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

data = read_jsonl(file_path)

references = [sample['target'] for sample in data]

def extract_raw_completion(sample):
    text = sample['resps'][0][0]
    return extract_code_block(text)

def extract_code_block(text):
    if '```python\n' in text:
        text = text.split('```python\n', 1)[-1]
    elif '```' in text:
        text = text.split('```', 1)[-1]
    return text.split('```')[0]

def infer_entrypoint(sample):
    doc = sample['doc']
    if 'entry_point' in doc:
        return doc['entry_point']
    for key in ('code', 'canonical_solution'):
        match = re.search(r'\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', doc.get(key, ''))
        if match:
            return match.group(1)
    match = re.search(r'\bassert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', sample.get('target', ''))
    if match:
        return match.group(1)
    return None

def has_entrypoint_definition(text, entrypoint):
    if not entrypoint:
        return False
    return re.search(rf'(?m)^\s*def\s+{re.escape(entrypoint)}\s*\(', text) is not None

def build_prediction(sample):
    doc = sample['doc']
    raw_completion = extract_raw_completion(sample)
    entrypoint = infer_entrypoint(sample)
    prompt = doc.get('prompt', '')
    text = f"{prompt}\n{raw_completion}" if prompt else raw_completion
    if has_entrypoint_definition(raw_completion, entrypoint):
        return sanitize(text, entrypoint), entrypoint, raw_completion
    return sanitize(text), entrypoint, raw_completion

prediction_items = [build_prediction(sample) for sample in data]
predictions = [[item[0]] for item in prediction_items]

pass_at_1s = [pass_at_1([reference], [prediction]) for reference, prediction in zip(references, predictions)]
print(sum(pass_at_1s)/len(pass_at_1s))

def write_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

res = [
    {
        "task_id": sample['doc']['task_id'],
        "completion": pred,
        "pass_at_1": res,
        "entrypoint": item[1],
        "raw_completion": item[2],
        "cleaned_completion": item[0],
    }
    for sample, pred, res, item in zip(data, predictions, pass_at_1s, prediction_items)
]
write_jsonl(res, file_path+'.cleaned')
