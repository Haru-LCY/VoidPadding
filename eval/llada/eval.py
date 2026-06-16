import accelerate
import torch
import re
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoConfig
from model.generation import (
    generate,
    generate_daedal,
    generate_rho_eos,
)
from model.modeling_llada import LLaDAModelLM
import json
import time
from model.chat_templates import clean_lora_chat_markers


def load_llada_model(model_path, **kwargs):
    return LLaDAModelLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        **kwargs,
    )


def default_tokenizer_path(model_path, tokenizer_path=None):
    if tokenizer_path:
        return tokenizer_path
    return model_path


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def as_optional_int(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def as_optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def _nfe_stats_path(rank: int) -> str | None:
    explicit = os.environ.get("EVAL_NFE_STATS_FILE")
    if explicit:
        return explicit
    stats_dir = os.environ.get("EVAL_NFE_STATS_DIR")
    if not stats_dir:
        return None
    return os.path.join(stats_dir, f"rank_{rank}.jsonl")


def _append_nfe_stats(path: str | None, records: list[dict]) -> None:
    if not path or not records:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _compact_token_runs(token_ids):
    runs = []
    for token_id in token_ids:
        token_id = int(token_id)
        if runs and runs[-1]["token_id"] == token_id:
            runs[-1]["count"] += 1
        else:
            runs.append({"token_id": token_id, "count": 1})
    return runs


def _token_id_or_none(tokenizer, token):
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    if isinstance(token_id, int):
        return int(token_id)
    return None


def _maybe_dump_raw_generations(
    *,
    generated_answer,
    input_ids,
    output_lengths,
    sample_nfes,
    tokenizer,
    eos_token_id,
    mask_id,
    dump_dir,
    dump_prefix,
    start_index,
):
    if not dump_dir:
        return

    os.makedirs(dump_dir, exist_ok=True)
    void_token_id = _token_id_or_none(tokenizer, "<|pad_0|>")
    for row in range(len(generated_answer)):
        sample_length = int(output_lengths[row]) if row < len(output_lengths) else generated_answer.shape[1] - input_ids.shape[1]
        suffix = generated_answer[row][input_ids.shape[1]:input_ids.shape[1] + sample_length].detach().cpu()
        raw_ids = [int(x) for x in suffix.tolist()]
        non_eos_positions = [idx for idx, token_id in enumerate(raw_ids) if token_id != int(eos_token_id)]
        eos_count = sum(1 for token_id in raw_ids if token_id == int(eos_token_id))
        mask_count = sum(1 for token_id in raw_ids if token_id == int(mask_id))
        void_count = (
            sum(1 for token_id in raw_ids if token_id == int(void_token_id))
            if void_token_id is not None
            else 0
        )
        last_non_eos = non_eos_positions[-1] if non_eos_positions else None
        tail_after_last_non_eos_is_all_eos = (
            True if last_non_eos is None else all(token_id == int(eos_token_id) for token_id in raw_ids[last_non_eos + 1:])
        )
        non_eos_ids = [raw_ids[idx] for idx in non_eos_positions]
        record = {
            "sample_index": int(start_index + row),
            "nfe": int(sample_nfes[row]) if row < len(sample_nfes) else None,
            "prompt_length": int(input_ids.shape[1]),
            "generated_length": int(sample_length),
            "eos_token_id": int(eos_token_id),
            "void_token_id": void_token_id,
            "mask_id": int(mask_id),
            "eos_count": int(eos_count),
            "non_eos_count": int(len(non_eos_positions)),
            "void_count": int(void_count),
            "mask_count": int(mask_count),
            "first_eos_position": next((idx for idx, token_id in enumerate(raw_ids) if token_id == int(eos_token_id)), None),
            "last_non_eos_position": last_non_eos,
            "tail_after_last_non_eos_is_all_eos": tail_after_last_non_eos_is_all_eos,
            "non_eos_positions": non_eos_positions,
            "non_eos_ids": non_eos_ids,
            "non_eos_text_skip_special": tokenizer.decode(non_eos_ids, skip_special_tokens=True),
            "visible_text_skip_special": tokenizer.decode(raw_ids, skip_special_tokens=True),
            "raw_token_runs": _compact_token_runs(raw_ids),
            "raw_token_ids": raw_ids,
        }
        path = os.path.join(dump_dir, f"{dump_prefix}_sample{start_index + row}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"raw_generation_dump: {path}")


def config_token_id(config, name, fallback=None):
    value = getattr(config, name, None)
    if value is None:
        return fallback
    return int(value)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=None,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        threshold=None,
        factor=None,
        save_dir=None,
        show_speed=False,
        tokenizer_path=None,
        ban_tokens=None,
        cut=False,
        chat_template=True,
        eos_token_id=None,
        void_expand=False,
        void_expand_max_length=None,
        void_expand_mode="dual_tail",
        void_expand_window=None,
        void_expand_tau_nonvoid=None,
        void_expand_tau_gap=0.0,
        void_expand_debug=False,
        generation_mode="default",
        daedal=False,
        rho_eos=False,
        initial_gen_length=64,
        max_gen_length=2048,
        cfg_scale=0.0,
        high_conf_threshold=0.90,
        low_conf_threshold=0.10,
        expansion_factor=8,
        eos_confidence_threshold=0.5,
        expand_eos_confidence_threshold=0.9,
        eos_check_tokens=32,
        low_density_threshold=0.4,
        high_density_threshold=0.6,
        scheduler="exp",
        daedal_debug=False,
        rho_eos_debug=False,
        assistant_prefix=None,
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA model path.
            mask_id: The token id of [MASK]. Defaults to config.mask_token_id.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()
        unsupported_dynamic_lora_args = {"lora_path", "merge_lora"} & set(kwargs)
        if unsupported_dynamic_lora_args:
            unsupported = ", ".join(sorted(unsupported_dynamic_lora_args))
            raise ValueError(
                f"LLaDA eval does not support dynamic LoRA loading ({unsupported}). "
                "Merge the adapter first with scripts/merge/merge_llada_lora.sh and pass the merged model as model_path."
            )

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        config = AutoConfig.from_pretrained(model_path)
        config.flash_attention = True
        self.model = load_llada_model(
            model_path,
            config=config,
            **model_kwargs,
        )
        self.model.eval()

        self._device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self._device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(self.device)

        self.tokenizer_path = default_tokenizer_path(model_path, tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)

        mask_id = as_optional_int(mask_id)
        if mask_id is None:
            mask_id = config_token_id(config, "mask_token_id", 126336)
        self.mask_id = int(mask_id)

        eos_token_id = as_optional_int(eos_token_id)
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            eos_token_id = self.tokenizer.eos_token_id
        elif eos_token_id is None:
            eos_token_id = config_token_id(config, "eos_token_id", 126081)
        self.eos_token_id = int(eos_token_id)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.threshold = threshold
        self.factor = factor
        self.generation_mode = str(generation_mode).strip().lower().replace("-", "_")
        if as_bool(daedal):
            self.generation_mode = "daedal"
        if as_bool(rho_eos):
            self.generation_mode = "rho_eos"
        if self.generation_mode not in {"default", "daedal", "rho_eos"}:
            raise ValueError("generation_mode must be one of: 'default', 'daedal', 'rho_eos'.")
        self.initial_gen_length = int(initial_gen_length)
        self.max_gen_length = int(max_gen_length)
        self.cfg_scale = float(cfg_scale)
        self.high_conf_threshold = float(high_conf_threshold)
        self.low_conf_threshold = float(low_conf_threshold)
        self.expansion_factor = int(expansion_factor)
        self.eos_confidence_threshold = float(eos_confidence_threshold)
        self.expand_eos_confidence_threshold = float(expand_eos_confidence_threshold)
        self.eos_check_tokens = int(eos_check_tokens)
        self.low_density_threshold = float(low_density_threshold)
        self.high_density_threshold = float(high_density_threshold)
        self.scheduler = str(scheduler)
        self.daedal_debug = as_bool(daedal_debug)
        self.rho_eos_debug = as_bool(rho_eos_debug)
        self.assistant_prefix = assistant_prefix
        self.if_apply_chat_template = as_bool(chat_template)
        self.is_instruct = self.if_apply_chat_template
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.ban_tokens = ban_tokens
        self.cut = as_bool(cut)
        self.void_expand = as_bool(void_expand)
        self.void_expand_max_length = as_optional_int(void_expand_max_length)
        self.void_expand_mode = str(void_expand_mode)
        self.void_expand_window = as_optional_int(void_expand_window)
        self.void_expand_tau_nonvoid = as_optional_float(void_expand_tau_nonvoid)
        self.void_expand_tau_gap = float(void_expand_tau_gap)
        self.void_expand_debug = as_bool(void_expand_debug)

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def _split_void_expand_metadata(self, metadata, batch_size):
        if not metadata:
            return [None] * int(batch_size)

        split_metadata = []
        candidates = metadata.get("void_expand_candidates") or []
        for row in range(int(batch_size)):
            row_metadata = dict(metadata)
            row_candidates = []
            for candidate in candidates:
                row_candidate = dict(candidate)
                row_nonvoid = candidate.get("row_mean_p_top_nonvoid")
                row_gap = candidate.get("row_mean_gap_prob")
                row_sufficient = candidate.get("row_sufficient")
                if isinstance(row_nonvoid, list) and row < len(row_nonvoid):
                    row_candidate["mean_p_top_nonvoid"] = row_nonvoid[row]
                if isinstance(row_gap, list) and row < len(row_gap):
                    row_candidate["mean_gap_prob"] = row_gap[row]
                if isinstance(row_sufficient, list) and row < len(row_sufficient):
                    row_candidate["sufficient"] = row_sufficient[row]
                row_candidate.pop("row_mean_p_top_nonvoid", None)
                row_candidate.pop("row_mean_gap_prob", None)
                row_candidate.pop("row_sufficient", None)
                row_candidates.append(row_candidate)
            row_metadata["void_expand_candidates"] = row_candidates
            split_metadata.append(row_metadata)
        return split_metadata

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    
    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        void_expand_selected_lengths = []
        void_expand_probe_steps = []
        void_expand_hit_max = []
        daedal_output_lengths = []
        daedal_stage1_probe_steps = []
        daedal_stage2_insertions = []
        rho_eos_output_lengths = []
        rho_eos_model_calls = []
        rho_eos_expansions = []
        rho_eos_contractions = []
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f'rank_{rank}.jsonl')
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, 'r', encoding='utf-8') as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                output = [
                    record.get("answer", record) if isinstance(record, dict) else record
                    for record in output
                ]
                print(f"processed_count: {processed_count}")
        
        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])
        
        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        start_time = time.time()
        nfe_stats_file = _nfe_stats_path(self.rank)

        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids = []
            max_len = 0
            pad_len = []
            for req in batch:
                question = req.args[0]
                if self.if_apply_chat_template:
                    m = [{"role": "user", "content": question}]
                    user_input = self.apply_chat_template(m, add_generation_prompt=True)
                    if self.assistant_prefix:
                        user_input += self.assistant_prefix
                    input_ids = self.tokenizer(user_input)['input_ids']
                else:
                    user_input = question
                    input_ids = self.tokenizer(user_input)['input_ids']
                batched_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
                pad_len.append(max_len - len(input_ids))
            unpadded_input_ids = [list(input_ids) for input_ids in batched_input_ids]
            
            # pad batched_input_ids to the same length
            batched_input_ids = [torch.cat([torch.full((1, max_len - len(input_ids)), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device), torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)], dim=1) for input_ids in batched_input_ids]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)
            
            if self.batch_size == 1:
                attention_mask = None
            else:
                attention_mask = torch.zeros((batched_input_ids.shape[0], 1, max_len+self.gen_length, max_len+self.gen_length), device=self.device, dtype=torch.bool)
                for i in range(len(pad_len)):
                    attention_mask[i, :, pad_len[i]:, pad_len[i]:] = True


            stop_tokens = batch[0].args[1].get('until', [])
            input_ids = batched_input_ids
            sample_daedal_metadata = [None] * len(batch)
            sample_rho_eos_metadata = [None] * len(batch)
            if self.generation_mode == "daedal":
                generation_kwargs = {
                    "initial_gen_length": self.initial_gen_length,
                    "max_gen_length": self.max_gen_length,
                    "block_length": self.block_length,
                    "temperature": 0,
                    "cfg_scale": self.cfg_scale,
                    "high_conf_threshold": self.high_conf_threshold,
                    "low_conf_threshold": self.low_conf_threshold,
                    "expansion_factor": self.expansion_factor,
                    "mask_id": self.mask_id,
                    "eos_token_id": self.eos_token_id,
                    "eos_confidence_threshold": self.eos_confidence_threshold,
                    "expand_eos_confidence_threshold": self.expand_eos_confidence_threshold,
                    "eos_check_tokens": self.eos_check_tokens,
                    "return_metadata": True,
                    "debug": self.daedal_debug,
                }
                sample_void_metadata = [None] * len(batch)
                sample_suffixes = []
                sample_daedal_metadata = []
                output_lengths = []
                sample_nfes = []
                nfe = 0
                for raw_input_ids in unpadded_input_ids:
                    single_input = torch.tensor(
                        raw_input_ids, dtype=torch.long, device=self.device
                    ).unsqueeze(0)
                    single_output, single_nfe, single_metadata = generate_daedal(
                        self.model, single_input, **generation_kwargs
                    )
                    single_output_length = int(single_metadata.get("daedal_output_lengths", [0])[0])
                    single_suffix = single_output[
                        0, single_input.shape[1]:single_input.shape[1] + single_output_length
                    ]
                    row_metadata = dict(single_metadata)
                    row_metadata["daedal_output_length"] = single_output_length
                    row_metadata["daedal_total_nfe"] = int(single_nfe)
                    sample_suffixes.append(single_suffix)
                    sample_daedal_metadata.append(row_metadata)
                    output_lengths.append(single_output_length)
                    decoding_nfe = int(row_metadata.get("daedal_stage2_model_calls", single_nfe))
                    sample_nfes.append(decoding_nfe)
                    nfe += decoding_nfe

                max_output_length = max(output_lengths) if output_lengths else 0
                generated_answer = torch.full(
                    (len(batch), input_ids.shape[1] + max_output_length),
                    self.eos_token_id,
                    dtype=torch.long,
                    device=self.device,
                )
                for row, suffix in enumerate(sample_suffixes):
                    generated_answer[row, input_ids.shape[1]:input_ids.shape[1] + suffix.shape[0]] = suffix

                daedal_output_lengths.extend(output_lengths)
                daedal_stage1_probe_steps.extend(
                    [int(meta.get("daedal_stage1_probe_steps", 0)) for meta in sample_daedal_metadata]
                )
                daedal_stage2_insertions.extend(
                    [int(meta.get("daedal_stage2_insertions", 0)) for meta in sample_daedal_metadata]
                )
            elif self.generation_mode == "rho_eos":
                generation_kwargs = {
                    "initial_gen_length": self.initial_gen_length,
                    "max_gen_length": self.max_gen_length,
                    "block_length": self.block_length,
                    "temperature": 0,
                    "cfg_scale": self.cfg_scale,
                    "high_conf_threshold": self.high_conf_threshold,
                    "low_conf_threshold": self.low_conf_threshold,
                    "expansion_factor": self.expansion_factor,
                    "mask_id": self.mask_id,
                    "eos_token_id": self.eos_token_id,
                    "low_density_threshold": self.low_density_threshold,
                    "high_density_threshold": self.high_density_threshold,
                    "scheduler": self.scheduler,
                    "return_metadata": True,
                    "debug": self.rho_eos_debug,
                }
                sample_void_metadata = [None] * len(batch)
                sample_suffixes = []
                sample_rho_eos_metadata = []
                output_lengths = []
                sample_nfes = []
                nfe = 0
                for raw_input_ids in unpadded_input_ids:
                    single_input = torch.tensor(
                        raw_input_ids, dtype=torch.long, device=self.device
                    ).unsqueeze(0)
                    single_output, single_nfe, single_metadata = generate_rho_eos(
                        self.model, single_input, **generation_kwargs
                    )
                    single_output_length = int(single_metadata.get("rho_eos_output_lengths", [0])[0])
                    single_suffix = single_output[
                        0, single_input.shape[1]:single_input.shape[1] + single_output_length
                    ]
                    row_metadata = dict(single_metadata)
                    row_metadata["rho_eos_output_length"] = single_output_length
                    sample_suffixes.append(single_suffix)
                    sample_rho_eos_metadata.append(row_metadata)
                    output_lengths.append(single_output_length)
                    sample_nfes.append(int(single_nfe))
                    nfe += int(single_nfe)

                max_output_length = max(output_lengths) if output_lengths else 0
                generated_answer = torch.full(
                    (len(batch), input_ids.shape[1] + max_output_length),
                    self.eos_token_id,
                    dtype=torch.long,
                    device=self.device,
                )
                for row, suffix in enumerate(sample_suffixes):
                    generated_answer[row, input_ids.shape[1]:input_ids.shape[1] + suffix.shape[0]] = suffix

                rho_eos_output_lengths.extend(output_lengths)
                rho_eos_model_calls.extend(
                    [int(meta.get("rho_eos_model_calls", 0)) for meta in sample_rho_eos_metadata]
                )
                rho_eos_expansions.extend(
                    [int(meta.get("rho_eos_expansions", 0)) for meta in sample_rho_eos_metadata]
                )
                rho_eos_contractions.extend(
                    [int(meta.get("rho_eos_contractions", 0)) for meta in sample_rho_eos_metadata]
                )
            else:
                generation_kwargs = {
                    "steps": self.steps,
                    "gen_length": self.gen_length,
                    "block_length": self.block_length,
                    "temperature": 0,
                    "remasking": self.remasking,
                    "mask_id": self.mask_id,
                    "threshold": self.threshold,
                    "factor": self.factor,
                    "ban_tokens": self.ban_tokens,
                    "cut": self.cut,
                    "tokenizer": self.tokenizer,
                    "eos_token_id": self.eos_token_id,
                    "void_expand": self.void_expand,
                    "void_expand_max_length": self.void_expand_max_length,
                    "void_expand_mode": self.void_expand_mode,
                    "void_expand_window": self.void_expand_window,
                    "void_expand_tau_nonvoid": self.void_expand_tau_nonvoid,
                    "void_expand_tau_gap": self.void_expand_tau_gap,
                    "void_expand_debug": self.void_expand_debug,
                    "return_metadata": self.void_expand,
                }
                generation_result = generate(self.model, input_ids, **generation_kwargs)

                if self.void_expand:
                    generated_answer, nfe, generation_metadata = generation_result
                    sample_void_metadata = self._split_void_expand_metadata(generation_metadata, len(batch))
                    selected_length = int(generation_metadata.get("void_expand_selected_gen_length", self.gen_length))
                    max_length = int(generation_metadata.get("void_expand_max_length", selected_length))
                    probe_steps = int(generation_metadata.get("void_expand_probe_steps", 0))
                    void_expand_selected_lengths.extend([selected_length] * len(batch))
                    void_expand_probe_steps.extend([probe_steps] * len(batch))
                    void_expand_hit_max.extend([selected_length == max_length] * len(batch))
                else:
                    generated_answer, nfe = generation_result
                    sample_void_metadata = [None] * len(batch)
                output_lengths = [generated_answer.shape[1] - input_ids.shape[1]] * len(batch)
                sample_nfes = [int(nfe)] * len(batch)

            _maybe_dump_raw_generations(
                generated_answer=generated_answer,
                input_ids=input_ids,
                output_lengths=output_lengths,
                sample_nfes=sample_nfes,
                tokenizer=self.tokenizer,
                eos_token_id=self.eos_token_id,
                mask_id=self.mask_id,
                dump_dir=os.environ.get("LLADA_RAW_DUMP_DIR"),
                dump_prefix=os.environ.get("LLADA_RAW_DUMP_PREFIX", "raw_generation"),
                start_index=len(output),
            )

            if self.is_instruct and 'task_id' in batch[0].doc and str(batch[0].doc['task_id']).lower().startswith('humaneval'):
                generated_answer_ids = generated_answer[:, input_ids.shape[1]:]
                if self.show_speed:
                    for i, sample_ids in enumerate(generated_answer_ids):
                        sample_length = int(output_lengths[i]) if i < len(output_lengths) else sample_ids.shape[0]
                        num_tokens += (sample_ids[:sample_length] != self.eos_token_id).sum()
                    if self.generation_mode in {"daedal", "rho_eos"}:
                        num_nfe += sum(sample_nfes)
                    else:
                        num_nfe += nfe
                batched_generated_answer = [
                    clean_lora_chat_markers(
                        self.tokenizer.decode(
                            generated_answer_ids[i][: int(output_lengths[i]) if i < len(output_lengths) else generated_answer_ids.shape[1]],
                            skip_special_tokens=True,
                        )
                    )
                    for i in range(len(generated_answer_ids))
                ]
            else:
                batched_generated_answer = []
                for i in range(len(generated_answer)):
                    sample_length = int(output_lengths[i]) if i < len(output_lengths) else generated_answer.shape[1] - input_ids.shape[1]
                    generated_answer_i = self.tokenizer.decode(
                        generated_answer[i][input_ids.shape[1]:input_ids.shape[1] + sample_length],
                        skip_special_tokens=False,
                    )
                    for stop_seq in stop_tokens:
                        if stop_seq in generated_answer_i:
                            generated_answer_i = generated_answer_i.split(stop_seq)[0]
                    generated_answer_ids = torch.tensor(self.tokenizer(generated_answer_i)["input_ids"])
                    if self.show_speed:
                        num_tokens += (generated_answer_ids != self.eos_token_id).sum()
                        num_nfe += sample_nfes[i] if i < len(sample_nfes) else nfe
                    generated_answer_i = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                    batched_generated_answer.append(generated_answer_i)

            # output.append(generated_answer)
            start_index = len(output)
            nfe_records = []
            for i, generated_answer_i in enumerate(batched_generated_answer):
                void_metadata = sample_void_metadata[i] if i < len(sample_void_metadata) else None
                daedal_metadata = sample_daedal_metadata[i] if i < len(sample_daedal_metadata) else None
                rho_eos_metadata = sample_rho_eos_metadata[i] if i < len(sample_rho_eos_metadata) else None
                expansion_probe_nfe = 0
                if void_metadata:
                    expansion_probe_nfe += int(void_metadata.get("void_expand_probe_steps", 0) or 0)
                if daedal_metadata:
                    expansion_probe_nfe += int(daedal_metadata.get("daedal_stage1_probe_steps", 0) or 0)
                nfe_records.append(
                    {
                        "model": "llada",
                        "sample_index": int(start_index + i),
                        "sample_in_batch": int(i),
                        "nfe": int(sample_nfes[i]) if i < len(sample_nfes) else int(nfe),
                        "nfe_scope": "decoding_only",
                        "configured_nfe": int(self.steps),
                        "expansion_probe_nfe": int(expansion_probe_nfe),
                        "generated_length": int(output_lengths[i]) if i < len(output_lengths) else None,
                        "gen_length": int(self.gen_length),
                        "block_length": int(self.block_length),
                        "cut": bool(self.cut),
                        "void_expand": bool(self.void_expand),
                        "generation_mode": self.generation_mode,
                        "void_expand_selected_gen_length": void_metadata.get("void_expand_selected_gen_length") if void_metadata else None,
                        "void_expand_stop_reason": void_metadata.get("void_expand_stop_reason") if void_metadata else None,
                        "daedal_stage1_probe_steps": daedal_metadata.get("daedal_stage1_probe_steps") if daedal_metadata else None,
                        "daedal_stage2_model_calls": daedal_metadata.get("daedal_stage2_model_calls") if daedal_metadata else None,
                        "daedal_total_nfe": daedal_metadata.get("daedal_total_nfe") if daedal_metadata else None,
                        "rho_eos_model_calls": rho_eos_metadata.get("rho_eos_model_calls") if rho_eos_metadata else None,
                    }
                )
            _append_nfe_stats(nfe_stats_file, nfe_records)
            output.extend(batched_generated_answer)

            if self.save_dir is not None:
                # Incrementally save newly generated answers
                with open(save_path, 'a', encoding='utf-8') as f:
                    for generated_answer, void_metadata, daedal_metadata, rho_eos_metadata in zip(
                        batched_generated_answer,
                        sample_void_metadata,
                        sample_daedal_metadata,
                        sample_rho_eos_metadata,
                    ):
                        if self.generation_mode == "daedal":
                            record = {"answer": generated_answer, "daedal": daedal_metadata}
                        elif self.generation_mode == "rho_eos":
                            record = {"answer": generated_answer, "rho_eos": rho_eos_metadata}
                        elif self.void_expand:
                            record = {"answer": generated_answer, "void_expand": void_metadata}
                        else:
                            record = generated_answer
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')

            for i in range(len(batched_generated_answer)):
                print('=' * 20)
                # print('question: ', question)
                print('answer: ', batched_generated_answer[i])
                print('nfe: ', nfe)
                if self.void_expand and sample_void_metadata[i] is not None:
                    print('void_expand_selected_gen_length: ', sample_void_metadata[i].get("void_expand_selected_gen_length"))
                    print('void_expand_probe_steps: ', sample_void_metadata[i].get("void_expand_probe_steps"))
                    print('void_expand_stop_reason: ', sample_void_metadata[i].get("void_expand_stop_reason"))
                if self.generation_mode == "daedal" and sample_daedal_metadata[i] is not None:
                    print('daedal_output_length: ', sample_daedal_metadata[i].get("daedal_output_length"))
                    print('daedal_stage1_probe_steps: ', sample_daedal_metadata[i].get("daedal_stage1_probe_steps"))
                    print('daedal_stage2_insertions: ', sample_daedal_metadata[i].get("daedal_stage2_insertions"))
                if self.generation_mode == "rho_eos" and sample_rho_eos_metadata[i] is not None:
                    print('rho_eos_output_length: ', sample_rho_eos_metadata[i].get("rho_eos_output_length"))
                    print('rho_eos_model_calls: ', sample_rho_eos_metadata[i].get("rho_eos_model_calls"))
                    print('rho_eos_expansions: ', sample_rho_eos_metadata[i].get("rho_eos_expansions"))
                    print('rho_eos_contractions: ', sample_rho_eos_metadata[i].get("rho_eos_contractions"))
                print('avg nfe: ', num_nfe / len(output))
                print('=' * 20, end='\n\n')
            # self.accelerator.wait_for_everyone()
        end_time = time.time()
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            print(f"Total NFE is {num_nfe}")
            if self.void_expand and void_expand_selected_lengths:
                selected = np.array(void_expand_selected_lengths, dtype=np.float64)
                probe = np.array(void_expand_probe_steps, dtype=np.float64)
                print(f"VoidExpand mean selected gen length: {selected.mean()}")
                print(f"VoidExpand median selected gen length: {np.median(selected)}")
                print(f"VoidExpand p90 selected gen length: {np.percentile(selected, 90)}")
                print(f"VoidExpand p95 selected gen length: {np.percentile(selected, 95)}")
                print(f"VoidExpand hit max rate: {float(np.mean(void_expand_hit_max))}")
                print(f"VoidExpand mean probe steps: {probe.mean()}")
            if self.generation_mode == "daedal" and daedal_output_lengths:
                lengths = np.array(daedal_output_lengths, dtype=np.float64)
                probes = np.array(daedal_stage1_probe_steps, dtype=np.float64)
                insertions = np.array(daedal_stage2_insertions, dtype=np.float64)
                print(f"DAEDAL mean output length: {lengths.mean()}")
                print(f"DAEDAL median output length: {np.median(lengths)}")
                print(f"DAEDAL p90 output length: {np.percentile(lengths, 90)}")
                print(f"DAEDAL mean stage1 probe steps: {probes.mean()}")
                print(f"DAEDAL mean stage2 insertions: {insertions.mean()}")
            if self.generation_mode == "rho_eos" and rho_eos_output_lengths:
                lengths = np.array(rho_eos_output_lengths, dtype=np.float64)
                model_calls = np.array(rho_eos_model_calls, dtype=np.float64)
                expansions = np.array(rho_eos_expansions, dtype=np.float64)
                contractions = np.array(rho_eos_contractions, dtype=np.float64)
                print(f"rho-EOS mean output length: {lengths.mean()}")
                print(f"rho-EOS median output length: {np.median(lengths)}")
                print(f"rho-EOS p90 output length: {np.percentile(lengths, 90)}")
                print(f"rho-EOS mean model calls: {model_calls.mean()}")
                print(f"rho-EOS mean expansions: {expansions.mean()}")
                print(f"rho-EOS mean contractions: {contractions.mean()}")
            
        return output


if __name__ == "__main__":
    cli_evaluate()
    
