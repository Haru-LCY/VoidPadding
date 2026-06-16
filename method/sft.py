import os
from datetime import datetime, timedelta
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    import swanlab
except ImportError:
    swanlab = None


class DreamWrapper(nn.Module):
    """Dream needs a wrapper so PEFT sees the expected generation API."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def generate_sft_data(tokenizer, max_data_len, data_size):
    ds1 = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
    ds2 = load_dataset("allenai/tulu-3-sft-mixture", split="train")

    ds1_mapped = ds1.map(convert_data, remove_columns=ds1.column_names, num_proc=32)
    ds2_mapped = ds2.map(convert_data, remove_columns=ds2.column_names, num_proc=32)
    ds1_filtered = ds1_mapped.filter(lambda x: x["valid"], num_proc=32)
    ds2_filtered = ds2_mapped.filter(lambda x: x["valid"], num_proc=32)
    combined_ds = concatenate_datasets([ds1_filtered, ds2_filtered])
    print(f"Total Num of Data: {len(combined_ds)}")

    combined_ds = deduplicate_text_dataset(combined_ds)
    tok = partial(tokenize_batch, tokenizer=tokenizer, max_data_len=max_data_len)

    tokenized_ds = combined_ds.map(
        tok,
        batched=True,
        num_proc=32,
        batch_size=1000,
        remove_columns=combined_ds.column_names,
    )
    return filter_and_sample(tokenized_ds, drop_cutoff=8192, sample_n=data_size, seed=42)


def convert_data(example):
    msg = example.get("messages")
    if msg and len(msg) == 2:
        return {"input": msg[0]["content"], "output": msg[1]["content"], "valid": True}
    return {"input": "", "output": "", "valid": False}


def tokenize_batch(batch, tokenizer, max_data_len):
    texts, prompt_lens, seq_lens = [], [], []

    for inp, out in zip(batch["input"], batch["output"]):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": inp}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt_text + out
        texts.append(full_text)

        encoded_prompt = tokenizer(prompt_text, add_special_tokens=True)
        prompt_lens.append(len(encoded_prompt["input_ids"]))

        encoded_full = tokenizer(full_text, add_special_tokens=True)
        seq_lens.append(len(encoded_full["input_ids"]))

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_data_len,
        padding=False,
        add_special_tokens=True,
    )
    tokenized["prompt_length"] = prompt_lens
    tokenized["seq_length"] = seq_lens
    return tokenized


def deduplicate_text_dataset(ds):
    def make_key(batch):
        return {
            "dedup_key": [inp.strip() + " ||| " + out.strip() for inp, out in zip(batch["input"], batch["output"])]
        }

    ds = ds.map(make_key, batched=True, num_proc=32)
    df = ds.to_pandas()
    before = len(df)
    df = df.drop_duplicates(subset=["dedup_key"])
    after = len(df)
    print(f"[Deduplication] Removed {before - after} duplicates -> {after} remain")

    ds = HFDataset.from_pandas(df, preserve_index=False)
    return ds.remove_columns("dedup_key")


def filter_and_sample(ds, drop_cutoff=4096, sample_n=500_000, seed=42):
    before = len(ds)
    ds = ds.filter(lambda x: x["seq_length"] <= drop_cutoff)
    print(f"[Cutoff] Dropped {before - len(ds)} samples longer than {drop_cutoff} -> {len(ds)} remain")

    if len(ds) > sample_n:
        ds = ds.shuffle(seed=seed).select(range(sample_n))
        print(f"[Sample] Randomly selected {sample_n} samples")
    else:
        print(f"[Sample] Dataset smaller than {sample_n}, kept all {len(ds)} samples")

    return ds


class SFTDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        accelerator,
        max_data_len=1024,
        cache_dir="./tokenized_sft_dataset",
        model_type="llada_base",
        data_size=500_000,
    ):
        self.tokenizer = tokenizer
        self.max_data_len = max_data_len

        using_default_cache_dir = cache_dir is None
        if using_default_cache_dir:
            cache_dir = "./tokenized_sft_dataset"

        if using_default_cache_dir and not os.path.exists(cache_dir):
            model_tag = model_type.split("_")[0]
            cache_dir += f"_{model_tag}_{data_size}"

        if os.path.exists(cache_dir):
            print(f"Loading pre-tokenized dataset from {cache_dir}")
        else:
            if accelerator.is_main_process:
                print("Preprocessing and tokenizing dataset...")
                tokenized_ds = generate_sft_data(self.tokenizer, self.max_data_len, data_size)
                tokenized_ds.save_to_disk(cache_dir)
                print("dataset saved")
            accelerator.wait_for_everyone()

        self.ds = load_from_disk(cache_dir)
        print(f"Total Num of Data: {len(self.ds)}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        prompt_length = example.get("prompt_length", example.get("prompt_lengths"))
        if prompt_length is None:
            raise KeyError("Dataset example is missing both 'prompt_length' and 'prompt_lengths'")
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "prompt_length": prompt_length,
        }


class SwanLabRun:
    def __init__(self, args, config, experiment_name, log_dir):
        self.enabled = False
        self._run = None

        if getattr(args, "swanlab_mode", "local") == "disabled":
            return
        if swanlab is None:
            print("SwanLab is not installed; local experiment logging is disabled.")
            return

        os.makedirs(log_dir, exist_ok=True)
        try:
            self._run = swanlab.init(
                project="void-padding",
                experiment_name=experiment_name,
                config=config,
                mode="local",
                logdir=log_dir,
            )
        except Exception as exc:
            print(f"SwanLab init failed; disabling logging for this run: {exc}")
            self._run = None
            return

        self.enabled = True

    def log(self, data, step):
        if self.enabled:
            swanlab.log(data, step=step)

    def finish(self):
        if self.enabled:
            swanlab.finish()


def build_experiment_name(args):
    if getattr(args, "run_name", None):
        return args.run_name
    if args.pad_strategy == "void":
        return f"{args.model_type}_padvoid_normalmask_normal"
    if args.pad_strategy == "eos":
        return f"{args.model_type}_padeos"
    return f"{args.model_type}_pad{args.pad_num}"


def build_swanlab_log_dir(args, experiment_name):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(args.swanlab_logdir, args.model_type, f"{stamp}_{experiment_name}")


def build_checkpoint_suffix(args):
    if args.pad_strategy == "void":
        pad_suffix = "padvoid_normalmask_normal"
        checkpoint_suffix = f"{args.method}_{args.learning_rate}_lora_rank{args.lora_rank}_{pad_suffix}"
        extra_suffix = getattr(args, "checkpoint_suffix_extra", "")
        if extra_suffix:
            checkpoint_suffix = f"{checkpoint_suffix}_{extra_suffix}"
        return checkpoint_suffix

    pad_suffix = "padeos" if args.pad_strategy == "eos" else f"pad{args.pad_num}"
    checkpoint_suffix = f"{args.method}_{args.learning_rate}_lora_rank{args.lora_rank}_{pad_suffix}"
    extra_suffix = getattr(args, "checkpoint_suffix_extra", "")
    if extra_suffix:
        checkpoint_suffix = f"{checkpoint_suffix}_{extra_suffix}"
    return checkpoint_suffix


def save_checkpoint(accelerator, model, optimizer, args, epoch, step, global_step, update_step, save_tag):
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return

    checkpoint_suffix = build_checkpoint_suffix(args)
    save_dir = os.path.join("model", args.model_type, f"{checkpoint_suffix}_{save_tag}")
    os.makedirs(save_dir, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(save_dir)

    state_dict = {
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "update_step": update_step,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "batch_size": args.batch_size,
    }
    torch.save(state_dict, os.path.join(save_dir, "training_state.pt"))
    print(f"Saved model to {save_dir}")
    accelerator.wait_for_everyone()


def parse_save_at_update_steps(value):
    if not value:
        return set()
    steps = set()
    for raw_step in str(value).split(","):
        raw_step = raw_step.strip()
        if not raw_step:
            continue
        step = int(raw_step)
        if step <= 0:
            raise ValueError("save_at_update_steps entries must be positive integers")
        steps.add(step)
    return steps


def get_pad0_id(pad_token_ids, model_name):
    if pad_token_ids is None:
        raise ValueError("pad_token_ids must be provided to resolve VOID token")
    if "llada" in model_name:
        return int(pad_token_ids)
    return int(pad_token_ids[0])


def collate_fn_pad(
    batch,
    eot_id,
    model_name,
    pad_token_ids,
    pad_num=7,
    pad_strategy="rainbow",
):
    input_ids_list = [torch.as_tensor(sample["input_ids"], dtype=torch.long) for sample in batch]
    prompt_lengths = torch.as_tensor([sample["prompt_length"] for sample in batch], dtype=torch.long)

    batch_size = len(input_ids_list)
    max_len = max(len(x) for x in input_ids_list)

    if pad_strategy == "void":
        max_len = max_len + 1
        void_id = get_pad0_id(pad_token_ids, model_name)
        padded_inputs = torch.full((batch_size, max_len), fill_value=void_id, dtype=torch.long)
        eos_loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        void_region_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for i, seq in enumerate(input_ids_list):
            seq_len = len(seq)
            padded_inputs[i, :seq_len] = seq
            padded_inputs[i, seq_len] = eot_id
            eos_loss_mask[i, seq_len] = True
            if seq_len + 1 < max_len:
                void_region_mask[i, seq_len + 1 :] = True

        return {
            "input_ids": padded_inputs,
            "prompt_lengths": prompt_lengths,
            "eos_loss_mask": eos_loss_mask,
            "void_region_mask": void_region_mask,
        }

    padded_inputs = torch.full((batch_size, max_len), fill_value=eot_id, dtype=torch.long)

    for i, seq in enumerate(input_ids_list):
        seq_len = len(seq)
        padded_inputs[i, :seq_len] = seq

        pad_len = max_len - seq_len
        if pad_len > 1 and pad_num > 0:
            if pad_strategy == "eos":
                continue
            if "llada" in model_name:
                cycle_tokens = torch.arange(pad_token_ids, pad_token_ids + pad_num, dtype=torch.long)
            else:
                cycle_tokens = torch.as_tensor(pad_token_ids[:pad_num], dtype=torch.long)
            idx = torch.arange(pad_len - 1) % pad_num
            padded_inputs[i, seq_len + 1 :] = cycle_tokens[idx]

    return {"input_ids": padded_inputs, "prompt_lengths": prompt_lengths}


def forward_process(input_ids, msk_id, eps=1e-3):
    batch_size, seq_len = input_ids.shape
    t = torch.rand(batch_size, device=input_ids.device)
    p_mask = ((1 - eps) * t + eps)[:, None].repeat(1, seq_len)
    masked_indices = torch.rand((batch_size, seq_len), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, torch.full_like(input_ids, msk_id), input_ids)
    return noisy_batch, masked_indices, p_mask


def build_answer_loss_mask(input_ids, prompt_length):
    token_positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand(
        input_ids.size(0), input_ids.size(1)
    )
    return token_positions >= prompt_length.unsqueeze(1).to(input_ids.device)


def build_void_response_loss_mask(input_ids, prompt_length, eos_loss_mask):
    token_positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand(
        input_ids.size(0), input_ids.size(1)
    )
    seq_len = input_ids.shape[1]
    eos_pos = torch.where(
        eos_loss_mask,
        token_positions,
        torch.full_like(token_positions, fill_value=seq_len),
    ).min(dim=1).values
    return (token_positions >= prompt_length.unsqueeze(1).to(input_ids.device)) & (
        token_positions < eos_pos.unsqueeze(1)
    )


def apply_void_noising(input_ids, noisy_batch, p_mask, prompt_length, eos_loss_mask, void_region_mask, msk_id):
    token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(
        noisy_batch.size(0), noisy_batch.size(1)
    )
    prompt_mask = token_positions < prompt_length.unsqueeze(1).to(noisy_batch.device)
    noisy_batch[prompt_mask] = input_ids[prompt_mask]

    eos_loss_mask = eos_loss_mask.to(noisy_batch.device)
    void_region_mask = void_region_mask.to(noisy_batch.device)

    noisy_batch[eos_loss_mask] = msk_id
    p_mask[eos_loss_mask] = 1.0

    void_loss_mask = void_region_mask & (noisy_batch == msk_id)
    response_target_mask = build_void_response_loss_mask(input_ids, prompt_length, eos_loss_mask)
    response_loss_mask = response_target_mask & (noisy_batch == msk_id)
    base_target_mask = response_target_mask | eos_loss_mask
    base_loss_mask = response_loss_mask | eos_loss_mask

    return noisy_batch, base_loss_mask, void_loss_mask, base_target_mask


def _llada_masked_loss(logits, labels, loss_mask, p_mask, normalizer):
    if not loss_mask.any():
        return logits.sum() * 0.0
    token_loss = F.cross_entropy(logits[loss_mask], labels[loss_mask], reduction="none") / p_mask[loss_mask]
    return torch.sum(token_loss / normalizer[loss_mask]) / labels.shape[0]


def _dream_masked_loss(logits, labels, loss_mask, p_mask):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    label_mask = loss_mask[:, 1:]
    p_mask_shift = p_mask[:, 1:]
    if not label_mask.any():
        return logits.sum() * 0.0
    token_loss = (
        F.cross_entropy(shift_logits[label_mask], shift_labels[label_mask], reduction="none")
        / p_mask_shift[label_mask]
    )
    return token_loss.sum() / label_mask.sum().clamp_min(1)


def get_model_constants(model_type):
    if "llada" in model_type:
        return {
            "tokenizer_path": "./tokenizer_llada",
            "eot_id": 126081,
            "msk_id": 126336,
            "pad_id": 126084,
        }
    return {
        "tokenizer_path": "./tokenizer_dream",
        "eot_id": 151643,
        "msk_id": 151666,
        "pad_id": [
            56940,
            86766,
            79871,
            65271,
            51475,
            95429,
            97417,
            39484,
            81221,
            58591,
            55799,
            14642,
            41056,
            46145,
            37144,
            22335,
            46771,
            42708,
            51068,
            19273,
            47549,
            38171,
            5238,
            15270,
            37083,
            31979,
            38484,
            11950,
            32678,
            26065,
        ],
    }


def build_lora_config(args):
    if "llada" in args.model_type:
        return LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "ff_proj", "up_proj", "ff_out"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def is_offline_mode():
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def train(args):
    print("SFT Training Started...")
    ddp_timeout = InitProcessGroupKwargs(timeout=timedelta(hours=6))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_timeout],
    )
    device = accelerator.device

    if args.pad_strategy not in {"rainbow", "eos", "void"}:
        raise ValueError("Only rainbow, eos, and void padding strategies are supported.")
    if args.pad_num < 0:
        raise ValueError("pad_num must be non-negative")

    constants = get_model_constants(args.model_type)
    eot_id = constants["eot_id"]
    msk_id = constants["msk_id"]
    pad_id = constants["pad_id"]

    local_files_only = is_offline_mode()
    tokenizer = AutoTokenizer.from_pretrained(
        constants["tokenizer_path"],
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    base_model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    base_model = base_model if "llada" in args.model_type else DreamWrapper(base_model)

    if args.resume_dir:
        model = PeftModel.from_pretrained(base_model, args.resume_dir, is_trainable=True)
        model.train()
    else:
        model = get_peft_model(base_model, build_lora_config(args))
        model.train()

        for name, param in model.named_parameters():
            if "lora_B" in name:
                with torch.no_grad():
                    param.zero_()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    resume_state = None
    if args.resume_dir:
        state_path = os.path.join(args.resume_dir, "training_state.pt")
        resume_state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(resume_state["optimizer"])

    experiment_name = build_experiment_name(args)
    swanlab_log_dir = build_swanlab_log_dir(args, experiment_name)
    save_at_update_steps = parse_save_at_update_steps(getattr(args, "save_at_update_steps", ""))
    stop_at_update_step = int(getattr(args, "stop_at_update_step", 0) or 0)
    if stop_at_update_step < 0:
        raise ValueError("stop_at_update_step must be non-negative")

    dataset = SFTDataset(
        tokenizer=tokenizer,
        model_type=args.model_type,
        accelerator=accelerator,
        cache_dir=args.sft_cache_dir,
    )
    collate_fn = partial(
        collate_fn_pad,
        eot_id=eot_id,
        model_name=args.model_type,
        pad_token_ids=pad_id,
        pad_num=args.pad_num,
        pad_strategy=args.pad_strategy,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    swanlab_run = None
    if accelerator.is_main_process:
        swanlab_config = {
            "model_type": args.model_type,
            "method": args.method,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "epochs": args.epochs,
            "pad_num": args.pad_num,
            "pad_strategy": args.pad_strategy,
            "stop_at_update_step": stop_at_update_step,
            "sft_cache_dir": args.sft_cache_dir,
        }
        swanlab_run = SwanLabRun(
            args=args,
            config=swanlab_config,
            experiment_name=experiment_name,
            log_dir=swanlab_log_dir,
        )
        if swanlab_run.enabled:
            print(f"SwanLab local log dir: {swanlab_log_dir}")

    global_step = 0
    update_step = 0
    start_epoch = 0
    resume_step_in_epoch = 0

    if resume_state is not None:
        global_step = int(resume_state.get("global_step", 0))
        update_step = int(resume_state.get("update_step", global_step))
        start_epoch = int(resume_state.get("epoch", 0))
        resume_step_in_epoch = int(resume_state.get("step", -1)) + 1
        saved_grad_accum = int(resume_state.get("gradient_accumulation_steps", args.gradient_accumulation_steps))
        saved_batch_size = int(resume_state.get("batch_size", args.batch_size))

        if resume_step_in_epoch >= len(dataloader):
            start_epoch += 1
            resume_step_in_epoch = 0

        if accelerator.is_main_process:
            if saved_grad_accum != args.gradient_accumulation_steps or saved_batch_size != args.batch_size:
                print(
                    "Warning: resumed checkpoint was saved with "
                    f"batch_size={saved_batch_size}, gradient_accumulation_steps={saved_grad_accum}; "
                    "changing them will change the effective batch and training trajectory."
                )
            print(
                "Resuming training "
                f"from epoch={start_epoch + 1}, step={resume_step_in_epoch + 1}, "
                f"global_step={global_step}, update_step={update_step}"
            )

    last_completed_epoch = start_epoch - 1
    last_completed_step = -1
    stopped_at_update_step = False
    for epoch in range(start_epoch, args.epochs):
        if accelerator.is_main_process:
            print(f"\n[Epoch {epoch + 1}/{args.epochs}]")

        epoch_loss_sum = 0.0
        epoch_step_count = 0
        skip_steps = resume_step_in_epoch if epoch == start_epoch else 0

        for step, batch in enumerate(tqdm(dataloader, disable=not accelerator.is_main_process)):
            if step < skip_steps:
                continue

            with accelerator.accumulate(model):
                input_ids = batch["input_ids"].to(device)
                prompt_length = batch["prompt_lengths"].to(device)

                noisy_batch, masked_indices, p_mask = forward_process(input_ids, msk_id=msk_id)

                if args.pad_strategy == "void":
                    noisy_batch, base_loss_mask, void_loss_mask, base_target_mask = apply_void_noising(
                        input_ids=input_ids,
                        noisy_batch=noisy_batch,
                        p_mask=p_mask,
                        prompt_length=prompt_length,
                        eos_loss_mask=batch["eos_loss_mask"].to(device),
                        void_region_mask=batch["void_region_mask"].to(device),
                        msk_id=msk_id,
                    )
                    masked_indices = base_loss_mask | void_loss_mask
                    answer_loss_mask = base_target_mask | batch["void_region_mask"].to(device)
                else:
                    answer_loss_mask = build_answer_loss_mask(input_ids, prompt_length)

                    token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(
                        noisy_batch.size(0), noisy_batch.size(1)
                    )
                    prompt_mask = token_positions < prompt_length.unsqueeze(1).to(noisy_batch.device)
                    noisy_batch[prompt_mask] = input_ids[prompt_mask]

                    masked_indices = (noisy_batch == msk_id) & answer_loss_mask

                answer_length = torch.sum(answer_loss_mask.to(torch.int64), dim=-1, keepdim=True).clamp_min(1)
                answer_length = answer_length.repeat(1, noisy_batch.shape[1])

                logits = model(input_ids=noisy_batch).logits
                if "llada" in args.model_type:
                    loss = _llada_masked_loss(logits, input_ids, masked_indices, p_mask, answer_length)
                else:
                    loss = _dream_masked_loss(logits, input_ids, masked_indices, p_mask)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss_sum += loss.item()
            epoch_step_count += 1
            global_step += 1
            last_completed_epoch = epoch
            last_completed_step = step
            if accelerator.sync_gradients:
                update_step += 1

            reduced_loss = accelerator.gather_for_metrics(loss.detach().reshape(1)).float().mean().cpu().item()
            if accelerator.is_main_process and swanlab_run is not None:
                swanlab_run.log({"train_loss": float(reduced_loss)}, step=global_step)

            if accelerator.is_main_process and (step + 1) % 100 == 0:
                print(f"[Epoch {epoch + 1} Step {step + 1}] Loss: {loss.item():.4f}")

            saved_step_checkpoint = False
            if (
                accelerator.sync_gradients
                and update_step in save_at_update_steps
            ):
                save_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    step=step,
                    global_step=global_step,
                    update_step=update_step,
                    save_tag=f"step{update_step}",
                )
                saved_step_checkpoint = True

            if accelerator.sync_gradients and stop_at_update_step > 0 and update_step >= stop_at_update_step:
                if not saved_step_checkpoint:
                    save_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        args=args,
                        epoch=epoch,
                        step=step,
                        global_step=global_step,
                        update_step=update_step,
                        save_tag=f"step{update_step}",
                    )
                stopped_at_update_step = True
                if accelerator.is_main_process:
                    print(f"Reached stop_at_update_step={stop_at_update_step}; stopping training.")
                break

        avg_loss = epoch_loss_sum / max(1, epoch_step_count)
        if accelerator.is_main_process:
            print(f"[Epoch {epoch + 1}] Average Loss: {avg_loss:.4f}")

        if stopped_at_update_step:
            break

        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                args=args,
                epoch=epoch,
                step=step,
                global_step=global_step,
                update_step=update_step,
                save_tag=f"epoch{epoch + 1}",
            )

    if start_epoch < args.epochs and not stopped_at_update_step:
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            args=args,
            epoch=max(last_completed_epoch, 0),
            step=max(last_completed_step, 0),
            global_step=global_step,
            update_step=update_step,
            save_tag="final",
        )

    if accelerator.is_main_process:
        if swanlab_run is not None:
            swanlab_run.finish()
        print("Training complete!")
