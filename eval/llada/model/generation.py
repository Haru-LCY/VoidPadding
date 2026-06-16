import torch
import numpy as np
import torch.nn.functional as F
import os
import re
import json
from transformers import AutoTokenizer, AutoModel
from .modeling_llada import add_attention_trace_step_field, finish_attention_trace_step, start_attention_trace_step

from .daedal import generate_daedal
from .generation_common import DEFAULT_EOS_TOKEN_ID, add_gumbel_noise
from .rho_eos import generate_rho_eos

from .void_expand import DEFAULT_VOID_TOKEN_ID, _convert_token_to_id, maybe_apply_void_expand


def _generation_return(x, nfe, metadata, return_metadata):
    if return_metadata:
        return x, nfe, metadata
    return x, nfe


def _parse_ban_tokens(ban_tokens):
    if ban_tokens is None:
        return ()
    raw_tokens = []
    if isinstance(ban_tokens, str):
        raw_tokens.extend(re.split(r"[,;+]", ban_tokens))
    else:
        for item in ban_tokens:
            raw_tokens.extend(str(item).split(","))

    parsed = []
    for raw_token in raw_tokens:
        token = raw_token.strip().lower().replace("-", "_")
        if not token:
            continue
        if token != "void":
            raise ValueError("ban_tokens only supports 'void' for LLaDA generation.")
        if token not in parsed:
            parsed.append(token)
    return tuple(parsed)



def resolve_ban_token_ids(ban_tokens=None, tokenizer=None, eos_token_id=DEFAULT_EOS_TOKEN_ID):
    names = _parse_ban_tokens(ban_tokens)
    if not names:
        return ()

    token_ids = []
    for name in names:
        if name == "void":
            void_id = _convert_token_to_id(tokenizer, "<|pad_0|>")
            if void_id is None:
                void_id = DEFAULT_VOID_TOKEN_ID
            token_ids.append(void_id)

    deduped = []
    for token_id in token_ids:
        if int(token_id) not in deduped:
            deduped.append(int(token_id))
    return tuple(deduped)



def apply_ban_token_ids(logits, ban_token_ids):
    token_ids = tuple(int(token_id) for token_id in (ban_token_ids or ()))
    if not token_ids:
        return logits
    vocab_size = logits.shape[-1]
    invalid_ids = [token_id for token_id in token_ids if token_id < 0 or token_id >= vocab_size]
    if invalid_ids:
        raise ValueError(f"Ban token id(s) outside vocab size {vocab_size}: {invalid_ids}")
    constrained_logits = logits.clone()
    constrained_logits[..., list(token_ids)] = float("-inf")
    return constrained_logits


def sample_tokens_from_logits(logits, temperature=0.0, ban_token_ids=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    logits_with_noise = apply_ban_token_ids(logits_with_noise, ban_token_ids)
    return torch.argmax(logits_with_noise, dim=-1)

def _find_cut_boundaries(x, prompt_length, gen_length, mask_id, eos_token_id):
    boundaries = []
    generation_end = int(prompt_length) + int(gen_length)
    for row in range(x.shape[0]):
        generated = x[row, int(prompt_length):generation_end]
        eos_positions = (generated == int(eos_token_id)).nonzero(as_tuple=True)[0]
        if eos_positions.numel() == 0:
            boundaries.append(None)
            continue
        boundary = int(prompt_length) + int(eos_positions[0].item()) + 1
        if (x[row, int(prompt_length):boundary] == int(mask_id)).sum().item() == 0:
            boundaries.append(boundary)
        else:
            boundaries.append(None)
    return boundaries


def _mask_after_cut_boundary(mask_index, cut_boundaries):
    if not cut_boundaries:
        return mask_index
    mask_index = mask_index.clone()
    for row, boundary in enumerate(cut_boundaries):
        if boundary is not None:
            mask_index[row, int(boundary):] = False
    return mask_index


def _finalize_cut_output(x, prompt_length, gen_length, cut_boundaries, eos_token_id):
    if not cut_boundaries or all(boundary is None for boundary in cut_boundaries):
        return x
    if x.shape[0] == 1:
        boundary = cut_boundaries[0]
        if boundary is not None:
            return x[:, :int(boundary)]
        return x

    generation_end = int(prompt_length) + int(gen_length)
    x = x.clone()
    for row, boundary in enumerate(cut_boundaries):
        if boundary is not None and int(boundary) < generation_end:
            x[row, int(boundary):generation_end] = int(eos_token_id)
    return x


def _fill_after_cut_boundaries(x, prompt_length, gen_length, cut_boundaries, eos_token_id):
    if not cut_boundaries or all(boundary is None for boundary in cut_boundaries):
        return x
    generation_end = int(prompt_length) + int(gen_length)
    for row, boundary in enumerate(cut_boundaries):
        if boundary is not None and int(boundary) < generation_end:
            x[row, int(boundary):generation_end] = int(eos_token_id)
    return x


def _init_attention_trace(prompt, gen_length, block_length, steps, mask_id, eos_token_id):
    trace_dir = os.environ.get("LLADA_ATTENTION_TRACE_DIR")
    if not trace_dir:
        return None
    return {
        "trace_dir": trace_dir,
        "trace_prefix": os.environ.get("LLADA_ATTENTION_TRACE_PREFIX", "attention_trace"),
        "trace_every": int(os.environ.get("LLADA_ATTENTION_TRACE_EVERY", "8")),
        "prompt_length": int(prompt.shape[1]),
        "gen_length": int(gen_length),
        "block_length": int(block_length),
        "steps": int(steps),
        "mask_id": int(mask_id),
        "eos_token_id": int(eos_token_id),
        "steps_data": [],
    }


def _save_attention_trace(trace):
    if not trace:
        return
    os.makedirs(trace["trace_dir"], exist_ok=True)
    path = os.path.join(trace["trace_dir"], f"{trace['trace_prefix']}.pt")
    torch.save(trace, path)
    meta = {k: v for k, v in trace.items() if k != "steps_data"}
    meta["num_steps_recorded"] = len(trace["steps_data"])
    meta["num_attention_steps"] = sum(1 for item in trace["steps_data"] if item.get("traced_attention"))
    with open(os.path.join(trace["trace_dir"], f"{trace['trace_prefix']}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"attention_trace_dump: {path}", flush=True)


def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)                  # (B,)
    base  = torch.div(total, steps, rounding_mode='floor')  # (B,)
    rem   = total - base * steps                         # (B,)

    # Start with base for all steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)

    # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
    cols = torch.arange(steps, device=device).unsqueeze(0)               # (1, steps)
    add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)

    return num_transfer_tokens






@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None,
             ban_tokens=None, cut=False, tokenizer=None, eos_token_id=DEFAULT_EOS_TOKEN_ID,
             void_expand=False, void_expand_max_length=None, void_expand_mode="dual_tail",
             void_expand_window=None, void_expand_tau_nonvoid=None, void_expand_tau_gap=0.0,
             void_expand_debug=False, return_metadata=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    gen_length, steps, metadata = maybe_apply_void_expand(
        model,
        prompt,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        mask_id=mask_id,
        tokenizer=tokenizer,
        void_expand=void_expand,
        void_expand_max_length=void_expand_max_length,
        void_expand_mode=void_expand_mode,
        void_expand_window=void_expand_window,
        void_expand_tau_nonvoid=void_expand_tau_nonvoid,
        void_expand_tau_gap=void_expand_tau_gap,
        void_expand_debug=void_expand_debug,
    )

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    ban_token_ids = resolve_ban_token_ids(ban_tokens, tokenizer=tokenizer, eos_token_id=eos_token_id)
    attention_trace = _init_attention_trace(prompt, gen_length, block_length, steps * num_blocks, mask_id, eos_token_id)
    cut_boundaries = [None] * prompt.shape[0]
    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            if attention_trace:
                start_attention_trace_step(
                    x,
                    prompt_length=prompt.shape[1],
                    step=nfe,
                    block=num_block,
                    inner_step=i,
                    mask_id=mask_id,
                    eos_token_id=eos_token_id,
                    trace_every=attention_trace["trace_every"],
            )
            logits = model(x).logits
            if attention_trace:
                gen_logits = logits[:, prompt.shape[1]:prompt.shape[1] + gen_length, :].float()
                p_eos = torch.exp(gen_logits[..., int(eos_token_id)] - torch.logsumexp(gen_logits, dim=-1))
                add_attention_trace_step_field("p_eos_generation", p_eos[0].detach().cpu().to(torch.float16))
            if attention_trace:
                step_record = finish_attention_trace_step()
                if step_record is not None:
                    attention_trace["steps_data"].append(step_record)
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if cut:
                mask_index = _mask_after_cut_boundary(mask_index, cut_boundaries)
            quota = None
            if threshold is None:
                if i < num_transfer_tokens.shape[1]:
                    quota = num_transfer_tokens[:, i]
                else:
                    quota = mask_index.sum(dim=1)
            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x,
                    quota,
                    threshold,
                    ban_token_ids,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x,
                    None,
                    factor,
                    ban_token_ids,
                )
            x[transfer_index] = x0[transfer_index]
            if cut:
                new_boundaries = _find_cut_boundaries(x, prompt.shape[1], gen_length, mask_id, eos_token_id)
                cut_boundaries = [old if old is not None else new for old, new in zip(cut_boundaries, new_boundaries)]
                x = _fill_after_cut_boundaries(x, prompt.shape[1], gen_length, cut_boundaries, eos_token_id)
                if all(boundary is not None for boundary in cut_boundaries):
                    final_x = _finalize_cut_output(x, prompt.shape[1], gen_length, cut_boundaries, eos_token_id)
                    _save_attention_trace(attention_trace)
                    return _generation_return(final_x, nfe, metadata, return_metadata)
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    if cut:
        x = _finalize_cut_output(x, prompt.shape[1], gen_length, cut_boundaries, eos_token_id)
    _save_attention_trace(attention_trace)
    return _generation_return(x, nfe, metadata, return_metadata)

def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
    ban_token_ids=None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    # 1) Sample proposal x0
    # Gumbel-noise for exploration; if temperature==0, add_gumbel_noise should no-op
    x0 = sample_tokens_from_logits(logits, temperature=temperature, ban_token_ids=ban_token_ids)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        # Use higher precision for softmax stability
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # 3) Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        # (No top-k; purely threshold-based)
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True) # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked (consider fully unmasked rows)
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    # idx: (B, L) gives positions in original sequence sorted by confidence
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    # Use integer scatter then cast to bool (scatter_ on bool can be finicky across versions)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8) # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index

def get_transfer_index_dynamic(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    factor=1,
    ban_token_ids=None,
):
    x0 = sample_tokens_from_logits(logits, temperature=temperature, ban_token_ids=ban_token_ids) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue
        
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index
