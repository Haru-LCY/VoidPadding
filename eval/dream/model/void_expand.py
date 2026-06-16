import torch

DEFAULT_VOID_TOKEN_ID = 56940
VOID_TOKEN_CANDIDATES = ("<pad_0>", "<|pad_0|>", "<|reserved_token_6|>")
VOID_EXPAND_MODES = ("dual_tail",)


def _coerce_token_ids(value):
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        ids = []
        for item in value:
            ids.extend(_coerce_token_ids(item))
        return ids
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def _convert_token_to_id(tokenizer, token):
    if tokenizer is None or not hasattr(tokenizer, "convert_tokens_to_ids"):
        return None
    token_id = tokenizer.convert_tokens_to_ids(token)
    token_ids = _coerce_token_ids(token_id)
    if len(token_ids) != 1:
        return None
    int_token_id = int(token_ids[0])
    unk_id = getattr(tokenizer, "unk_token_id", None)
    unk_token = getattr(tokenizer, "unk_token", None)
    if unk_id is not None and int_token_id == int(unk_id) and token != str(unk_token):
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            vocab = {}
        if token not in vocab:
            return None
    return int_token_id


def resolve_void_token_id(tokenizer=None):
    for token in VOID_TOKEN_CANDIDATES:
        token_id = _convert_token_to_id(tokenizer, token)
        if token_id is not None:
            return int(token_id)
    return int(DEFAULT_VOID_TOKEN_ID)


def get_void_expand_candidate_lengths(gen_length, expand_block_length, max_length):
    if int(expand_block_length) <= 0:
        raise ValueError("void_expand requires void_expand_block_length > 0.")
    if int(gen_length) < 1:
        raise ValueError("void_expand requires gen_length >= 1.")
    if int(max_length) < int(gen_length):
        raise ValueError("void_expand_max_length must be >= gen_length.")

    block = int(expand_block_length)
    start = ((int(gen_length) + block - 1) // block) * block
    max_block = (int(max_length) // block) * block
    if max_block < start:
        raise ValueError("void_expand_max_length must allow at least one block-aligned candidate.")
    return list(range(start, max_block + 1, block))


def _score_void_expand_dual_tail(response_logits, *, void_token_id, window, tau_nonvoid, tau_gap):
    response_logits = response_logits.detach().float()
    if response_logits.dim() == 2:
        response_logits = response_logits.unsqueeze(0)
    if response_logits.dim() != 3 or response_logits.shape[1] < 1:
        raise ValueError("VOID probe logits must have shape [B, length, vocab].")
    if response_logits.shape[-1] <= int(void_token_id):
        raise ValueError(f"VOID token id {void_token_id} is outside vocab size {response_logits.shape[-1]}.")

    length = int(response_logits.shape[1])
    effective_window = min(int(window), length)
    tail_logits = response_logits[:, length - effective_window:length, :]

    top2_values, top2_indices = torch.topk(tail_logits, k=2, dim=-1)
    argmax_is_void = top2_indices[..., 0].eq(int(void_token_id))
    best_nonvoid_logits = torch.where(argmax_is_void, top2_values[..., 1], top2_values[..., 0])

    log_denom = torch.logsumexp(tail_logits, dim=-1)
    p_void = torch.exp(tail_logits[..., int(void_token_id)] - log_denom)
    p_top_nonvoid = torch.exp(best_nonvoid_logits - log_denom)
    gap_prob = p_void - p_top_nonvoid

    row_mean_p_top_nonvoid = p_top_nonvoid.float().mean(dim=1)
    row_mean_gap_prob = gap_prob.float().mean(dim=1)
    row_sufficient = (row_mean_p_top_nonvoid <= float(tau_nonvoid)) & (row_mean_gap_prob >= float(tau_gap))

    return {
        "mean_p_top_nonvoid": float(row_mean_p_top_nonvoid.mean().item()),
        "mean_gap_prob": float(row_mean_gap_prob.mean().item()),
        "sufficient": bool(row_sufficient.all().item()),
        "row_mean_p_top_nonvoid": [float(x) for x in row_mean_p_top_nonvoid.detach().cpu().tolist()],
        "row_mean_gap_prob": [float(x) for x in row_mean_gap_prob.detach().cpu().tolist()],
        "row_sufficient": [bool(x) for x in row_sufficient.detach().cpu().tolist()],
    }


def _prepare_probe_attention_mask(attention_mask, prompt_length, candidate_length):
    if attention_mask is None or not torch.any(attention_mask == 0.0):
        return "full", None

    probe_attention_mask = torch.nn.functional.pad(attention_mask, (0, int(candidate_length)), value=1.0)
    tok_idx = probe_attention_mask.long().cumsum(-1) - 1
    tok_idx.masked_fill_(probe_attention_mask == 0, 1)
    probe_attention_mask = torch.logical_and(
        probe_attention_mask.unsqueeze(1).unsqueeze(-2),
        probe_attention_mask.unsqueeze(1).unsqueeze(-1),
    )
    return probe_attention_mask, tok_idx


@torch.no_grad()
def run_void_expand_probe(
    model,
    prompt,
    *,
    attention_mask=None,
    tokenizer=None,
    mask_id=None,
    gen_length=128,
    expand_block_length=32,
    max_length=None,
    mode="dual_tail",
    window=None,
    tau_nonvoid=None,
    tau_gap=0.0,
):
    if mode not in VOID_EXPAND_MODES:
        raise ValueError(f"void_expand_mode must be one of: {', '.join(VOID_EXPAND_MODES)}")
    if mask_id is None:
        raise ValueError("void_expand requires mask_token_id.")
    if max_length is None:
        raise ValueError("void_expand_max_length must be provided when void_expand=True.")
    if tau_nonvoid is None:
        raise ValueError("void_expand_tau_nonvoid must be provided when void_expand=True.")
    if window is not None and int(window) < 1:
        raise ValueError("void_expand_window must be >= 1.")
    if not 0.0 <= float(tau_nonvoid) <= 1.0:
        raise ValueError("void_expand_tau_nonvoid must be in [0, 1].")

    void_token_id = resolve_void_token_id(tokenizer)
    effective_window = int(window) if window is not None else max(8, int(expand_block_length) // 4)
    lengths = get_void_expand_candidate_lengths(gen_length, expand_block_length, max_length)
    candidates = []
    selected = lengths[-1]
    stop_reason = "max_length"

    for length in lengths:
        probe_x = torch.full(
            (prompt.shape[0], prompt.shape[1] + int(length)),
            int(mask_id),
            dtype=torch.long,
            device=prompt.device,
        )
        probe_x[:, :prompt.shape[1]] = prompt.clone()
        probe_attention_mask, tok_idx = _prepare_probe_attention_mask(attention_mask, prompt.shape[1], int(length))
        probe_logits = model(probe_x, probe_attention_mask, tok_idx).logits
        probe_logits = torch.cat([probe_logits[:, :1], probe_logits[:, :-1]], dim=1)
        response_logits = probe_logits[:, prompt.shape[1]:prompt.shape[1] + int(length), :]
        candidate = _score_void_expand_dual_tail(
            response_logits,
            void_token_id=void_token_id,
            window=effective_window,
            tau_nonvoid=tau_nonvoid,
            tau_gap=tau_gap,
        )
        candidate["gen_length"] = int(length)
        candidates.append(candidate)
        if candidate["sufficient"]:
            selected = int(length)
            stop_reason = f"sufficient_{mode}"
            break

    return {
        "void_expand_enabled": True,
        "void_expand_original_gen_length": int(gen_length),
        "void_expand_selected_gen_length": int(selected),
        "void_expand_max_length": int(max_length),
        "void_expand_block_length": int(expand_block_length),
        "void_expand_mode": mode,
        "void_expand_window": int(effective_window),
        "void_expand_tau_nonvoid": float(tau_nonvoid),
        "void_expand_tau_gap": float(tau_gap),
        "void_expand_void_token_id": int(void_token_id),
        "void_expand_probe_steps": int(len(candidates)),
        "void_expand_stop_reason": stop_reason,
        "void_expand_candidates": candidates,
    }


def print_void_expand_debug(metadata):
    candidates = metadata.get("void_expand_candidates") or []
    last = candidates[-1] if candidates else {}
    print(
        "VoidExpand: "
        f"original_gen_length={metadata.get('void_expand_original_gen_length')}, "
        f"selected={metadata.get('void_expand_selected_gen_length')}, "
        f"max={metadata.get('void_expand_max_length')}, "
        f"block={metadata.get('void_expand_block_length')}, "
        f"mode={metadata.get('void_expand_mode')}, "
        f"window={metadata.get('void_expand_window')}, "
        f"steps={metadata.get('void_expand_probe_steps')}, "
        f"stop={metadata.get('void_expand_stop_reason')}, "
        f"last_nonvoid={float(last.get('mean_p_top_nonvoid') or 0.0):.4f}, "
        f"last_gap_prob={float(last.get('mean_gap_prob') or 0.0):.4f}",
        flush=True,
    )


def maybe_apply_void_expand(
    model,
    prompt,
    *,
    attention_mask=None,
    steps,
    gen_length,
    expand_block_length,
    mask_id,
    tokenizer,
    void_expand=False,
    void_expand_max_length=None,
    void_expand_mode="dual_tail",
    void_expand_window=None,
    void_expand_tau_nonvoid=None,
    void_expand_tau_gap=0.0,
    void_expand_debug=False,
):
    if not void_expand:
        return int(gen_length), int(steps), {"void_expand_enabled": False}

    original_steps = int(steps)
    original_gen_length = int(gen_length)
    metadata = run_void_expand_probe(
        model,
        prompt,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        mask_id=mask_id,
        gen_length=original_gen_length,
        expand_block_length=expand_block_length,
        max_length=void_expand_max_length,
        mode=void_expand_mode,
        window=void_expand_window,
        tau_nonvoid=void_expand_tau_nonvoid,
        tau_gap=void_expand_tau_gap,
    )
    selected_gen_length = int(metadata["void_expand_selected_gen_length"])
    block = int(expand_block_length)
    if (
        block > 0
        and original_gen_length % block == 0
        and selected_gen_length % block == 0
        and original_steps % (original_gen_length // block) == 0
    ):
        steps = (selected_gen_length // block) * (original_steps // (original_gen_length // block))
    else:
        steps = max(1, round(selected_gen_length * original_steps / original_gen_length))
    if void_expand_debug:
        print_void_expand_debug(metadata)
    return selected_gen_length, int(steps), metadata
