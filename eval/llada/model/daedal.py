import torch
import torch.nn.functional as F

from .generation_common import DEFAULT_EOS_TOKEN_ID, add_gumbel_noise, _maybe_cuda_autocast


def _daedal_eos_confidence(logits, total_lengths, prompt_length, eos_token_id, eos_check_tokens):
    confidences = F.softmax(logits.float(), dim=-1)
    predicted_tokens = torch.argmax(logits, dim=-1)
    batch_confidences = []
    for row in range(logits.shape[0]):
        eos_confidences = []
        for pos in range(int(total_lengths[row].item()) - 1, int(prompt_length) - 1, -1):
            if len(eos_confidences) >= int(eos_check_tokens):
                break
            if int(predicted_tokens[row, pos].item()) == int(eos_token_id):
                eos_confidences.append(float(confidences[row, pos, int(eos_token_id)].item()))
        batch_confidences.append(sum(eos_confidences) / float(eos_check_tokens))
    return torch.tensor(batch_confidences, dtype=torch.float32, device=logits.device)


@torch.no_grad()
def generate_daedal(
    model,
    prompt,
    initial_gen_length=64,
    max_gen_length=2048,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    high_conf_threshold=0.90,
    low_conf_threshold=0.10,
    expansion_factor=8,
    mask_id=126336,
    eos_token_id=DEFAULT_EOS_TOKEN_ID,
    eos_confidence_threshold=0.5,
    expand_eos_confidence_threshold=0.9,
    eos_check_tokens=32,
    return_metadata=False,
    debug=False,
):
    if eos_token_id is None:
        raise ValueError("generate_daedal requires eos_token_id.")
    if int(initial_gen_length) < 1:
        raise ValueError("initial_gen_length must be >= 1.")
    if int(max_gen_length) < int(initial_gen_length):
        raise ValueError("max_gen_length must be >= initial_gen_length.")
    if int(block_length) < 1:
        raise ValueError("block_length must be >= 1.")
    if int(expansion_factor) < 1:
        raise ValueError("expansion_factor must be >= 1.")
    if int(eos_check_tokens) < 1:
        raise ValueError("eos_check_tokens must be >= 1.")

    batch_size = int(prompt.shape[0])
    prompt_length = int(prompt.shape[1])
    device = prompt.device
    if batch_size > 1:
        outputs = []
        metadatas = []
        total_nfe = 0
        output_lengths = []
        for row in range(batch_size):
            row_output, row_nfe, row_metadata = generate_daedal(
                model,
                prompt[row:row + 1],
                initial_gen_length=initial_gen_length,
                max_gen_length=max_gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                high_conf_threshold=high_conf_threshold,
                low_conf_threshold=low_conf_threshold,
                expansion_factor=expansion_factor,
                mask_id=mask_id,
                eos_token_id=eos_token_id,
                eos_confidence_threshold=eos_confidence_threshold,
                expand_eos_confidence_threshold=expand_eos_confidence_threshold,
                eos_check_tokens=eos_check_tokens,
                return_metadata=True,
                debug=debug,
            )
            outputs.append(row_output[0])
            metadatas.append(row_metadata)
            total_nfe += int(row_nfe)
            output_lengths.append(int(row_metadata["daedal_output_lengths"][0]))

        max_total_length = max(output.shape[0] for output in outputs)
        final_x = torch.full(
            (batch_size, max_total_length),
            int(eos_token_id),
            dtype=torch.long,
            device=device,
        )
        for row, output in enumerate(outputs):
            final_x[row, :output.shape[0]] = output

        metadata = {
            "daedal_enabled": True,
            "daedal_initial_gen_length": int(initial_gen_length),
            "daedal_max_gen_length": int(max_gen_length),
            "daedal_block_length": int(block_length),
            "daedal_expansion_factor": int(expansion_factor),
            "daedal_stage1_probe_steps": sum(
                int(meta.get("daedal_stage1_probe_steps", 0)) for meta in metadatas
            ),
            "daedal_stage2_model_calls": sum(
                int(meta.get("daedal_stage2_model_calls", 0)) for meta in metadatas
            ),
            "daedal_stage2_insertions": sum(
                int(meta.get("daedal_stage2_insertions", 0)) for meta in metadatas
            ),
            "daedal_stopped_stagnant": any(
                bool(meta.get("daedal_stopped_stagnant", False)) for meta in metadatas
            ),
            "daedal_output_lengths": output_lengths,
            "daedal_final_gen_lengths": output_lengths,
            "daedal_nfe": int(total_nfe),
            "daedal_unbatched_rows": batch_size,
            "daedal_row_metadata": metadatas,
        }
        if return_metadata:
            return final_x, total_nfe, metadata
        return final_x, total_nfe

    nfe = 0
    metadata = {
        "daedal_enabled": True,
        "daedal_initial_gen_length": int(initial_gen_length),
        "daedal_max_gen_length": int(max_gen_length),
        "daedal_block_length": int(block_length),
        "daedal_expansion_factor": int(expansion_factor),
        "daedal_stage1_probe_steps": 0,
        "daedal_stage2_model_calls": 0,
        "daedal_stage2_insertions": 0,
        "daedal_stopped_stagnant": False,
    }

    gen_lengths = torch.full((batch_size,), int(initial_gen_length), dtype=torch.long, device=device)
    x = torch.full(
        (batch_size, prompt_length + int(initial_gen_length)),
        int(mask_id),
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_length] = prompt.clone()

    with _maybe_cuda_autocast(device):
        # Stage 1: expand the initial masked canvas until EOS confidence is high enough.
        while True:
            total_lengths = prompt_length + gen_lengths
            arange_tensor = torch.arange(x.shape[1], device=device).expand(batch_size, -1)
            attention_mask = (arange_tensor < total_lengths.unsqueeze(1)).long()
            logits = model(x, attention_mask=attention_mask).logits
            nfe += 1
            metadata["daedal_stage1_probe_steps"] += 1

            eos_confidences = _daedal_eos_confidence(
                logits, total_lengths, prompt_length, eos_token_id, eos_check_tokens
            )
            should_expand = (eos_confidences < float(eos_confidence_threshold)) & (
                gen_lengths < int(max_gen_length)
            )
            if not should_expand.any():
                metadata["daedal_stage1_stop_reason"] = "eos_confidence_or_max_length"
                break

            new_gen_lengths = gen_lengths.clone()
            new_gen_lengths[should_expand] = torch.clamp(
                gen_lengths[should_expand] + int(expansion_factor),
                max=int(max_gen_length),
            )
            if int(new_gen_lengths.max().item()) <= int(gen_lengths.max().item()):
                metadata["daedal_stage1_stop_reason"] = "max_length"
                break

            new_x = torch.full(
                (batch_size, prompt_length + int(new_gen_lengths.max().item())),
                int(eos_token_id),
                dtype=torch.long,
                device=device,
            )
            for row in range(batch_size):
                old_total_len = prompt_length + int(gen_lengths[row].item())
                new_x[row, :old_total_len] = x[row, :old_total_len]
                if bool(should_expand[row].item()):
                    new_total_len = prompt_length + int(new_gen_lengths[row].item())
                    new_x[row, old_total_len:new_total_len] = int(mask_id)
            x = new_x
            gen_lengths = new_gen_lengths

        stage1_lengths = gen_lengths.clone()
        gen_lengths = torch.clamp(gen_lengths + int(eos_check_tokens) // 2, max=int(max_gen_length))
        new_x = torch.full(
            (batch_size, prompt_length + int(gen_lengths.max().item())),
            int(eos_token_id),
            dtype=torch.long,
            device=device,
        )
        for row in range(batch_size):
            new_x[row, :prompt_length] = x[row, :prompt_length]
            new_x[row, prompt_length:prompt_length + int(stage1_lengths[row].item())] = int(mask_id)
        x = new_x
        metadata["daedal_stage1_selected_gen_lengths"] = [
            int(v) for v in gen_lengths.detach().cpu().tolist()
        ]

        # Stage 2: denoise the current block and insert masks at low-confidence positions.
        current_pos = torch.full((batch_size,), prompt_length, dtype=torch.long, device=device)
        denoise_only_mode = torch.zeros(batch_size, dtype=torch.bool, device=device)

        while (current_pos < prompt_length + gen_lengths).any():
            total_lengths = prompt_length + gen_lengths
            x_before_step = x.clone()

            for row in range(batch_size):
                if gen_lengths[row] >= int(max_gen_length) and not denoise_only_mode[row]:
                    if current_pos[row] < total_lengths[row]:
                        denoise_only_mode[row] = True

            arange_tensor = torch.arange(x.shape[1], device=device).expand(batch_size, -1)
            attention_mask = (arange_tensor < total_lengths.unsqueeze(1)).long()
            if float(cfg_scale) > 0.0:
                prompt_index = torch.zeros_like(x, dtype=torch.bool)
                prompt_index[:, :prompt_length] = True
                un_x = x.clone()
                un_x[prompt_index] = int(mask_id)
                logits, un_logits = torch.chunk(
                    model(
                        torch.cat([x, un_x], dim=0),
                        attention_mask=torch.cat([attention_mask, attention_mask], dim=0),
                    ).logits,
                    2,
                    dim=0,
                )
                logits = un_logits + (float(cfg_scale) + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits
            nfe += 1
            metadata["daedal_stage2_model_calls"] += 1

            predicted_tokens = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            confidences = F.softmax(logits.float(), dim=-1)
            predicted_confidences = torch.gather(
                confidences, dim=-1, index=predicted_tokens.unsqueeze(-1)
            ).squeeze(-1)
            eos_confidences = _daedal_eos_confidence(
                logits, total_lengths, prompt_length, eos_token_id, eos_check_tokens
            )

            block_mask = torch.zeros_like(x, dtype=torch.bool, device=device)
            for row in range(batch_size):
                if current_pos[row] >= total_lengths[row]:
                    continue
                start_idx = int(current_pos[row].item())
                end_idx = min(start_idx + int(block_length), int(total_lengths[row].item()))
                block_mask[row, start_idx:end_idx] = True

            currently_masked = x == int(mask_id)
            high_conf_indices = (
                (predicted_confidences > float(high_conf_threshold))
                & block_mask
                & currently_masked
                & (predicted_tokens != int(mask_id))
            )

            for row in range(batch_size):
                if current_pos[row] >= total_lengths[row]:
                    continue
                start_idx = int(current_pos[row].item())
                end_idx = min(start_idx + int(block_length), int(total_lengths[row].item()))
                if high_conf_indices[row, start_idx:end_idx].any():
                    continue
                valid_fallback_mask = block_mask[row] & currently_masked[row]
                if not valid_fallback_mask.any():
                    continue
                candidate_indices = torch.where(valid_fallback_mask)[0]
                candidate_confs = predicted_confidences[row, candidate_indices]
                candidate_tokens = predicted_tokens[row, candidate_indices]
                _, sort_indices = torch.sort(candidate_confs, descending=True)
                best_idx = None
                for sorted_idx in sort_indices:
                    if int(candidate_tokens[sorted_idx].item()) != int(mask_id):
                        best_idx = candidate_indices[sorted_idx]
                        break
                if best_idx is not None:
                    high_conf_indices[row, best_idx] = True
                else:
                    stuck_logits = logits[row, candidate_indices].clone()
                    stuck_logits[:, int(mask_id)] = -torch.inf
                    new_confidences = F.softmax(stuck_logits.float(), dim=-1)
                    new_best_confs, new_best_tokens = torch.max(new_confidences, dim=-1)
                    best_local_idx = torch.argmax(new_best_confs)
                    pos_to_fill = candidate_indices[best_local_idx]
                    predicted_tokens[row, pos_to_fill] = new_best_tokens[best_local_idx]
                    high_conf_indices[row, pos_to_fill] = True

            potential_expand_mask = (
                (predicted_confidences < float(low_conf_threshold))
                & block_mask
                & currently_masked
                & (~high_conf_indices)
            )
            expand_indices = torch.zeros_like(x, dtype=torch.bool, device=device)
            for row in range(batch_size):
                if eos_confidences[row] >= float(expand_eos_confidence_threshold):
                    continue
                if gen_lengths[row] >= int(max_gen_length) or denoise_only_mode[row]:
                    continue
                if current_pos[row] >= total_lengths[row]:
                    continue
                candidates = torch.where(potential_expand_mask[row])[0]
                if candidates.numel() == 0:
                    continue
                candidate_confs = predicted_confidences[row, candidates]
                _, lowest_idx = torch.topk(candidate_confs, 1, largest=False)
                expand_indices[row, candidates[lowest_idx]] = True

            x[high_conf_indices] = predicted_tokens[high_conf_indices]
            if expand_indices.any():
                new_gen_lengths = gen_lengths.clone()
                for row in range(batch_size):
                    expansion_count = int(expand_indices[row].sum().item())
                    if expansion_count > 0:
                        new_len = int(gen_lengths[row].item()) + expansion_count * (int(expansion_factor) - 1)
                        new_gen_lengths[row] = min(new_len, int(max_gen_length))
                        metadata["daedal_stage2_insertions"] += expansion_count

                new_x = torch.full(
                    (batch_size, prompt_length + int(new_gen_lengths.max().item())),
                    int(eos_token_id),
                    dtype=torch.long,
                    device=device,
                )
                packed_lengths = torch.zeros_like(gen_lengths)
                for row in range(batch_size):
                    if not expand_indices[row].any():
                        total_len = prompt_length + int(gen_lengths[row].item())
                        new_x[row, :total_len] = x[row, :total_len]
                        packed_lengths[row] = gen_lengths[row]
                        continue
                    write_ptr = prompt_length
                    new_x[row, :prompt_length] = x[row, :prompt_length]
                    for col in range(prompt_length, prompt_length + int(gen_lengths[row].item())):
                        if write_ptr >= new_x.shape[1]:
                            break
                        if expand_indices[row, col]:
                            end_write = min(write_ptr + int(expansion_factor), new_x.shape[1])
                            new_x[row, write_ptr:end_write] = int(mask_id)
                            write_ptr = end_write
                        else:
                            new_x[row, write_ptr] = x[row, col]
                            write_ptr += 1
                    packed_lengths[row] = write_ptr - prompt_length
                x = new_x
                gen_lengths = packed_lengths

            for row in range(batch_size):
                total_len = prompt_length + int(gen_lengths[row].item())
                while int(current_pos[row].item()) < total_len:
                    start = int(current_pos[row].item())
                    end = min(start + int(block_length), total_len)
                    if start == end:
                        break
                    if not (x[row, start:end] == int(mask_id)).any():
                        current_pos[row] = start + int(block_length)
                    else:
                        break

            if torch.equal(x, x_before_step):
                metadata["daedal_stopped_stagnant"] = True
                break

    output_lengths = [int(v) for v in gen_lengths.detach().cpu().tolist()]
    final_x = torch.full(
        (batch_size, prompt_length + max(output_lengths)),
        int(eos_token_id),
        dtype=torch.long,
        device=device,
    )
    for row, output_length in enumerate(output_lengths):
        final_x[row, :prompt_length + output_length] = x[row, :prompt_length + output_length]

    metadata["daedal_output_lengths"] = output_lengths
    metadata["daedal_final_gen_lengths"] = output_lengths
    metadata["daedal_nfe"] = int(nfe)
    if debug:
        print(
            "DAEDAL: "
            f"stage1_probes={metadata['daedal_stage1_probe_steps']} "
            f"stage2_calls={metadata['daedal_stage2_model_calls']} "
            f"insertions={metadata['daedal_stage2_insertions']} "
            f"output_lengths={output_lengths}",
            flush=True,
        )
    if return_metadata:
        return final_x, nfe, metadata
    return final_x, nfe
