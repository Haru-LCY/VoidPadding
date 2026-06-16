import torch
import torch.nn.functional as F

from .generation_common import DEFAULT_EOS_TOKEN_ID, add_gumbel_noise, _maybe_cuda_autocast


def _rho_eos_density(logits, currently_masked, eos_token_id):
    predicted_tokens = torch.argmax(logits, dim=-1)
    is_eos = currently_masked & (predicted_tokens == int(eos_token_id))
    mask_count = currently_masked.sum(dim=1)
    eos_count = is_eos.sum(dim=1)
    non_eos_count = mask_count - eos_count

    density = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
    nonzero = mask_count > 0
    density[nonzero] = eos_count[nonzero].float() / mask_count[nonzero].float()

    return density, eos_count.long(), non_eos_count.long()


def _rho_eos_factor(
    density,
    expansion_factor,
    scheduler,
    low_density_threshold,
    high_density_threshold,
    density_interval=0.1,
):
    if float(density_interval) <= 0:
        raise ValueError("density_interval must be > 0.")

    scheduler = str(scheduler).strip().lower()
    action = torch.zeros(density.shape[0], dtype=torch.long, device=density.device)
    action = torch.where(density > float(high_density_threshold), torch.ones_like(action), action)
    action = torch.where(density < float(low_density_threshold), torch.full_like(action, 2), action)

    gap = torch.zeros_like(density)
    expand_mask = action == 2
    contract_mask = action == 1
    if expand_mask.any():
        gap[expand_mask] = (float(low_density_threshold) - density[expand_mask]).clamp_min(0.0)
    if contract_mask.any():
        gap[contract_mask] = (density[contract_mask] - float(high_density_threshold)).clamp_min(0.0)

    segment = torch.zeros(density.shape[0], dtype=torch.long, device=density.device)
    outside = action != 0
    if outside.any():
        segment[outside] = torch.ceil((gap[outside] + 1e-12) / float(density_interval)).long()

    if scheduler == "const":
        multiplier = torch.ones_like(segment)
    elif scheduler == "linear":
        multiplier = torch.clamp(segment, min=1)
    elif scheduler == "exp":
        multiplier = torch.where(segment > 0, (2 ** (segment - 1)).long(), torch.zeros_like(segment))
        multiplier = torch.clamp(multiplier, min=1)
    else:
        raise ValueError("scheduler must be one of: const, linear, exp.")

    multiplier = torch.clamp(multiplier, max=8)
    factor = multiplier * int(expansion_factor)
    factor = torch.where(action == 0, torch.zeros_like(factor), factor)
    return factor.long(), action


def _rho_eos_adjust_length(
    x,
    gen_lengths,
    prompt_length,
    density,
    expansion_factor,
    max_gen_length,
    scheduler,
    low_density_threshold,
    high_density_threshold,
    eos_token_id,
    mask_id,
):
    factor, action = _rho_eos_factor(
        density=density,
        expansion_factor=expansion_factor,
        scheduler=scheduler,
        low_density_threshold=low_density_threshold,
        high_density_threshold=high_density_threshold,
    )

    new_gen_lengths = gen_lengths.clone()
    expand_mask = action == 2
    contract_mask = action == 1
    if expand_mask.any():
        new_gen_lengths[expand_mask] = torch.clamp(
            gen_lengths[expand_mask] + factor[expand_mask],
            max=int(max_gen_length),
        )
    if contract_mask.any():
        new_gen_lengths[contract_mask] = torch.clamp(
            gen_lengths[contract_mask] - factor[contract_mask],
            min=0,
        )
    if (action == 0).all():
        return x, gen_lengths, action

    batch_size = int(x.shape[0])
    new_max_total_length = int(prompt_length) + int(new_gen_lengths.max().item())
    new_x = torch.full(
        (batch_size, new_max_total_length),
        int(eos_token_id),
        dtype=torch.long,
        device=x.device,
    )
    new_x[:, :int(prompt_length)] = x[:, :int(prompt_length)]

    for row in range(batch_size):
        old_total_length = int(prompt_length) + int(gen_lengths[row].item())
        new_total_length = int(prompt_length) + int(new_gen_lengths[row].item())
        row_action = int(action[row].item())

        if row_action == 0:
            new_x[row, :old_total_length] = x[row, :old_total_length]
        elif row_action == 1:
            new_x[row, :new_total_length] = x[row, :new_total_length]
        elif row_action == 2:
            added = int(new_gen_lengths[row].item()) - int(gen_lengths[row].item())
            generation_region = x[row, int(prompt_length):old_total_length]
            eos_positions = (generation_region == int(eos_token_id)).nonzero(as_tuple=True)[0]
            if eos_positions.numel() > 0:
                insert_pos = int(prompt_length) + int(eos_positions[0].item())
                new_x[row, :insert_pos] = x[row, :insert_pos]
                new_x[row, insert_pos:insert_pos + added] = int(mask_id)
                remaining = old_total_length - insert_pos
                new_x[row, insert_pos + added:insert_pos + added + remaining] = x[row, insert_pos:old_total_length]
            else:
                new_x[row, :old_total_length] = x[row, :old_total_length]
                new_x[row, old_total_length:new_total_length] = int(mask_id)

    return new_x, new_gen_lengths, action


@torch.no_grad()
def generate_rho_eos(
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
    low_density_threshold=0.4,
    high_density_threshold=0.6,
    scheduler="exp",
    max_adjust_length_steps=64,
    return_metadata=False,
    debug=False,
):
    if eos_token_id is None:
        raise ValueError("generate_rho_eos requires eos_token_id.")
    if int(initial_gen_length) < 1:
        raise ValueError("initial_gen_length must be >= 1.")
    if int(max_gen_length) < int(initial_gen_length):
        raise ValueError("max_gen_length must be >= initial_gen_length.")
    if int(block_length) < 1:
        raise ValueError("block_length must be >= 1.")
    if int(expansion_factor) < 1:
        raise ValueError("expansion_factor must be >= 1.")
    if float(low_density_threshold) > float(high_density_threshold):
        raise ValueError("low_density_threshold must be <= high_density_threshold.")

    batch_size = int(prompt.shape[0])
    prompt_length = int(prompt.shape[1])
    device = prompt.device
    if batch_size > 1:
        outputs = []
        metadatas = []
        total_nfe = 0
        output_lengths = []
        for row in range(batch_size):
            row_output, row_nfe, row_metadata = generate_rho_eos(
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
                low_density_threshold=low_density_threshold,
                high_density_threshold=high_density_threshold,
                scheduler=scheduler,
                max_adjust_length_steps=max_adjust_length_steps,
                return_metadata=True,
                debug=debug,
            )
            outputs.append(row_output[0])
            metadatas.append(row_metadata)
            total_nfe += int(row_nfe)
            output_lengths.append(int(row_metadata["rho_eos_output_lengths"][0]))

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
            "rho_eos_enabled": True,
            "rho_eos_initial_gen_length": int(initial_gen_length),
            "rho_eos_max_gen_length": int(max_gen_length),
            "rho_eos_block_length": int(block_length),
            "rho_eos_expansion_factor": int(expansion_factor),
            "rho_eos_low_density_threshold": float(low_density_threshold),
            "rho_eos_high_density_threshold": float(high_density_threshold),
            "rho_eos_scheduler": str(scheduler),
            "rho_eos_model_calls": sum(int(meta.get("rho_eos_model_calls", 0)) for meta in metadatas),
            "rho_eos_expansions": sum(int(meta.get("rho_eos_expansions", 0)) for meta in metadatas),
            "rho_eos_contractions": sum(int(meta.get("rho_eos_contractions", 0)) for meta in metadatas),
            "rho_eos_stopped_stagnant": any(bool(meta.get("rho_eos_stopped_stagnant", False)) for meta in metadatas),
            "rho_eos_output_lengths": output_lengths,
            "rho_eos_final_gen_lengths": output_lengths,
            "rho_eos_nfe": int(total_nfe),
            "rho_eos_unbatched_rows": batch_size,
            "rho_eos_row_metadata": metadatas,
        }
        if return_metadata:
            return final_x, total_nfe, metadata
        return final_x, total_nfe

    metadata = {
        "rho_eos_enabled": True,
        "rho_eos_initial_gen_length": int(initial_gen_length),
        "rho_eos_max_gen_length": int(max_gen_length),
        "rho_eos_block_length": int(block_length),
        "rho_eos_expansion_factor": int(expansion_factor),
        "rho_eos_low_density_threshold": float(low_density_threshold),
        "rho_eos_high_density_threshold": float(high_density_threshold),
        "rho_eos_scheduler": str(scheduler),
        "rho_eos_model_calls": 0,
        "rho_eos_expansions": 0,
        "rho_eos_contractions": 0,
        "rho_eos_keep_actions": 0,
        "rho_eos_stopped_stagnant": False,
    }

    nfe = 0
    gen_lengths = torch.full((batch_size,), int(initial_gen_length), dtype=torch.long, device=device)
    x = torch.full(
        (batch_size, prompt_length + int(initial_gen_length)),
        int(mask_id),
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_length] = prompt.clone()
    prompt_index = x != int(mask_id)
    current_pos = torch.full((batch_size,), prompt_length, dtype=torch.long, device=device)
    denoise_only_mode = torch.zeros(batch_size, dtype=torch.bool, device=device)
    stop_reason = "completed"
    step = 0

    with _maybe_cuda_autocast(device):
        while (current_pos < prompt_length + gen_lengths).any():
            step += 1
            if step > int(max_gen_length):
                stop_reason = "max_decode_steps"
                break

            total_lengths = prompt_length + gen_lengths
            x_before_step = x.clone()
            old_shape = x.shape

            denoise_only_mode |= gen_lengths >= int(max_gen_length)
            if step > int(max_adjust_length_steps):
                denoise_only_mode[:] = True

            arange_tensor = torch.arange(x.shape[1], device=device).expand(batch_size, -1)
            attention_mask = (arange_tensor < total_lengths.unsqueeze(1)).long()
            if float(cfg_scale) > 0.0:
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
            metadata["rho_eos_model_calls"] += 1

            predicted_tokens = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            confidences = F.softmax(logits.float(), dim=-1)
            predicted_confidences = torch.gather(
                confidences, dim=-1, index=predicted_tokens.unsqueeze(-1)
            ).squeeze(-1)

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

            x[high_conf_indices] = predicted_tokens[high_conf_indices]

            for row in range(batch_size):
                total_len = prompt_length + int(gen_lengths[row].item())
                while int(current_pos[row].item()) < total_len:
                    start = int(current_pos[row].item())
                    end = min(start + int(block_length), total_len)
                    if start == end:
                        break
                    if not (x[row, start:end] == int(mask_id)).any():
                        current_pos[row] = end
                    else:
                        break

            density, eos_count, non_eos_count = _rho_eos_density(
                logits=logits,
                currently_masked=currently_masked,
                eos_token_id=eos_token_id,
            )
            density = torch.where(
                denoise_only_mode,
                torch.clamp(
                    density,
                    min=float(low_density_threshold),
                    max=float(high_density_threshold),
                ),
                density,
            )

            new_x, new_gen_lengths, action = _rho_eos_adjust_length(
                x=x,
                gen_lengths=gen_lengths,
                prompt_length=prompt_length,
                density=density,
                expansion_factor=expansion_factor,
                max_gen_length=max_gen_length,
                scheduler=scheduler,
                low_density_threshold=low_density_threshold,
                high_density_threshold=high_density_threshold,
                eos_token_id=eos_token_id,
                mask_id=mask_id,
            )
            metadata["rho_eos_expansions"] += int((action == 2).sum().item())
            metadata["rho_eos_contractions"] += int((action == 1).sum().item())
            metadata["rho_eos_keep_actions"] += int((action == 0).sum().item())
            metadata["rho_eos_last_density"] = [float(v) for v in density.detach().cpu().tolist()]
            metadata["rho_eos_last_eos_count"] = [int(v) for v in eos_count.detach().cpu().tolist()]
            metadata["rho_eos_last_non_eos_count"] = [int(v) for v in non_eos_count.detach().cpu().tolist()]

            if debug:
                print(
                    "rho-EOS: "
                    f"step={step} density={metadata['rho_eos_last_density']} "
                    f"action={[int(v) for v in action.detach().cpu().tolist()]} "
                    f"gen={[int(v) for v in gen_lengths.detach().cpu().tolist()]} -> "
                    f"{[int(v) for v in new_gen_lengths.detach().cpu().tolist()]}",
                    flush=True,
                )

            if (
                new_x.shape != x.shape
                or not torch.equal(new_gen_lengths, gen_lengths)
                or not torch.equal(new_x, x)
            ):
                x = new_x
                gen_lengths = new_gen_lengths
                prompt_index = x != int(mask_id)
                current_pos = torch.min(current_pos, prompt_length + gen_lengths)

            if torch.equal(x, x_before_step) and x.shape == old_shape:
                metadata["rho_eos_stopped_stagnant"] = True
                stop_reason = "stagnant"
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

    metadata["rho_eos_output_lengths"] = output_lengths
    metadata["rho_eos_final_gen_lengths"] = output_lengths
    metadata["rho_eos_nfe"] = int(nfe)
    metadata["rho_eos_stop_reason"] = stop_reason
    if return_metadata:
        return final_x, nfe, metadata
    return final_x, nfe

