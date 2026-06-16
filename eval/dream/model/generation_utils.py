import time
import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)
try:
    from model.void_expand import maybe_apply_void_expand
except ImportError:
    from .void_expand import maybe_apply_void_expand

logger = logging.get_logger(__name__)

DEFAULT_EOS_TOKEN_ID = 151643
DEFAULT_VOID_TOKEN_ID = 56940


def _parse_ban_tokens(ban_tokens):
    if ban_tokens is None:
        return ()
    raw_tokens = []
    if isinstance(ban_tokens, str):
        raw_tokens.extend(ban_tokens.split(","))
    else:
        for item in ban_tokens:
            raw_tokens.extend(str(item).split(","))

    parsed = []
    for raw_token in raw_tokens:
        token = raw_token.strip().lower().replace("-", "_")
        if not token:
            continue
        if token != "void":
            raise ValueError("ban_tokens only supports 'void' for Dream generation.")
        if token not in parsed:
            parsed.append(token)
    return tuple(parsed)


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


def resolve_ban_token_ids(ban_tokens=None, tokenizer=None, eos_token_id=DEFAULT_EOS_TOKEN_ID):
    names = _parse_ban_tokens(ban_tokens)
    if not names:
        return ()

    token_ids = []
    for name in names:
        if name == "void":
            void_id = _convert_token_to_id(tokenizer, "<pad_0>")
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


def _mask_after_cut_boundary(mask_index, cut_boundaries, start_pos=0):
    if not cut_boundaries:
        return mask_index
    mask_index = mask_index.clone()
    for row, boundary in enumerate(cut_boundaries):
        if boundary is not None:
            rel_boundary = int(boundary) - int(start_pos)
            if rel_boundary < mask_index.shape[1]:
                mask_index[row, max(0, rel_boundary):] = False
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


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False, ban_token_ids=None):

    if temperature > 0:
        logits = logits / temperature
    logits = apply_ban_token_ids(logits, ban_token_ids)
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    metadata: Optional[Dict[str, Any]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.
        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )
        threshold = kwargs.get("threshold", 0.9)
        block_length = kwargs.get("block_length", kwargs.get("block_size", None))
        ban_tokens = kwargs.get("ban_tokens", None)
        cut = kwargs.get("cut", False)
        tokenizer = kwargs.get("tokenizer", None)
        eos_token_id = kwargs.get("eos_token_id", None)
        void_expand = kwargs.get("void_expand", False)
        void_expand_max_length = kwargs.get("void_expand_max_length", None)
        void_expand_block_length = kwargs.get("void_expand_block_length", 32)
        void_expand_mode = kwargs.get("void_expand_mode", "dual_tail")
        void_expand_window = kwargs.get("void_expand_window", None)
        void_expand_tau_nonvoid = kwargs.get("void_expand_tau_nonvoid", None)
        void_expand_tau_gap = kwargs.get("void_expand_tau_gap", 0.0)
        void_expand_debug = kwargs.get("void_expand_debug", False)

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
            threshold=threshold,
            block_length=block_length,
            ban_tokens=ban_tokens,
            cut=cut,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            void_expand=void_expand,
            void_expand_max_length=void_expand_max_length,
            void_expand_block_length=void_expand_block_length,
            void_expand_mode=void_expand_mode,
            void_expand_window=void_expand_window,
            void_expand_tau_nonvoid=void_expand_tau_nonvoid,
            void_expand_tau_gap=void_expand_tau_gap,
            void_expand_debug=void_expand_debug,
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        threshold: Optional[float] = 0.9,
        block_length: Optional[int] = None,
        ban_tokens=None,
        cut: bool = False,
        tokenizer=None,
        eos_token_id=None,
        void_expand: bool = False,
        void_expand_max_length=None,
        void_expand_block_length=None,
        void_expand_mode: str = "dual_tail",
        void_expand_window=None,
        void_expand_tau_nonvoid=None,
        void_expand_tau_gap: float = 0.0,
        void_expand_debug: bool = False,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        if eos_token_id is None:
            eos_token_id = generation_config.eos_token_id
        if eos_token_id is None:
            eos_token_id = DEFAULT_EOS_TOKEN_ID
        ban_token_ids = resolve_ban_token_ids(ban_tokens, tokenizer=tokenizer, eos_token_id=eos_token_id)
        prompt_length = input_ids.shape[1]
        gen_length = max_length - prompt_length
        if void_expand_block_length is None:
            void_expand_block_length = 32
        gen_length, steps, void_expand_metadata = maybe_apply_void_expand(
            self,
            input_ids,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=gen_length,
            expand_block_length=void_expand_block_length,
            mask_id=mask_token_id,
            tokenizer=tokenizer,
            void_expand=void_expand,
            void_expand_max_length=void_expand_max_length,
            void_expand_mode=void_expand_mode,
            void_expand_window=void_expand_window,
            void_expand_tau_nonvoid=void_expand_tau_nonvoid,
            void_expand_tau_gap=void_expand_tau_gap,
            void_expand_debug=void_expand_debug,
        )
        max_length = prompt_length + gen_length
        cut_boundaries = [None] * input_ids.shape[0]

        histories = [] if (return_dict_in_generate and output_history) else None
        start_time = time.time()
        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        if block_length is None:
            block_length = gen_length
        block_length = int(block_length)
        if block_length <= 0:
            raise ValueError("block_length must be a positive integer.")
        if gen_length % block_length != 0:
            raise ValueError(f"gen_length ({gen_length}) must be divisible by block_length ({block_length}).")
        num_blocks = gen_length // block_length
        if steps % num_blocks != 0:
            raise ValueError(f"steps ({steps}) must be divisible by num_blocks ({num_blocks}).")
        steps_per_block = steps // num_blocks
        timesteps = torch.linspace(1, eps, steps_per_block + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        used_steps = 0
        sample_used_steps = torch.zeros(input_ids.shape[0], device=x.device, dtype=torch.long)
        for num_block in range(num_blocks):
            current_block_start = prompt_length + num_block * block_length
            current_block_end = current_block_start + block_length
            block_steps = steps_per_block
            i = 0
            if alg == 'confidence_threshold':
                mask_index = (x[:, current_block_start:current_block_end] == mask_token_id)
                assert mask_index.sum() % block_steps == 0, "mask_index.sum() must be divisible by steps_per_block"
                assert x.shape[0] == 1, "batch size must be 1"

                number_transfer_tokens = mask_index.sum().item() // block_steps
                left_tokens_last_step = 0
            while i < block_steps:
                mask_index = torch.zeros_like(x, device=x.device, dtype=torch.bool)
                mask_index[:, current_block_start:current_block_end] = (
                    x[:, current_block_start:current_block_end] == mask_token_id
                )
                if cut:
                    mask_index = _mask_after_cut_boundary(mask_index, cut_boundaries)
                if mask_index.sum() == 0:
                    break

                active_rows = mask_index.view(mask_index.shape[0], -1).any(dim=1)
                logits = self(x, attention_mask, tok_idx).logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

                # this allows user-defined logits control of the intermediate steps
                logits = generation_logits_hook_func(used_steps, x, logits)
                used_steps += 1
                sample_used_steps += active_rows.long()

                mask_logits = logits[mask_index]
                if not alg == 'confidence_threshold':
                    t = timesteps[i]
                    s = timesteps[i + 1]

                if alg == 'origin':
                    p_transfer = 1 - s / t if i < block_steps - 1 else 1
                    x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                    transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                    _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k, ban_token_ids=ban_token_ids)
                    x[mask_index] = x0.clone()
                elif alg == 'confidence_threshold':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, ban_token_ids=ban_token_ids)
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_index] = confidence
                    current_transfer_tokens = number_transfer_tokens + left_tokens_last_step
                    left_tokens_last_step = 0
                    selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
                    transfer_index = torch.zeros_like(x, device=x.device, dtype=torch.bool)
                    select_index = select_index.to(x.device)
                    transfer_index[0, select_index[0]] = True
                    for k in range(1, current_transfer_tokens):
                        if selected_confidence[0, k] < threshold:
                            if i < block_steps - 1:
                                left_tokens_last_step += 1
                                transfer_index[0, select_index[0, k]] = False
                            else:
                                number_transfer_tokens = 0
                                block_steps += 1
                                left_tokens_last_step += 1
                                transfer_index[0, select_index[0, k]] = False

                    x[transfer_index] = x_[transfer_index].clone()

                else:
                    if alg == 'maskgit_plus':
                        confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, ban_token_ids=ban_token_ids)
                    elif alg == 'topk_margin':
                        confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True, ban_token_ids=ban_token_ids)
                    elif alg == 'entropy':
                        confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True, ban_token_ids=ban_token_ids)
                    else:
                        raise RuntimeError(f"Unknown alg: {alg}")
                    num_mask_token = mask_index.sum() / mask_index.shape[0]
                    number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < block_steps - 1 else int(num_mask_token)
                    full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_index] = confidence
                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                        else:
                            full_confidence = full_confidence / alg_temp
                            full_confidence = F.softmax(full_confidence, dim=-1)
                            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                        x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                        x_[mask_index] = x0.clone()
                        row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                        x[row_indices,transfer_index] = x_[row_indices,transfer_index]

                if cut:
                    new_boundaries = _find_cut_boundaries(x, prompt_length, gen_length, mask_token_id, eos_token_id)
                    cut_boundaries = [old if old is not None else new for old, new in zip(cut_boundaries, new_boundaries)]
                    x = _fill_after_cut_boundaries(x, prompt_length, gen_length, cut_boundaries, eos_token_id)
                    if all(boundary is not None for boundary in cut_boundaries):
                        x = _finalize_cut_output(x, prompt_length, gen_length, cut_boundaries, eos_token_id)
                        break

                # this allows user-defined token control of the intermediate steps
                x = generation_tokens_hook_func(used_steps, x, logits)

                if histories is not None:
                    histories.append(x.clone())
                i += 1

            if cut and all(boundary is not None for boundary in cut_boundaries):
                break
        
        print(f'used steps: {used_steps}')
        end_time = time.time()
        print(f'used time: {end_time - start_time}')
        if cut:
            x = _finalize_cut_output(x, prompt_length, gen_length, cut_boundaries, eos_token_id)
        if return_dict_in_generate:
            if void_expand_metadata is None:
                void_expand_metadata = {}
            void_expand_metadata["effective_nfe"] = int(used_steps)
            void_expand_metadata["configured_nfe"] = int(steps)
            void_expand_metadata["sample_effective_nfe"] = [int(v) for v in sample_used_steps.detach().cpu().tolist()]
            return DreamModelOutput(
                sequences=x,
                history=histories,
                metadata=void_expand_metadata,
            )
        else:
            return x
