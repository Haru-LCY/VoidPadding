import torch
from contextlib import nullcontext

DEFAULT_EOS_TOKEN_ID = 126081


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise



def _maybe_cuda_autocast(device):
    if torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()
