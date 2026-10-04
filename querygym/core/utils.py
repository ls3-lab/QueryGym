from __future__ import annotations
import random
import re
from typing import Optional, Tuple

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_trace(text: str) -> Tuple[str, bool]:
    """Remove <think>...</think> reasoning traces from model output.

    Args:
        text: Raw model output

    Returns:
        (text without reasoning traces, whether an unclosed <think> was found).
        When a <think> block is never closed (generation ran out of tokens while
        reasoning), everything from the opening tag on is dropped. When only the
        closing tag is present (chat templates that put <think> in the prompt),
        everything up to it is dropped.
    """
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    if "<think>" in text:
        return text.split("<think>", 1)[0], True
    return text, False


def seed_everything(seed: Optional[int] = 42):
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
