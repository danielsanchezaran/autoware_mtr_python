import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Set random seed value.

    Args:
    ----
        seed (int): Seed value.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
