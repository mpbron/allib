from typing import Optional

import numpy as np

def get_random_generator(
        rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    if rng is not None:
        return rng
    return np.random.default_rng()
