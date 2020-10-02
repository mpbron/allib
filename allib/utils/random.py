from typing import Optional

import numpy as np # type: ignore

def get_random_generator(
        rng: Optional[np.random.Generator] = None) -> np.random.Generator: # type: ignore
    if rng is not None:
        return rng # type: ignore
    return np.random.default_rng() # type: ignore
