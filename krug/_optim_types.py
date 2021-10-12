from typing import Union, Callable
import numpy as np

Number = Union[int, float]
ObjectiveFunc = Callable[[np.ndarray], Number]
