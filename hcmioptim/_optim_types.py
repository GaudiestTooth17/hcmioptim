from typing import Sequence, TypeVar, Union, Callable
import numpy as np

T = TypeVar('T', Sequence[int], Sequence[float], np.ndarray)
Number = Union[int, float]
ObjectiveFunc = Callable[[T], Number]
