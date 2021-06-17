from typing import Callable, Tuple
import numpy as np
from hcmioptim._optim_types import Number


class SAOptimizer():
    def __init__(self, objective: Callable[[np.ndarray], Number],
                 next_temp: Callable[[], float],
                 neighbor: Callable[[np.ndarray], np.ndarray],
                 encoding0: np.ndarray,
                 remember_cost: bool) -> None:
        """
        A class that lets the simulated annealing algorithm run 1 step at a time.

        The provided parameters fill in the blanks in the general simulated annealing algorithm.
        objective: Assign a value to a solution.
        next_temp: Return the next temperature to use. Temperatures generally decrease over time.
        neighbor: Return a solution that differs slightly from the one it is given.
        encoding0: The starting guess.
        remember_energy: If True, the optimizer saves the value of each solution after running the
                         objective function and attempts to look up solutions before running the
                         objective function. Otherwise, it just runs the objective each time.
        """
        self._objective = objective
        self._next_temp = next_temp
        self._neighbor = neighbor
        self._encoding = encoding0
        self._cost = self._objective(self._encoding)
        self._T = self._next_temp()
        self._remember_cost = remember_cost
        self._encoding_to_cost = {}

    def step(self) -> Tuple[np.ndarray, float]:
        """Execute 1 step of the simulated annealing algorithm."""
        sigma_prime = self._neighbor(self._encoding)
        self._cost = self._call_objective(self._encoding)
        energy_prime = self._call_objective(sigma_prime)
        if P(self._cost, energy_prime, self._T) >= np.random.rand():
            self._encoding = sigma_prime
            self._cost = energy_prime
        self._T = self._next_temp()

        return self._encoding, self._cost

    def update_solution(self, new: np.ndarray, new_energy: np.ndarray) -> None:
        """Change the stored solution."""
        self._encoding = new
        self._cost = new_energy

    def _call_objective(self, solution: np.ndarray) -> Number:
        """Run the objective function or possibly return a saved value."""
        if self._remember_cost:
            hashable_solution = tuple(solution)
            if hashable_solution not in self._encoding_to_cost:
                self._encoding_to_cost[hashable_solution] = self._objective(solution)
            return self._encoding_to_cost[hashable_solution]
        return self._objective(solution)


def P(energy: float, energy_prime: float, temp: float) -> float:
    if energy_prime < energy:
        acceptance_prob = 1.0
    else:
        acceptance_prob = np.exp(-(energy_prime-energy)/temp) if temp != 0 else 0
    return acceptance_prob


def make_fast_schedule(temp0: float) -> Callable[[], float]:
    """Rapidly decrease the temperature."""
    num_steps = -1

    def next_temp() -> float:
        nonlocal num_steps
        num_steps += 1
        return temp0 / (num_steps + 1)

    return next_temp


def make_linear_schedule(T0: float, delta_temp: float) -> Callable[[], float]:
    """Decrease the temperature linearly."""
    temp = T0 + delta_temp

    def schedule() -> float:
        nonlocal temp
        temp -= delta_temp
        return max(0, temp)

    return schedule
