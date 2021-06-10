from hcmioptim.ga import Number
from typing import Callable, Tuple
import numpy as np
Solution = np.ndarray


class SAOptimizer:
    def __init__(self, objective: Callable[[Solution], Number],
                 next_temp: Callable[[], float],
                 neighbor: Callable[[Solution], Solution],
                 sigma0: Solution,
                 remember_energy: bool) -> None:
        """
        A class that lets the simulated annealing algorithm run 1 step at a time.

        The provided parameters fill in the blanks in the general simulated annealing algorithm.
        objective: Assign a value to a solution.
        next_temp: Return the next temperature to use. Temperatures generally decrease over time.
        neighbor: Return a solution that differs slightly from the one it is given.
        sigma0: The starting guess.
        remember_energy: If True, the optimizer saves the value of each solution after running the objective function and
                        attempts to look up solutions before running the objective function. Otherwise, it just runs the
                        objective each time.
        """
        self._objective = objective
        self._next_temp = next_temp
        self._neighbor = neighbor
        self._sigma = sigma0
        self._energy = self._objective(self._sigma)
        self._T = self._next_temp()
        self._remember_energy = remember_energy
        self._solution_to_energy = {}

    def step(self) -> Tuple[Solution, float]:
        """Execute 1 step of the simulated annealing algorithm."""
        sigma_prime = self._neighbor(self._sigma)
        self._energy = self._run_objective(self._sigma)
        energy_prime = self._run_objective(sigma_prime)
        if P(self._energy, energy_prime, self._T) >= np.random.rand():
            self._sigma = sigma_prime
            self._energy = energy_prime
        self._T = self._next_temp()

        return self._sigma, self._energy

    def update_solution(self, new: Solution, new_energy: Solution) -> None:
        """Change the stored solution."""
        self._sigma = new
        self._energy = new_energy

    def _run_objective(self, solution: Solution) -> Number:
        """Run the objective function or possibly return a saved value."""
        if self._remember_energy:
            hashable_solution = tuple(solution)
            if hashable_solution not in self._solution_to_energy:
                self._solution_to_energy[hashable_solution] = self._objective(solution)
            return self._solution_to_energy[hashable_solution]
        return self._objective(solution)


def P(energy, energy_prime, T) -> float:
    if energy_prime < energy:
        acceptance_prob = 1.0
    else:
        acceptance_prob = np.exp(-(energy_prime-energy)/T) if T != 0 else 0
    return acceptance_prob


def make_fast_schedule(T0: float) -> Callable[[], float]:
    """Rapidly decrease the temperature."""
    num_steps = -1

    def next_temp() -> float:
        nonlocal num_steps
        num_steps += 1
        return T0 / (num_steps + 1)

    return next_temp


def make_linear_schedule(T0: float, delta_T: float) -> Callable[[], float]:
    """Decrease the temperature linearly."""
    T = T0 + delta_T

    def schedule() -> float:
        nonlocal T
        T -= delta_T
        return max(0, T)

    return schedule
