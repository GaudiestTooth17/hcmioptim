from hcmioptim.ga import Number
from typing import Callable, Sequence, Union, Tuple
import numpy as np
Solution = np.ndarray


def make_sa_optimizer(objective: Callable[[Solution], Number],
                      next_temp: Callable[[], float],
                      neighbor: Callable[[Solution], Solution],
                      sigma0: Solution) -> Callable[[], Tuple[Solution, float]]:
    """
    Create a closure that can be called iteratively to the simulated annealing algorithm 1 step at a time.

    The provided parameters fill in the blanks in the general simulated annealing algorithm.
    objective: Assign a value to a solution.
    next_temp: Return the next temperature to use. Temperatures generally decrease over time.
    neighbor: Return a solution that differs slightly from the one it is given.
    sigma0: The starting guess.
    return: A closure that accepts no arguments and returns the current solution along with it's value.
    """
    T = next_temp()
    sigma = sigma0

    def step() -> Tuple[Solution, float]:
        nonlocal sigma, T
        sigma_prime = neighbor(sigma)
        energy = objective(sigma)
        energy_prime = objective(sigma_prime)
        curr_energy = energy
        if P(energy, energy_prime, T) >= np.random.rand():
            sigma = sigma_prime
            curr_energy = energy_prime
        T = next_temp()

        return sigma, curr_energy

    return step


def P(energy, energy_prime, T) -> float:
    acceptance_prob = 1.0 if energy_prime < energy else np.exp(-(energy_prime-energy)/T)  # type: ignore
    return acceptance_prob


def make_fast_schedule(T0: float) -> Callable:
    """Rapidly decrease the temperature."""
    num_steps = -1

    def next_temp():
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
