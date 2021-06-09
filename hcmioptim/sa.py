from typing import Callable, Sequence, Union, Tuple
import numpy as np


def make_sa_optimizer(objective: Callable, next_temp: Callable, neighbor: Callable,
                      sigma0: Union[Sequence[int], Sequence[float]]) -> Callable[[], Tuple[Sequence, float]]:
    T = next_temp()
    sigma = sigma0

    def step():
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
    num_steps = -1

    def next_temp():
        nonlocal num_steps
        num_steps += 1
        return T0 / (num_steps + 1)

    return next_temp


def make_linear_schedule(T0: float, delta_T: float) -> Callable[[], float]:
    T = T0 + delta_T

    def schedule() -> float:
        nonlocal T
        T -= delta_T
        return max(0, T)

    return schedule
