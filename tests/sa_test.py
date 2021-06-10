from unittest import TestCase
import numpy as np
from hcmioptim.sa import SAOptimizer, make_fast_schedule


def sequence_objective(sigma: np.ndarray) -> int:
    size = sigma.shape[0]
    fitness = 10*size
    for i, x in enumerate(sigma):  # type: ignore
        fitness -= abs(i-x)
    return -fitness


def sequence_neighbor(solution: np.ndarray) -> np.ndarray:
    new = np.copy(solution)
    new[np.random.randint(0, new.shape[0])] += np.random.choice((-1, 1))
    return new


class TestSA(TestCase):
    def test_in_order_sequence(self):
        sequence_length = 10
        T0 = 100.0
        max_steps = 1000
        sigma0 = np.ones(sequence_length, dtype='int')
        optimizer = SAOptimizer(sequence_objective, make_fast_schedule(T0),
                                sequence_neighbor, sigma0, False)

        best_solution = None
        energies = np.zeros(max_steps)
        for step in range(max_steps):
            best_solution, energy = optimizer.step()
            energies[step] = energy

        self.assertIsNotNone(best_solution, 'Solution is None.')
        expected = list(range(sequence_length))
        actual = list(best_solution)  # type: ignore
        self.assertEqual(actual, expected, f'{best_solution} != {expected}')
