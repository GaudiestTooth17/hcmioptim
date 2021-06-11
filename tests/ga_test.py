from typing import Sequence, Tuple
from itertools import product
from unittest import TestCase
import numpy as np
from hcmioptim import ga
from hcmioptim._optim_types import T, Number


def mutate(population: Sequence[T], prob: float) -> None:
    for i, j in product(range(len(population)), range(len(population[0]))):
        if np.random.rand() < prob:
            population[i][j] += np.random.choice((-1, 1))  # type: ignore


def cost(genotype: T) -> int:
    cost = 0
    for i, gene in enumerate(genotype):
        cost += np.abs(i-gene)
    return cost


def next_gen(population: Sequence[Tuple[Number, T]]) -> Sequence[T]:
    parent_pairs = ga.roulette_wheel_cost_selection(population)
    child_pairs = (ga.single_point_crossover(couple[0], couple[1]) for couple in parent_pairs)
    children = tuple(child for pair in child_pairs for child in pair)
    mutate(children, .25)
    return children


class TestGA(TestCase):
    def test_in_order_sequence(self):
        pop_size = 50
        sequence_length = 10
        expected = np.array(range(sequence_length))
        max_steps = 3000
        population = [np.abs(np.random.randint(sequence_length, size=sequence_length))
                      for _ in range(pop_size)]
        optimizer = ga.GAOptimizer(cost, next_gen, population, True)

        for _ in range(max_steps):
            population_with_fitness = optimizer.step()

        best_answer = min(population_with_fitness, key=lambda x: x[0])[1]  # type: ignore

        # This assertion leaves a little wiggle room for answers that aren't completely right
        self.assertTrue(np.sum(np.abs(best_answer-expected)) < 5, f'{best_answer} != {expected}')
