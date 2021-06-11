from typing import Sequence, Tuple
from itertools import product
from unittest import TestCase
import numpy as np
from hcmioptim.ga import Number, make_optimizer, Genotype, roullete_wheel_selection, single_point_crossover


def test_mutate(population: Sequence[Genotype], prob: float) -> None:
    for i, j in product(range(len(population)), range(population[0].shape[0])):
        if np.random.rand() < prob:
            population[i][j] += np.random.choice((-1, 1))


def test_fitness(genotype: Genotype) -> int:
    size = genotype.shape[0]
    fitness = 10*size
    for i, gene in enumerate(genotype):
        fitness -= np.abs(i-gene)
    return fitness


def test_next_gen(max_fitness, population: Sequence[Tuple[Number, Genotype]]) -> Sequence[Genotype]:
    parent_pairs = (roullete_wheel_selection(max_fitness, population)
                    for i in range(len(population)//2))
    child_pairs = (single_point_crossover(couple[0], couple[1]) for couple in parent_pairs)
    children = tuple(child for pair in child_pairs for child in pair)
    test_mutate(children, .25)
    return children


class TestGA(TestCase):
    def test_in_order_sequence(self):
        pop_size = 50
        sequence_length = 10
        expected = np.array(range(sequence_length))
        max_fitness = test_fitness(expected)
        max_steps = 1000
        population = [np.abs(np.random.randint(sequence_length, size=sequence_length))
                      for _ in range(pop_size)]
        optimizer_step = make_optimizer(test_fitness, test_next_gen, max_fitness, population)

        for _ in range(max_steps):
            population_with_fitness = optimizer_step()

        best_answer = max(population_with_fitness, key=lambda x: x[0])[1]  # type: ignore

        # This assertion leaves a little wiggle room for answers that aren't completely right
        self.assertTrue(np.sum(np.abs(best_answer-expected)) < 5, f'{best_answer} != {expected}')
