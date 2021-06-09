import numpy as np
from typing import Callable, Sequence, Tuple, Union
from itertools import product
Number = Union[int, float]
Genotype = np.ndarray


def make_optimizer(fitness_fn: Callable[[Genotype], Number],
                   next_gen_fn: Callable[[Number, Sequence[Tuple[Number, Genotype]]],
                                         Sequence[Genotype]],
                   max_fitness: Number,
                   starting_population: Sequence[Genotype])\
                       -> Callable[[], Sequence[Tuple[Number, Genotype]]]:
    population = starting_population

    def optimizer_step() -> Sequence[Tuple[Number, Genotype]]:
        nonlocal population
        fitness_to_genotype = [(fitness_fn(genotype), genotype) for genotype in population]
        population = next_gen_fn(max_fitness, fitness_to_genotype)
        return fitness_to_genotype

    return optimizer_step


def _calc_normalized_fitnesses(max_fitness: Number, fitnesses: np.ndarray) -> Sequence[Number]:
    standardized_fitnesses: np.ndarray = max_fitness - fitnesses
    adjusted_fitnesses = 1 / (1 + standardized_fitnesses)
    sum_adjusted_fitnesses = np.sum(adjusted_fitnesses)
    return adjusted_fitnesses / sum_adjusted_fitnesses


def roullete_wheel_selection(max_fitness: Number,
                              fitness_to_genotype: Sequence[Tuple[Number, Genotype]])\
                                 -> Tuple[Genotype, Genotype]:
    normalized_fitnesses = _calc_normalized_fitnesses(max_fitness,
                                                      np.array(tuple(x[0]
                                                                     for x in fitness_to_genotype)))
    genotypes = tuple(x[1] for x in fitness_to_genotype)
    winners = np.random.choice(range(len(genotypes)),
                               p=normalized_fitnesses, size=2)
    return genotypes[winners[0]], genotypes[winners[1]]


def crossover(alpha: Genotype, omega: Genotype) -> Tuple[Genotype, Genotype]:
    size = alpha.shape[0]
    type_ = alpha.dtype
    locus = np.random.randint(size)
    child0 = np.zeros(size, dtype=type_)
    child0[:locus], child0[locus:] = alpha[:locus], omega[locus:]
    child1 = np.zeros(size, dtype=type_)
    child1[:locus], child1[locus:] = omega[:locus], alpha[locus:]
    return child0, child1


def mutate(population: Sequence[Genotype], prob: float) -> None:
    for i, j in product(range(len(population)), range(population[0].shape[0])):
        if np.random.rand() < prob:
            population[i][j] += np.random.choice((-1, 1))