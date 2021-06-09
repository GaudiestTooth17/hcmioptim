import numpy as np
from typing import Callable, List, Sequence, Tuple, Union
Number = Union[int, float]
Genotype = np.ndarray


def make_optimizer(fitness_fn: Callable[[Genotype], Number],
                   next_gen_fn: Callable[[Number, Sequence[Tuple[Number, Genotype]]],
                                         Sequence[Genotype]],
                   max_fitness: Number,
                   starting_population: Sequence[Genotype])\
                       -> Callable[[], List[Tuple[Number, Genotype]]]:
    """
    Create a closure that executes 1 genetic algorithm step and saves the data needed to take another step.

    The provided parameters fill in the blanks in the general genetic algorithm form.
    fitness_fn: Take a genotype and return a fitness value not exceeding max_fitness.
    next_gen_fn: Take the maximum fitness a genotype can have and a list of (fitness, genotype)
                 and return a vector of genotypes to be the next generation.
    max_fitness: The best estimate of the highest value fitness_fn can return.
    starting_population: The population of genotypes that the optimizer begins with.
    return: A closure that accepts no arguments and returns a list of genotypes associated with their fitnesses.
    """
    population = starting_population

    def optimizer_step() -> List[Tuple[Number, Genotype]]:
        nonlocal population
        fitness_to_genotype = [(fitness_fn(genotype), genotype) for genotype in population]
        population = next_gen_fn(max_fitness, fitness_to_genotype)
        return fitness_to_genotype

    return optimizer_step


def _calc_normalized_fitnesses(max_fitness: Number, fitnesses: np.ndarray) -> Sequence[Number]:
    """Return the normalized values of fitnesses with max_fitness."""
    standardized_fitnesses: np.ndarray = max_fitness - fitnesses
    adjusted_fitnesses = 1 / (1 + standardized_fitnesses)
    sum_adjusted_fitnesses = np.sum(adjusted_fitnesses)
    return adjusted_fitnesses / sum_adjusted_fitnesses


def roullete_wheel_selection(max_fitness: Number,
                              fitness_to_genotype: Sequence[Tuple[Number, Genotype]])\
                                 -> Tuple[Genotype, Genotype]:
    """Choose two genotypes from fitness_to_genotype using the roullete wheel selection method."""
    normalized_fitnesses = _calc_normalized_fitnesses(max_fitness,
                                                      np.array(tuple(x[0]
                                                                     for x in fitness_to_genotype)))
    genotypes = tuple(x[1] for x in fitness_to_genotype)
    winners = np.random.choice(range(len(genotypes)),
                               p=normalized_fitnesses, size=2)
    return genotypes[winners[0]], genotypes[winners[1]]


def crossover(alpha: Genotype, beta: Genotype) -> Tuple[Genotype, Genotype]:
    """
    Recombine genotypes alpha and beta.

    Choose a random point along the length of the genotypes. Give genes from alpha before this point
    to one child and genes from beta after that point to the same child. Do the reverse for the
    other child.
    alpha: some genotype
    beta: some genotype
    return: the two children
    """
    size = alpha.shape[0]
    type_ = alpha.dtype
    locus = np.random.randint(size)
    child0 = np.zeros(size, dtype=type_)
    child0[:locus], child0[locus:] = alpha[:locus], beta[locus:]
    child1 = np.zeros(size, dtype=type_)
    child1[:locus], child1[locus:] = beta[:locus], alpha[locus:]
    return child0, child1
