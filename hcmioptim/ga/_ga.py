import numpy as np
from typing import Any, Callable, Generic, List, Sequence, Tuple, TypeVar, Union
from numpy import random
# TODO: check out multiprocessing (Process and Value)
Number = Union[int, float]
T = TypeVar('T', Sequence[int], Sequence[float], np.ndarray)
FitnessFunc = Callable[[T], Number]
NextGenFunc = Callable[[Number, Sequence[Tuple[Number, T]]], Sequence[T]]


class GAOptimizer(Generic[T]):
    def __init__(self, fitness_fn: FitnessFunc,
                 next_gen_fn: NextGenFunc,
                 max_fitness: Number,
                 starting_population: Sequence[T],
                 remember_fitness: bool) -> None:
        """
        A class that lets a genetic algorithm run one step at a time.

        The provided parameters fill in the blanks in the general genetic algorithm form.
        fitness_fn: Take a genotype and return a fitness value not exceeding max_fitness.
        next_gen_fn: Take the maximum fitness a genotype can have and a list of (fitness, genotype).
                     and return a vector of genotypes to be the next generation.
        max_fitness: The best estimate of the highest value fitness_fn can return.
        starting_population: The population of genotypes that the optimizer begins with.
        remember_fitness: If True, the optimizer saves the fitness of each genotype that passes
                          through the fitness function. If a genotype has already been through,
                          the fitness function is not run and the saved value is returned. If False,
                          the fitness function is run each time. 
        """
        self._fitness_fn = fitness_fn
        self._next_gen_fn = next_gen_fn
        self._max_fitness = max_fitness
        self._population = starting_population
        self._remember_fitness = remember_fitness
        self._genotype_to_fitness = {}

    def step(self) -> Sequence[Tuple[Number, T]]:
        fitness_to_genotype = tuple((self._call_fitness(genotype), genotype)
                                    for genotype in self._population)
        self._population = self._next_gen_fn(self._max_fitness, fitness_to_genotype)
        return fitness_to_genotype

    def _call_fitness(self, genotype: T) -> Number:
        if self._remember_fitness:
            hashable_genotype = tuple(genotype)
            if hashable_genotype not in self._genotype_to_fitness:
                self._genotype_to_fitness[hashable_genotype] = self._fitness_fn(genotype)
            return self._genotype_to_fitness[hashable_genotype]
        return self._fitness_fn(genotype)


def _calc_normalized_fitnesses(max_fitness: Number, fitnesses: np.ndarray) -> Sequence[Number]:
    """Return the normalized values of fitnesses with max_fitness."""
    standardized_fitnesses: np.ndarray = max_fitness - fitnesses
    adjusted_fitnesses = 1 / (1 + standardized_fitnesses)
    sum_adjusted_fitnesses = np.sum(adjusted_fitnesses)
    return adjusted_fitnesses / sum_adjusted_fitnesses


def roulette_wheel_fitness_selection(max_fitness: Number,
                                     fitness_to_genotype: Sequence[Tuple[Number, T]])\
                                         -> Tuple[Tuple[T, T], ...]:
    """
    Create a sequence of pairings of genotypes to crossover using the roulette wheel selection method.
    This uses fitness (cost) weighting.
    """
    normalized_fitnesses = _calc_normalized_fitnesses(max_fitness,
                                                      np.array(tuple(x[0]
                                                                     for x in fitness_to_genotype)))
    genotypes = tuple(x[1] for x in fitness_to_genotype)
    inds = np.random.choice(range(len(genotypes)),
                            p=normalized_fitnesses,
                            size=len(fitness_to_genotype))
    pairings = tuple((genotypes[inds[i]], genotypes[inds[i+1]])
                     for i in range(0, len(inds), 2))
    return pairings


def roulette_wheel_rank_selection(fitness_to_genotype: Sequence[Tuple[Number, T]])\
    -> Tuple[Tuple[T, T], ...]:
    """
    Create a sequence of pairings of genotypes to crossover using the roulette wheel selection method.
    This uses rank weighting.
    """
    sorted_ftg = sorted(fitness_to_genotype, key=lambda x: x[0], reverse=True)
    genotypes = tuple(x[1] for x in sorted_ftg)
    N = len(fitness_to_genotype)
    denominator = np.sum(range(1, N+1))
    selection_probabilities = tuple((N-n+1)/denominator for n in range(1, N+1))
    inds = np.random.choice(range(N),
                            p=selection_probabilities,
                            size=N)
    pairings = tuple((genotypes[inds[i]], genotypes[inds[i]])
                     for i in range(0, len(inds), 2))
    return pairings


def uniform_random_pairing_selection(fitness_to_genotype: Sequence[Tuple[Number, T]]) -> Tuple[Tuple[T, T], ...]:
    genotypes = tuple(x[1] for x in fitness_to_genotype)
    inds = np.random.choice(range(len(genotypes)),
                            size=len(fitness_to_genotype))
    pairings = tuple((genotypes[inds[i]], genotypes[inds[i+1]])
                     for i in range(0, len(inds), 2))
    return pairings


def tournament_selection(fitness_to_genotype: Sequence[Tuple[Number, T]],
                         tournament_size=2) -> Tuple[Tuple[T, T], ...]:
    """
    Create a sequence of pairings of genotypes to crossover using the tournament selection method.
    """
    inds = np.arange(len(fitness_to_genotype))
    def run_competition():
        competitors = np.random.choice(inds, size=tournament_size)
        winner = max((fitness_to_genotype[c] for c in competitors), key=lambda x: x[0])[1]
        return winner
    pairings = tuple((run_competition(), run_competition()) for _ in range(len(inds)//2))
    return pairings


def single_point_crossover(alpha: T, beta: T) -> Tuple[T, T]:
    """
    Recombine genotypes alpha and beta.

    Choose a random point along the length of the genotypes. Give genes from alpha before this point
    to one child and genes from beta after that point to the same child. Do the reverse for the
    other child.
    alpha: some genotype
    beta: some genotype
    return: the two children
    """
    size = len(alpha)
    locus = np.random.randint(size)
    if isinstance(alpha, np.ndarray):
        type_ = alpha.dtype
        child0 = np.zeros(size, dtype=type_)
        child1 = np.zeros(size, dtype=type_)
        child0[:locus], child0[locus:] = alpha[:locus], beta[locus:]
        child1[:locus], child1[locus:] = beta[:locus], alpha[locus:]
    else:
        constructor = list if isinstance(alpha, list) else tuple
        child0 = constructor(alpha[i] if i < locus else beta[i] for i in range(size))
        child1 = constructor(beta[i] if i < locus else alpha[i] for i in range(size))
    
    return child0, child1  # type: ignore
