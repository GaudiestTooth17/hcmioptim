import numpy as np
from typing import Callable, Sequence, Tuple
from hcmioptim._optim_types import Number, ObjectiveFunc
from multiprocessing import Pool
NextGenFunc = Callable[[Sequence[Tuple[Number, np.ndarray]]], Sequence[np.ndarray]]
RNG = np.random.default_rng()


class GAOptimizer:
    def __init__(self, objective: ObjectiveFunc,
                 next_gen_fn: NextGenFunc,
                 starting_population: Sequence[np.ndarray],
                 remember_cost: bool,
                 num_processes=1) -> None:
        """
        A class that lets a genetic algorithm run one step at a time.

        The provided parameters fill in the blanks in the general genetic algorithm form.
        objective: Take a encoding and return a cost value not exceeding max_cost.
        next_gen_fn: Take the maximum cost a encoding can have and a list of (cost, encoding).
                     and return a vector of encodings to be the next generation.
        starting_population: The population of encodings that the optimizer begins with.
        remember_cost: If True, the optimizer saves the cost of each encoding that passes
                       through the cost function. If a encoding has already been through,
                       the cost function is not run and the saved value is returned. If False,
                       the cost function is run each time.
        """
        self._objective = objective
        self._next_gen_fn = next_gen_fn
        self._population = starting_population
        self._remember_cost = remember_cost
        self._encoding_to_cost = {}
        self._num_processes = num_processes
        self._n_cache_hits = 0  # records the number of hits _encoding_to_cost has had

    def step(self) -> Sequence[Tuple[Number, np.ndarray]]:
        if self._num_processes <= 1:
            cost_to_encoding = tuple((self._call_objective(encoding), encoding)
                                     for encoding in self._population)
        else:
            with Pool(self._num_processes) as p:
                cost_to_encoding = p.map(_rate_encoding,
                                         ((self._call_objective, encoding)
                                          for encoding in self._population),
                                         len(self._population)//self._num_processes)
        self._population = self._next_gen_fn(cost_to_encoding)
        return cost_to_encoding

    def _call_objective(self, encoding: np.ndarray) -> Number:
        if self._remember_cost:
            hashable_encoding = tuple(encoding)
            if hashable_encoding not in self._encoding_to_cost:
                self._encoding_to_cost[hashable_encoding] = self._objective(encoding)
            else:
                self._n_cache_hits += 1
            return self._encoding_to_cost[hashable_encoding]
        return self._objective(encoding)

    @property
    def num_cache_hits(self) -> int:
        return self._n_cache_hits


def _rate_encoding(args) -> Tuple[Number, np.ndarray]:
    objective, encoding = args
    return (objective(encoding), encoding)


def _calc_cost_selection_probs(costs: np.ndarray) -> np.ndarray:
    """Return the normalized values of costs with max_cost."""
    max_cost_ind = np.argmax(costs)
    max_cost = costs[max_cost_ind]
    normalized_costs = costs - max_cost
    denominator = np.sum(normalized_costs[:max_cost_ind]) + np.sum(normalized_costs[max_cost_ind:])
    if denominator == 0:
        return np.zeros(costs.shape[0], dtype=np.float64)
    P_n = np.abs(normalized_costs / denominator)
    return P_n


def roulette_wheel_cost_selection(cost_to_encoding: Sequence[Tuple[Number, np.ndarray]], rng=RNG)\
        -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    Create a sequence of pairings of encodings to crossover using
    the roulette wheel selection method.
    This uses cost weighting.
    """
    P_n = _calc_cost_selection_probs(np.array(tuple(x[0] for x in cost_to_encoding)))
    if np.sum(P_n) == 0:
        P_n = np.ones(len(cost_to_encoding), dtype=np.float64) / len(cost_to_encoding)
    encodings = tuple(x[1] for x in cost_to_encoding)
    inds = rng.choice(range(len(encodings)),
                      p=P_n,
                      size=len(cost_to_encoding))
    pairings = tuple((encodings[inds[i]], encodings[inds[i+1]])
                     for i in range(0, len(inds), 2))
    return pairings


def roulette_wheel_rank_selection(cost_to_encoding: Sequence[Tuple[Number, np.ndarray]], rng=RNG)\
        -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    Create a sequence of pairings of encodings to crossover using
    the roulette wheel selection method.
    This uses rank weighting.
    """
    sorted_ftg = sorted(cost_to_encoding, key=lambda x: x[0])
    encodings = tuple(x[1] for x in sorted_ftg)
    N = len(cost_to_encoding)
    denominator = sum(range(1, N+1))
    selection_probabilities = tuple((N-n+1)/denominator for n in range(1, N+1))
    inds = rng.choice(range(N),
                      p=selection_probabilities,
                      size=N)
    pairings = tuple((encodings[inds[i]], encodings[inds[i+1]])
                     for i in range(0, len(inds), 2))
    return pairings


def uniform_random_pairing_selection(cost_to_encoding: Sequence[Tuple[Number, np.ndarray]],
                                     rng=RNG) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """Select parent pairs uniformly at random from the population. I think it's trash TBH."""
    encodings = tuple(x[1] for x in cost_to_encoding)
    inds = rng.choice(range(len(encodings)),
                      size=len(cost_to_encoding))
    pairings = tuple((encodings[inds[i]], encodings[inds[i+1]])
                     for i in range(0, len(inds), 2))
    return pairings


def tournament_selection(cost_to_encoding: Sequence[Tuple[Number, np.ndarray]],
                         tournament_size=2, rng=RNG) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    Create a sequence of pairings of encodings to crossover using the tournament selection method.

    tournament_size: How many uniformly randomly selected genotypes get compared in each tournament.
                     len(cost_to_encoding) must be divisible by tournament_size.
    """
    inds = np.arange(len(cost_to_encoding))

    # I decided to try a different approach so that it would hopefully run faster
    # def run_competition():
    #     competitors = rng.choice(inds, size=tournament_size)
    #     winner = min((cost_to_encoding[c] for c in competitors), key=lambda x: x[0])[1]
    #     return winner
    # pairings = tuple((run_competition(), run_competition()) for _ in range(len(inds)//2))

    # There needs to be len(inds) genotype pairings. Each of these pairings is selected from a
    # tournament, this gives the size parameter.
    tournament_indices: Sequence[int] = rng.choice(inds, size=len(inds)*tournament_size)
    grouped_indices = (tournament_indices[i:i+tournament_size]
                       for i in range(0,
                                      len(tournament_indices),
                                      tournament_size))
    tournament_groups = tuple((cost_to_encoding[i] for i in group) for group in grouped_indices)
    pairings = tuple((min(tournament_groups[i], key=lambda x: x[0])[1],
                      min(tournament_groups[i+1], key=lambda x: x[0])[1])
                     for i in range(0, len(tournament_groups), 2))
    return pairings


def single_point_crossover(alpha: np.ndarray, beta: np.ndarray,
                           rng=RNG) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recombine encodings alpha and beta.

    Choose a random point along the length of the encodings. Give genes from alpha before this point
    to one child and genes from beta after that point to the same child. Do the reverse for the
    other child.
    alpha: some encoding
    beta: some encoding
    return: the two children
    """
    size = len(alpha)
    locus = rng.integers(size)
    type_ = alpha.dtype
    child0 = np.zeros(size, dtype=type_)
    child1 = np.zeros(size, dtype=type_)
    child0[:locus], child0[locus:] = alpha[:locus], beta[locus:]
    child1[:locus], child1[locus:] = beta[:locus], alpha[locus:]

    return child0, child1


def cycle_crossover(alpha: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recombine encodings alpha and beta while ensuring neither have duplicate entries.
    Use this for combinatorial problems.

    Exchange the first value in alpha with the first value in beta. If there are duplicates
    in alpha (and thus beta), exchange the original duplicate value with the value in beta
    at the same index. Continue this process until there are no duplicates.
    """
    child0: np.ndarray = np.copy(alpha)
    child1: np.ndarray = np.copy(beta)
    child0[0], child1[0] = child1[0], child0[0]
    swapped_indices = {0}

    while has_duplicate_values(child0):
        # find original duplicate
        seen_values = set(child0[i] for i in swapped_indices)
        for i in (i for i in range(len(child0)) if i not in swapped_indices):
            # There is no need to update seen_values because the only time a duplicate
            # could enter is when swapping occurs.
            if child0[i] in seen_values:
                child0_val = child0[i]
                child0[i] = child1[i]
                child1[i] = child0_val
                swapped_indices.add(i)
                break

    return child0, child1


def bitset_crossover(alpha: np.ndarray, beta: np.ndarray, rng=RNG)\
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Recombine encodings alpha and beta while keeping the sum constant. This can
    be used for bitset encodings. It is assumed that sum(alpha) == sum(beta).

    Choose swap the values between two randomly chosen locations. While
    sum(child) != sum(parent), swap the values at each index in the children
    starting at the index directly after the last one originally swapped, continuing
    to the end, and wrapping around to the beginning.
    """
    genotype_len = len(alpha)
    child0: np.ndarray = np.copy(alpha)
    child1: np.ndarray = np.copy(beta)
    locii = rng.integers(genotype_len, size=2)
    locus0 = np.min(locii)
    locus1 = np.max(locii)
    child0[locus0:locus1], child1[locus0:locus1]\
        = beta[locus0:locus1].copy(), alpha[locus0:locus1].copy()

    difference = np.sum(child0) - np.sum(child1)
    if difference != 0:
        child_with_excess = child0 if difference > 0 else child1
        child_with_deficit = child1 if difference > 0 else child0
        swapple_indices = np.where((child_with_excess > 0) & (child_with_deficit == 0))[0]
        rng.shuffle(swapple_indices)
        to_swap = swapple_indices[:abs(difference)//2]
        for i in to_swap:
            child_with_excess[i] = 1 - child_with_excess[i]
            child_with_deficit[i] = 1 - child_with_deficit[i]

    return child0, child1


def has_duplicate_values(array: np.ndarray):
    s = np.sort(array)
    return (s == np.append(s[1:], s[0])).any()
