import numpy as np
from typing import Callable, Sequence, Tuple, List
from hcmioptim._optim_types import Number, ObjectiveFunc
from multiprocessing import Pool
import itertools as it
NextGenFunc = Callable[[Sequence[Tuple[Number, np.ndarray]]], Sequence[np.ndarray]]


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

    def step(self) -> Sequence[Tuple[Number, np.ndarray]]:
        if self._num_processes <= 1:
            cost_to_encoding = tuple((self._call_objective(encoding), encoding)
                                     for encoding in self._population)
        else:
            # divide up work between the processes
            encodings_per_process = len(self._population) // self._num_processes
            leftover = len(self._population) % self._num_processes
            boundaries = [[i, i+encodings_per_process]
                          for i in range(0, len(self._population)-leftover, encodings_per_process)]
            boundaries[-1][1] += leftover
            # run the processes to get a list of lists
            with Pool(self._num_processes) as p:
                cost_to_encoding = p.map(_rate_multiple_encodings,
                                         ((self._call_objective, self._population[i:j])
                                          for i, j in boundaries))
            # flatten the list
            cost_to_encoding = list(it.chain(*cost_to_encoding))
        self._population = self._next_gen_fn(cost_to_encoding)
        return cost_to_encoding

    def _call_objective(self, encoding: np.ndarray) -> Number:
        if self._remember_cost:
            hashable_encoding = tuple(encoding)
            if hashable_encoding not in self._encoding_to_cost:
                self._encoding_to_cost[hashable_encoding] = self._objective(encoding)
            return self._encoding_to_cost[hashable_encoding]
        return self._objective(encoding)


def _rate_encoding(args) -> Tuple[Number, np.ndarray]:
    objective, encoding = args
    return (objective(encoding), encoding)


def _rate_multiple_encodings(args) -> List[Tuple[Number, np.ndarray]]:
    objective, encodings = args
    return [(objective(encoding), encoding) for encoding in encodings]


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


def roulette_wheel_cost_selection(cost_to_encoding: Sequence[Tuple[Number, np.ndarray]])\
        -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    Create a sequence of pairings of encodings to crossover using the roulette wheel selection method.
    This uses cost weighting.
    """
    P_n = _calc_cost_selection_probs(np.array(tuple(x[0] for x in cost_to_encoding)))
    if np.sum(P_n) == 0:
        P_n = np.ones(len(cost_to_encoding), dtype=np.float64) / len(cost_to_encoding)
    encodings = tuple(x[1] for x in cost_to_encoding)
    inds = np.random.choice(range(len(encodings)),
                            p=P_n,
                            size=len(cost_to_encoding))
    pairings = tuple((encodings[inds[i]], encodings[inds[i+1]])
                     for i in range(0, len(inds), 2))
    return pairings


def roulette_wheel_rank_selection(cost_to_encoding: Sequence[Tuple[Number, np.ndarray]])\
        -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    Create a sequence of pairings of encodings to crossover using the roulette wheel selection method.
    This uses rank weighting.
    """
    sorted_ftg = sorted(cost_to_encoding, key=lambda x: x[0])
    encodings = tuple(x[1] for x in sorted_ftg)
    N = len(cost_to_encoding)
    denominator = np.sum(range(1, N+1))
    selection_probabilities = tuple((N-n+1)/denominator for n in range(1, N+1))
    inds = np.random.choice(range(N),
                            p=selection_probabilities,
                            size=N)
    pairings = tuple((encodings[inds[i]], encodings[inds[i]])
                     for i in range(0, len(inds), 2))
    return pairings


def uniform_random_pairing_selection(cost_to_encoding: Sequence[Tuple[Number, np.ndarray]])\
        -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """Select parent pairs uniformly at random from the population. I think it's trash TBH."""
    encodings = tuple(x[1] for x in cost_to_encoding)
    inds = np.random.choice(range(len(encodings)),
                            size=len(cost_to_encoding))
    pairings = tuple((encodings[inds[i]], encodings[inds[i+1]])
                     for i in range(0, len(inds), 2))
    return pairings


def tournament_selection(cost_to_encoding: Sequence[Tuple[Number, np.ndarray]],
                         tournament_size=2) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    Create a sequence of pairings of encodings to crossover using the tournament selection method.
    """
    inds = np.arange(len(cost_to_encoding))

    def run_competition():
        competitors = np.random.choice(inds, size=tournament_size)
        winner = min((cost_to_encoding[c] for c in competitors), key=lambda x: x[0])[1]
        return winner

    pairings = tuple((run_competition(), run_competition()) for _ in range(len(inds)//2))
    return pairings


def single_point_crossover(alpha: np.ndarray, beta: np.ndarray)\
        -> Tuple[np.ndarray, np.ndarray]:
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
