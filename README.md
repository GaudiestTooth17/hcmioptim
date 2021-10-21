# Krug
This package contains a flexible implementation of simulated annealing and genetic algorithms. It's possible that particle swarm optimization may also be added in the future.

## Genetic Algorithms
### Usage
The heart and soul of the `ga` module is `ga.GAOptimizer`. Once the optimizer is constructor, all the programmer needs to do is call the `step` method repeatedly to (theoretically) produce better and better solutions to the objective function.

Here is what the construtor's interface looks like:
```python
class GAOptimizer:
    def __init__(self, objective: ObjectiveFunc,
                 next_gen_fn: NextGenFunc,
                 starting_population: Sequence[np.ndarray],
                 remember_cost: bool,
                 num_processes=1) -> None:
        # see source code for implementation
```
The constructor asks for
* A function to measure how well the phenotype encoded by a genotype fulfills some requirement. Lower is better. The return value is called the cost
* A function that takes in a population of genotypes and their associated costs and returns the next generation of genotypes
* A boolean signifying whether or not you want to memoize the cost of each genotype by associating the array's equivalent tuple to the genotype's cost in a dictionary. This is very useful if you have a computationally expensive objective function and many duplicate genotypes.
* The number of processes you would like the optimizer to use. Forking a process is expensive, so you don't want this too high, however, many non-trivial objective functions take long enough to run that having multiple processes will significantly speed up the time it takes for one step to complete.

A few of the type annotations are specific to krug. Their definitions are given here.
```python
ObjectiveFunc = Callable[[np.ndarray], Number]
NextGenFunc = Callable[[Sequence[Tuple[Number, np.ndarray]]], Sequence[np.ndarray]]
Number = Union[int, float]
```
The way to read these is:
* ObjectiveFunc is a Callable (a function or an object with \_\_call__ defined) that takes a NumPy array (genotype) as input and returns its genotype's cost. So, ObjectiveFunc must also convert from a NumPy array into whatever is actually being optimized (the phenotype).
* NextGenFunc is a Callable that takes a Sequence (tuple or list) of two-tuples. Each two-tuple has either a float or int as the first element and a NumPy array (1D) as the second element. The Number is the cost given to the genotype by the objective function. This is similar to a dictionary because it is a collection of pairs of values. I chose to not use dictionaries because NumPy arrays cannot be keys.
* Number is either an `int` or a `float`

```python
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
```

## Simulated Annealing
### Background
Simulated annealing is a random algorithm that attempts to *minimize* an objective function. It does this by tweaking the current best solution slightly and determining if it is better or not. If it is better, the tweaked version becomes the the current best solution. If it isn't, it still may become the current best solution. This has to do with how "hot" the algorithm is. The hotter it is, the more likely poor solutions are to get accepted. The hope is that accepting less optimal solutions occassionally will get solution out of local minima.
### Usage
As the end user, you must provide:
1. The function to minimize. This will typically transform the solution into some more usasble form.
2. A neighbor function that returns a slightly perturbed version solution passed to it.
3. A function that returns a temperature when called. It's best practice to have each temperature be cooler than the last. There are built in temperature schedules avaible to use.
4. A starting solution. Simulated annealing could in theory work with any sort of solution space, but to simplify the framework, we only support 1 dimensional NumPy arrays.

All of of these are passed into the constructor of `krug.sa.SAOptimizer`. Typically, you will call the step method of an `SAOptimizer` in a loop and use the last returned solution as your answer. This is up to you though as the design is flexible enough to allow for a range of uses. `SAOptimizer` also suports replacing the current best solution with a new one. This could be useful if you want to run multiple instances at once and have them interact.