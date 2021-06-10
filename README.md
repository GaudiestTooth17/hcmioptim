# HCMI Optim
This package contains a flexible implementation of simulated annealing and genetic algorithms. It's possible that particle swarm optimization may also be added in the future.

## Simulated Annealing
### Background
Simulated annealing is a random algorithm that attempts to *minimize* an objective function. It does this by tweaking the current best solution slightly and determining if it is better or not. If it is better, the tweaked version becomes the the current best solution. If it isn't, it still may become the current best solution. This has to do with how "hot" the algorithm is. The hotter it is, the more likely poor solutions are to get accepted. The hope is that accepting less optimal solutions occassionally will get solution out of local minima.
### Usage
As the end user, you must provide:
1. The function to minimize. This will typically transform the solution into some more usasble form.
2. A neighbor function that returns a slightly perturbed version solution passed to it.
3. A function that returns a temperature when called. It's best practice to have each temperature be cooler than the last. There are built in temperature schedules avaible to use.
4. A starting solution. Simulated annealing could in theory work with any sort of solution space, but to simplify the framework, we only support 1 dimensional numpy arrays.

All of of these are passed into the constructor of `ho.sa.SAOptimizer`. Typically, you will call the step method of an `SAOptimizer` in a loop and use the last returned solution as your answer. This is up to you though as the design is flexible enough to allow for a range of uses. `SAOptimizer` also suports replacing the current best solution with a new one. This could be useful if you want to run multiple instances at once and have them interact.