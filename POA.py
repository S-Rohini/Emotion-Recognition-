import time

import numpy as np


# Porcupines Optimization Algorithm
def POA(X, fitness_func, lb, ub, max_iter):
    n, dim = X.shape[0], X.shape[1]
    eta = 0.4
    tau1 = 0.01
    sigma_t = 0.1

    # Initial Fitness
    fitness = X(fitness_func[:])
    best_idx = np.argmin(fitness)
    worst_idx = np.argmax(fitness)

    best = X[best_idx].copy()
    worst = X[worst_idx].copy()
    best_fitness = fitness[best_idx]

    convergence = np.zeros((1, max_iter))
    ct = time.time()
    for t in range(max_iter):
        #  Original CPO Position Update
        r1 = np.random.rand(n, dim)
        r2 = np.random.rand(n, dim)
        X_new = X + r1 * (best - np.abs(X)) - r2 * (worst - np.abs(X))

        #  Eq.(18) Cauchy Inverse Cumulative Operator
        r = np.random.rand(n, dim)
        C = tau1 * np.tan(np.pi * (r - 0.5))
        X_new = X_new + C

        # Eq.(22) Adaptive Gaussian Mutation
        G = sigma_t * np.random.randn(n, dim)
        X_new = X_new + G

        # Boundary Control
        X_new = np.clip(X_new, lb, ub)

        # Greedy Selection
        new_fitness = np.apply_along_axis(fitness_func, 1, X_new)
        mask = new_fitness < fitness

        X[mask] = X_new[mask]
        fitness[mask] = new_fitness[mask]

        # Update Best / Worst
        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)

        best = X[best_idx].copy()
        worst = X[worst_idx].copy()
        best_fitness = fitness[best_idx]

        convergence.append(best_fitness)
    best_fitness = convergence[max_iter - 1][0]
    ct = time.time() - ct
    return best_fitness, convergence, best, ct
