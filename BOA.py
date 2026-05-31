import time

import numpy as np


# Bobcat Optimization Algorithm
def BOA(pop, obj_func, lb, ub, max_iter):
    pop_size, dim = pop.shape[0], pop.shape[1]
    fitness = np.array([obj_func(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best_sol = pop[best_idx].copy()
    best_fit = fitness[best_idx]

    convergence = []

    # Main loop
    ct = time.time()
    for t in range(max_iter):

        # ================= Exploration Phase =================
        new_pop = pop.copy()

        for i in range(pop_size):

            # Eq.(4) Candidate prey selection
            better_idx = np.where(fitness < fitness[i])[0]
            better_idx = better_idx[better_idx != i]

            if len(better_idx) == 0:
                prey = pop[i]
            else:
                prey = pop[np.random.choice(better_idx)]

            candidate = pop[i].copy()

            # Eq.(5) Exploration update
            for j in range(dim):
                r = np.random.rand()
                I = np.random.choice([1, 2])
                candidate[j] = pop[i, j] + (1 - 2 * r) * (prey[j] - I * pop[i, j])

            candidate = np.clip(candidate, lb, ub)

            # Eq.(6) Greedy selection
            f_new = obj_func(candidate)
            if f_new <= fitness[i]:
                new_pop[i] = candidate
                fitness[i] = f_new

        pop = new_pop.copy()

        # ================= Exploitation Phase =================
        new_pop = pop.copy()

        for i in range(pop_size):
            candidate = pop[i].copy()

            # Eq.(7) Exploitation update
            for j in range(dim):
                r = np.random.rand()
                candidate[j] = pop[i, j] + ((1 - 2 * r) / (1 + t)) * pop[i, j]

            candidate = np.clip(candidate, lb, ub)

            f_new = obj_func(candidate)
            if f_new <= fitness[i]:
                new_pop[i] = candidate
                fitness[i] = f_new

        pop = new_pop.copy()

        # ---- Global best update ----
        idx = np.argmin(fitness)
        if fitness[idx] < best_fit:
            best_fit = fitness[idx]
            best_sol = pop[idx].copy()

        convergence.append(best_fit)
    ct = time.time() - ct
    return best_fit, convergence, best_sol, ct
