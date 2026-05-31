import time

import numpy as np


# Tomtit Flock Metaheuristic Optimization Algorithm (TFMOA)
def TFMOA(population, obj_fun, lb, ub, max_iter):
    NP, dim = population.shape[0], population.shape[1]
    alpha = 0.1
    gamma = 0.9
    eta = 0.6
    lamda = 1.5
    rho = 5
    c1 = 2
    c2 = 2
    c3 = 2
    h = 0.05
    L = 5
    mu = 2
    epsilon = 1e-9
    best_position = float('inf')

    Convergence_curve = np.zeros((1, max_iter))
    fitness = population(obj_fun[:])
    for i in range(NP):
        fitness[i] = obj_fun(population[i])

    personal_best = population.copy()
    personal_best_fit = fitness.copy()
    best_global = population[np.argmin(fitness)].copy()
    best_global_fit = np.min(fitness)
    Pool_X = []
    Pool_F = []
    r = 1.0
    p = 0
    ct = time.time()
    t = 0
    while p < max_iter:
        k = 0
        Memory_X = []
        Memory_F = []
        while k < max_iter and r > epsilon:
            idx = np.argsort(fitness)
            population = population[idx]
            fitness = fitness[idx]
            personal_best = personal_best[idx]
            personal_best_fit = personal_best_fit[idx]

            best_global = population[0].copy()
            best_global_fit = fitness[0]
            new_population = population.copy()

            for j in range(NP):
                x = population[j].copy()
                best_search = x.copy()
                best_search_fit = obj_fun(best_search)
                dist = np.linalg.norm(population - x, axis=1)
                neigh = np.where(dist <= rho)[0]
                neigh_fit = fitness[neigh]
                local_best = population[neigh[np.argmin(neigh_fit)]].copy()
                l = 0
                while l < L:
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    r3 = np.random.rand()
                    drift = c1 * r1 * (best_global - x)
                    diffusion = (c2 * r2 * (personal_best[j] - x) + c3 * r3 * (local_best - x))
                    a1 = np.random.rand()
                    a2 = np.random.rand()
                    xi = np.sqrt(-2 * np.log(a1)) * np.cos(2 * np.pi * a2)
                    xe = x + h * drift + np.sqrt(h) * diffusion * xi
                    beta = np.random.rand()
                    if beta <= mu * h:
                        delta = np.minimum(ub - xe, xe - lb)
                        theta = np.random.uniform(-delta, delta)
                        x_new = xe + theta

                    else:
                        x_new = xe.copy()
                    x_new = np.maximum(x_new, lb)
                    x_new = np.minimum(x_new, ub)
                    x = x_new.copy()
                    fit = obj_fun(x)
                    if fit < best_search_fit:
                        best_search_fit = fit
                        best_search = x.copy()
                    l += 1
                new_population[j] = best_search.copy()
            levy = np.zeros(dim)
            eps = 1e-7
            for d in range(dim):
                R = np.random.uniform(eps, 1.0)
                theta = R * 2 * np.pi
                Ls = (R + eps) ** (-1 / lamda)
                if d < dim // 2:
                    levy[d] = Ls * np.sin(theta)
                else:
                    levy[d] = Ls * np.cos(theta)
            leader = best_global + (alpha / (k + 1)) * levy
            leader = np.maximum(leader, lb)
            leader = np.minimum(leader, ub)
            new_population[0] = leader.copy()
            population = new_population.copy()
            for i in range(NP):
                fitness[i] = obj_fun(population[i])
            for i in range(NP):

                if fitness[i] < personal_best_fit[i]:
                    personal_best_fit[i] = fitness[i]

                    personal_best[i] = population[i].copy()
            best_idx = np.argmin(fitness)
            Memory_X.append(population[best_idx].copy())
            Memory_F.append(fitness[best_idx])
            r = gamma * r
            k += 1
        best_mem_idx = np.argmin(Memory_F)
        Pool_X.append(Memory_X[best_mem_idx].copy())
        Pool_F.append(Memory_F[best_mem_idx])
        best_pass = Memory_X[best_mem_idx].copy()
        p += 1
        r = eta ** p
        population[0] = best_pass.copy()
        for j in range(1, NP):
            span = r * (ub - lb)
            low = np.maximum(lb, best_pass - span / 2)
            high = np.minimum(ub, best_pass + span / 2)
            population[j] = np.random.uniform(low, high)
        for i in range(NP):
            fitness[i] = obj_fun(population[i])

        final_idx = np.argmin(Pool_F)
        best_position = Pool_X[final_idx]
        best_fitness = Pool_F[final_idx]

        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[max_iter - 1][0]
    ct = time.time() - ct
    return best_fitness, Convergence_curve, best_position, ct
