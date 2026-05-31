import numpy as np
import time


def TSA(positions, objective, lowerbound, upperbound, max_iterations):
    pop_size, dimensions = positions.shape[0], positions.shape[1]
    """
    Tunicate Swarm Algorithm (TSA)

    Parameters
    ----------
    search_agents : int
        Number of search agents

    max_iterations : int
        Maximum number of iterations

    lowerbound : float or list
        Lower search bound

    upperbound : float or list
        Upper search bound

    dimensions : int
        Number of dimensions

    objective : function
        Objective function to minimize

    Returns
    -------
    score : float
        Best fitness value

    position : ndarray
        Best solution

    convergence : ndarray
        Convergence curve
    """

    ct = time.time()
    # Best solution
    position = np.zeros(dimensions)
    score = np.inf

    convergence = np.zeros(max_iterations)
    t = 0
    while t < max_iterations:
        # -------- Fitness Evaluation --------
        for i in range(pop_size):
            # Boundary check
            positions[i] = np.clip(positions[i], lowerbound, upperbound)
            # Fitness
            fitness = objective(positions[i])

            # Update best solution
            if fitness < score:
                score = fitness
                position = positions[i].copy()

        # -------- TSA Position Update --------
        xmin = 1
        xmax = 4
        xr = np.random.randint(xmin, xmax + 1)
        for i in range(positions.shape[0]):
            for j in range(dimensions):
                # Equation:
                # A1 = ((r1 + r2) - 2*r3) / xr
                A1 = ((np.random.rand() + np.random.rand()) - (2 * np.random.rand())) / xr

                c2 = np.random.rand()
                c3 = np.random.rand()

                d_pos = abs(position[j] - c2 * positions[i, j])
                if i == 0:
                    # Leader update equation
                    if c3 >= 0:
                        positions[i, j] = (position[j] + A1 * d_pos)
                    else:
                        positions[i, j] = (position[j] - A1 * d_pos)

                else:
                    # Follower update equation
                    if c3 >= 0:
                        pos_ij = (position[j] + A1 * d_pos)
                    else:
                        pos_ij = (position[j] - A1 * d_pos)

                    positions[i, j] = (pos_ij + positions[i - 1, j]) / 2
        t += 1
        convergence[t - 1] = score
        print(f"Iteration {t} | Best Score = {score}")
    ct = time.time() - ct
    return score, convergence, position, ct
