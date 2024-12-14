from numba import njit
import numpy as np
import math


@njit(cache=True)
def initializeFreestream(stateVectors, M, a, y):
    """Initializes the solution by setting everything based on the freestream state.

    :param stateVectors: Array of 2-D state vectors
    :param M: Freestream Mach number
    :param a: Freestream angle of attack
    :param y: Ratio of specific heats - gamma
    """

    stateVectors[:, 0] = 1                                 # rho
    stateVectors[:, 1] = M * math.cos(a * math.pi / 180)   # rho*u
    stateVectors[:, 2] = M * math.sin(a * math.pi / 180)   # rho*v
    stateVectors[:, 3] = 1 / (y - 1) / y + M ** 2 / 2      # rho*E

    return