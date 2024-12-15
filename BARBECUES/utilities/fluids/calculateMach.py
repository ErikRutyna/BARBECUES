import numpy as np
from numba import njit

@njit(cache=True)
def calculateMach(stateVectors, y):
    """Calculates the Mach number for each unique state vector.

    :param stateVectors: State vectors of the given mesh
    :param y: Ratio of specific heats of the working fluid, gamma
    :return: Returns mach number
    """
    # Velocity magnitude, q = sqrt(U^2 + V^2)
    q = np.sqrt(np.power(np.divide(stateVectors[:, 1], stateVectors[:, 0]), 2) + np.power(np.divide(stateVectors[:, 2], stateVectors[:, 0]), 2))

    # Static pressure, p = (y - 1) * (rho*E - 0.5 * rho * q^2)
    pressure = (y - 1) * (stateVectors[:, 3] - 0.5 * np.multiply(stateVectors[:, 0], np.power(q, 2)))

    # Speed of sound, c = sqrt(y*p/rho)
    c = np.sqrt(y) * np.sqrt(np.divide(pressure, stateVectors[:, 0]))

    mach = np.divide(q, c)

    return mach