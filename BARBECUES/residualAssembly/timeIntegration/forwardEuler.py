from numba import njit
import numpy as np
from BARBECUES.flux import roe


@njit(cache=True)
def updateStateRK1(IE, BE, state, beLength, beNorm, ieLength, ieNorm, areas, CFL, M, a, y):
    """Updates the state using RK1 (Forward-Euler)"""
    residuals, sumSL = flux_roe.compResidualsRoeVectorized(IE, BE, state, beLength, beNorm, ieLength, ieNorm, M, a, y)

    deltaT = 2 * CFL * np.divide(areas, sumSL)

    stateUpdate = np.zeros((state.shape[0], 4))

    stateUpdate[:, 0] = np.divide(np.multiply(residuals[:, 0], deltaT), areas)
    stateUpdate[:, 1] = np.divide(np.multiply(residuals[:, 1], deltaT), areas)
    stateUpdate[:, 2] = np.divide(np.multiply(residuals[:, 2], deltaT), areas)
    stateUpdate[:, 3] = np.divide(np.multiply(residuals[:, 3], deltaT), areas)

    state -= stateUpdate

    return residuals