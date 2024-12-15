from numba import njit
import numpy as np
from BARBECUES.flux import roe


@njit(cache=True)
def updateStateRK4(IE, BE, state, beLength, beNorm, ieLength, ieNorm, areas, cfl, M, a, y):
    """Updates the state using RK4"""
    residuals, sumSL = flux_roe.compResidualsRoeVectorized(IE, BE, state, beLength, beNorm, ieLength, ieNorm, M, a, y)

    deltaT = 2 * cfl * np.divide(areas, sumSL)

    stateUpdate = np.zeros((state.shape[0], 4))

    stateUpdate[:, 0] = np.divide(np.multiply(residuals[:, 0], deltaT/2), areas)
    stateUpdate[:, 1] = np.divide(np.multiply(residuals[:, 1], deltaT/2), areas)
    stateUpdate[:, 2] = np.divide(np.multiply(residuals[:, 2], deltaT/2), areas)
    stateUpdate[:, 3] = np.divide(np.multiply(residuals[:, 3], deltaT/2), areas)

    updatedState = state - stateUpdate

    residuals1, _ = flux_roe.compResidualsRoeVectorized(IE, BE, updatedState, beLength, beNorm, ieLength, ieNorm, M, a, y)

    stateUpdate = np.zeros((state.shape[0], 4))

    stateUpdate[:, 0] = np.divide(np.multiply(residuals1[:, 0], deltaT / 2), areas)
    stateUpdate[:, 1] = np.divide(np.multiply(residuals1[:, 1], deltaT / 2), areas)
    stateUpdate[:, 2] = np.divide(np.multiply(residuals1[:, 2], deltaT / 2), areas)
    stateUpdate[:, 3] = np.divide(np.multiply(residuals1[:, 3], deltaT / 2), areas)

    state1 = state - stateUpdate

    residuals2, _ = flux_roe.compResidualsRoeVectorized(IE, BE, state1, beLength, beNorm, ieLength, ieNorm, M, a, y)

    stateUpdate = np.zeros((state.shape[0], 4))

    stateUpdate[:, 0] = np.divide(np.multiply(residuals2[:, 0], deltaT), areas)
    stateUpdate[:, 1] = np.divide(np.multiply(residuals2[:, 1], deltaT), areas)
    stateUpdate[:, 2] = np.divide(np.multiply(residuals2[:, 2], deltaT), areas)
    stateUpdate[:, 3] = np.divide(np.multiply(residuals2[:, 3], deltaT), areas)

    state2 = state - stateUpdate

    residuals3, _ = flux_roe.compResidualsRoeVectorized(IE, BE, state2, beLength, beNorm, ieLength, ieNorm, M, a, y)


    residualsFinal = (residuals + 2*residuals1 + 2*residuals2 + residuals3) / 6

    stateUpdate = np.zeros((state.shape[0], 4))

    stateUpdate[:, 0] = np.divide(np.multiply(residualsFinal[:, 0], deltaT), areas)
    stateUpdate[:, 1] = np.divide(np.multiply(residualsFinal[:, 1], deltaT), areas)
    stateUpdate[:, 2] = np.divide(np.multiply(residualsFinal[:, 2], deltaT), areas)
    stateUpdate[:, 3] = np.divide(np.multiply(residualsFinal[:, 3], deltaT), areas)

    state -= stateUpdate

    return residualsFinal