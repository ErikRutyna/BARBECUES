from numba import njit
import numpy as np
import flux_roe


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



@njit(cache=True)
def updateStateRK2(IE, BE, state, beLength, beNorm, ieLength, ieNorm, areas, cfl, M, a, y):
    """Updates the state using RK2 (Predictor-Corrector)"""
    residuals, sumSL = flux_roe.compResidualsRoeVectorized(IE, BE, state, beLength, beNorm, ieLength, ieNorm, M, a, y)

    deltaT = 2 * cfl * np.divide(areas, sumSL)

    stateUpdate = np.zeros((state.shape[0], 4))

    stateUpdate[:, 0] = np.divide(np.multiply(residuals[:, 0], deltaT), areas)
    stateUpdate[:, 1] = np.divide(np.multiply(residuals[:, 1], deltaT), areas)
    stateUpdate[:, 2] = np.divide(np.multiply(residuals[:, 2], deltaT), areas)
    stateUpdate[:, 3] = np.divide(np.multiply(residuals[:, 3], deltaT), areas)

    updatedState = state - stateUpdate

    residuals1, _ = flux_roe.compResidualsRoeVectorized(IE, BE, updatedState, beLength, beNorm, ieLength, ieNorm, M, a, y)

    stateUpdate2 = np.zeros((state.shape[0], 4))

    stateUpdate2[:, 0] = np.divide(np.multiply(residuals1[:, 0] + residuals[:, 0], deltaT/2), areas)
    stateUpdate2[:, 1] = np.divide(np.multiply(residuals1[:, 1] + residuals[:, 1], deltaT/2), areas)
    stateUpdate2[:, 2] = np.divide(np.multiply(residuals1[:, 2] + residuals[:, 2], deltaT/2), areas)
    stateUpdate2[:, 3] = np.divide(np.multiply(residuals1[:, 3] + residuals[:, 3], deltaT/2), areas)

    state -= stateUpdate2

    return 0.5 * (residuals + residuals1)


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