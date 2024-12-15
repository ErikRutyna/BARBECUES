import math
import numpy as np
import helper
import initialization as intlzn
from numba import njit


@njit(cache=True)
def stateFluxEuler2D(u, p):
    """Computes the flux vector for the 2D compressible Euler equations from
    the state vector.

    :param u: State vector
    :param p: Local state's pressure
    :return: The flux vector for the 2D compressible Euler equations, each column is a direction
    """
    F = np.zeros((4, 2))

    # Continuity
    F[0, 0] = u[1]
    F[0, 1] = u[2]

    off_momentum = u[1] * u[2] / u[0]
    # x-momentum
    F[1, 0] = (u[1]*u[1]) / u[0] + p
    F[1, 1] = off_momentum

    # y-momentum
    F[2, 0] = off_momentum
    F[2, 1] = (u[2]*u[2]) / u[0] + p

    # Enthalpy
    H = (u[3] + p) / u[0]

    # Energy
    F[3, 0] = u[1] * H
    F[3, 1] = u[2] * H

    return F


@njit(cache=True)
def roeEuler2D(stateLeft, stateRight, pressureLeft, pressureRight, n, y):
    """Computes the Roe Flux from the left cell into the right cell for
    the inviscid Euler Equations

    :param stateLeft: State vector in the left cell [rho, rho*u, rho*v, rho*E]
    :param stateRight: State vector in the right cell [rho, rho*u, rho*v, rho*E]
    :param pressureLeft: Static pressure in the left cell
    :param pressureRight: Static pressure in the right cell
    :param n: Unit normal pointing from left cell to right cell
    :param y: Ratio of specific heats - gamma
    :return: flux: Flux of state vector from left to right cell, s: Maximum propagation speed of the state variables
    """

    # Delta state
    deltaState = np.subtract(stateRight, stateLeft)

    sqrtRhoLeft = math.sqrt(stateLeft[0])
    sqrtRhoRight = math.sqrt(stateRight[0])

    # Roe-average states
    roeAvgU = (sqrtRhoLeft * stateLeft[1] / stateLeft[0] + sqrtRhoRight * stateRight[1] / stateRight[0]) \
                / (sqrtRhoLeft + sqrtRhoRight)
    roeAvgV = (sqrtRhoLeft * stateLeft[2] / stateLeft[0] + sqrtRhoRight * stateRight[2] / stateRight[0]) \
                / (sqrtRhoLeft + sqrtRhoRight)
    qRoeSquared = roeAvgU*roeAvgU + roeAvgV*roeAvgV

    enthalpyLeft = (stateLeft[3] + pressureLeft) / stateLeft[0]

    enthalpyRight = (stateRight[3] + pressureRight) / stateRight[0]

    roeAvgEnthalpy = (sqrtRhoLeft * enthalpyLeft + sqrtRhoRight * enthalpyRight) / (sqrtRhoLeft + sqrtRhoRight)

    # Speed of sound
    c = math.sqrt((y - 1) * (roeAvgEnthalpy - qRoeSquared / 2))

    # Speed
    u = roeAvgU * n[0] + roeAvgV * n[1]

    # Eigenvalues of system
    eigens = np.abs(np.array([u + c, u - c, u]))

    # Entropy fix
    for i in range(len(eigens)):
        if eigens[i] < (0.05 * c):
            eigens[i] = ((0.0025 * c * c) + (eigens[i] * eigens[i])) / (0.1 * c)

    # Maximum propagation speed
    s_max = c + abs(u)

    # Intermediate constants
    s = np.array([0.5 * (eigens[0] + eigens[1]),
                  0.5 * (eigens[0] - eigens[1])])

    g1 = (y - 1) * (qRoeSquared / 2 * deltaState[0] - (roeAvgU * deltaState[1] + roeAvgV * deltaState[2]) + deltaState[3])
    g2 = -u * deltaState[0] + (deltaState[1] * n[0] + deltaState[2] * n[1])

    c1 = g1 / (c*c) * (s[0] - eigens[2]) + g2 / c * s[1]
    c2 = g1 / c * s[1] + (s[0] - eigens[2]) * g2

    # Flux vectorization & normals
    F_l = stateFluxEuler2D(stateLeft, pressureLeft)
    F_l = F_l[:, 0] * n[0] + F_l[:, 1] * n[1]

    F_r = stateFluxEuler2D(stateRight, pressureRight)
    F_r = F_r[:, 0] * n[0] + F_r[:, 1] * n[1]

    # Actual Roe Flux
    flux = 0.5 * (F_l + F_r) - 0.5 * np.array([eigens[2] * deltaState[0] + c1,
                                               eigens[2] * deltaState[1] + c1 * roeAvgU + c2 * n[0],
                                               eigens[2] * deltaState[2] + c1 * roeAvgV + c2 * n[1],
                                               eigens[2] * deltaState[3] + c1 * roeAvgEnthalpy + c2 * u])

    return flux, s_max


@njit(cache=True)
def compResidualsRoe(IE, BE, state, be_l, be_n, ie_l, ie_n, M, a, y):
    """Computes the residuals and sum of speed*edge_lengths for the state on the given mesh using the Roe Flux method.

    :param IE: Internal edges [nodeA, nodeB, left cell, right cell]
    :param BE: Boundary edges [nodeA, nodeB, cell, boundary edge flag]
    :param state: [:, 4] Numpy array of state vectors, [rho, rho*u, rho*v, rho*E]
    :param be_l: [:, 1] Boundary edge lengths
    :param be_n: [:, 2] Boundary edge normal vectors
    :param ie_l: [:, 1] Internal edge lengths
    :param ie_n: [:, 2] Internal edge normal vectors
    :param M: Freestream Mach number
    :param a: Freestream AoA
    :param y: Ratio of specific heats - gamma
    :returns: residuals: Flow field residuals at each cell, sum_sl: Propagation speeds at each cell
    """
    residuals = np.zeros((state.shape[0], 4), dtype=np.float64)  # Residuals from fluxes
    sum_sl = np.transpose(np.zeros((state.shape[0]), dtype=np.float64))

    freestreamState = intlzn.init_state_mach(M, a, y)

    pressureFreestream = helper.calculate_static_pressure_single(freestreamState, y)

    pressures = helper.calculate_static_pressure(state, y)

    # Boundary edges with their respective conditions
    for i in range(BE.shape[0]):
        if BE[i, 3] == 0:
            # Apply inviscid wall boundary condition
            # Boundary velocity
            u_plus = state[BE[i, 2], 1] / state[BE[i, 2], 0]
            v_plus = state[BE[i, 2], 2] / state[BE[i, 2], 0]
            V_plus = np.array([u_plus, v_plus])
            be_vel = V_plus - np.multiply(np.dot(V_plus, be_n[i]), be_n[i])

            # Boundary pressure
            be_pressure = (y - 1) * (state[BE[i, 2], 3] -
                                     0.5 * state[BE[i, 2], 0] * ((be_vel[0]*be_vel[0]) + (be_vel[1]*be_vel[1])))

            # Enforcing no flow through with pressure condition
            be_flux = np.array((0, be_pressure * be_n[i, 0], be_pressure * be_n[i, 1], 0))

            # Calculate propagation speed
            local_p = pressures[BE[i, 2]]
            h = (state[BE[i, 2], 3] + local_p) / state[BE[i, 2], 0]
            be_smax = abs(u_plus * be_n[i, 0] + v_plus * be_n[i, 1]) \
                      + math.sqrt((y - 1) * (h - ((u_plus*u_plus) + (v_plus*v_plus)) / 2))

        elif BE[i, 3] == 1 or BE[i, 3] == 2:
            # Apply supersonic outflow boundary conditions
            be_flux, be_smax = roeEuler2D(state[BE[i, 2]], state[BE[i, 2]],
                                          pressures[BE[i, 2]], pressures[BE[i, 2]], be_n[i], y)

        elif BE[i, 3] == 3:
            # Apply freestream inflow boundary conditions
            be_flux, be_smax = roeEuler2D(state[BE[i, 2]], freestreamState,
                                          pressures[BE[i, 2]], pressureFreestream, be_n[i], y)

        residuals[BE[i, 2]] += be_flux * be_l[i]
        sum_sl[BE[i, 2]] += be_smax * be_l[i]

    # Internal edges
    for i in range(IE.shape[0]):
        ie_flux, ie_smax = roeEuler2D(state[IE[i, 2]], state[IE[i, 3]],
                                      pressures[IE[i, 2]], pressures[IE[i, 3]], ie_n[i], y)

        residuals[IE[i, 2]] += ie_flux * ie_l[i]  # Summing residuals to be taken out of cell i
        residuals[IE[i, 3]] -= ie_flux * ie_l[
            i]  # Summing residuals to negative taken out of cell N (added to cell N)

        sum_sl[IE[i, 2]] += ie_smax * ie_l[i]
        sum_sl[IE[i, 3]] += ie_smax * ie_l[i]

    return residuals, sum_sl


@njit(cache=True)
def compResidualsRoeVectorized(IE, BE, state, be_l, be_n, ie_l, ie_n, M, a, y):
    """Computes the residuals and sum of speed*edge_lengths for the state on the given mesh using the Roe Flux method.

    :param IE: Internal edges [nodeA, nodeB, left cell, right cell]
    :param BE: Boundary edges [nodeA, nodeB, cell, boundary edge flag]
    :param state: [:, 4] Numpy array of state vectors, [rho, rho*u, rho*v, rho*E]
    :param be_l: [:, 1] Boundary edge lengths
    :param be_n: [:, 2] Boundary edge normal vectors
    :param ie_l: [:, 1] Internal edge lengths
    :param ie_n: [:, 2] Internal edge normal vectors
    :param M: Freestream Mach number
    :param a: Freestream AoA
    :param y: Ratio of specific heats - gamma
    :returns: residuals: Flow field residuals at each cell, sum_sl: Propagation speeds at each cell
    """
    residuals = np.zeros((state.shape[0], 4), dtype=np.float64)
    sum_sl = np.transpose(np.zeros((state.shape[0]), dtype=np.float64))

    freestreamState = intlzn.init_state_mach(M, a, y)

    pressureFreestream = helper.calculate_static_pressure_single(freestreamState, y)

    fluxFreestream = stateFluxEuler2D(freestreamState, pressureFreestream)

    sqrtRho, u, v, pressures, H, fluxEuler, mach = computeStateVals(state, y)

    # Wall boundary edge flux computations
    residualsWallBE, sumSLWallBE = wallBEFlux(BE, u, v, state[:, 0], state[:, 3], mach, be_n, be_l, y)
    residuals += residualsWallBE
    sum_sl += sumSLWallBE

    # Exit boundary edge flux computations
    residualsOutflowBEs, sumSLExitBE = outflowBEFluxFast(BE, u, v, mach, fluxEuler, be_n, be_l)
    residuals += residualsOutflowBEs
    sum_sl += sumSLExitBE

    # Inflow boundary edge flux computations
    residualsInflowBEs, sumSLInflowBE = inflowBEFlux(BE, state, u, v, sqrtRho, H, fluxEuler, be_n, be_l, freestreamState, pressureFreestream, fluxFreestream, y)
    residuals += residualsInflowBEs
    sum_sl += sumSLInflowBE

    # Internal boundary edge flux computations
    residualIEs, sumSLIEs = roeEuler2DInternalEdges(IE, state, sqrtRho, u, v, H, fluxEuler, ie_n, ie_l, y, pressures)
    residuals += residualIEs
    sum_sl += sumSLIEs

    return residuals, sum_sl

@njit(cache=True)
def computeStateVals(state, y):
    """Computes a set of vectored quantities from the given state vector array and freestream conditions.

    Parameters
    ----------
    :param state: [:, 4] Numpy array of state variables, [rho, rho*u, rho*v, rho*E]
    :param y: Ratio of specific heats, gamma

    Returns
    -------
    :returns: sqrtRho: [:, 1] Numpy array of the square root of cell's local density
    :returns: u: [:, 1] Numpy array of x-velocities
    :returns: v: [:, 1] Numpy array of y-velocities
    :returns: P: [:, 1] Numpy array of local static pressure
    :returns: H: [:, 1] Numpy array of local enthalpy
    :returns: fluxEuler: [:, 4, 2] Numpy array of Euler Equation fluxes where each row is a [4, 2] array consisting of
        [continuity, momentum, momentum, energy] fluxes
    :returns: a: [:, 1] Numpy array of local speed of sounds
    """
    # Sqrt of density used in Roe-Averaging
    sqrtRho = np.sqrt(state[:, 0])

    # State vector x-velocities
    u = np.divide(state[:, 1], state[:, 0])

    # State vector y-velocities
    v = np.divide(state[:, 2], state[:, 0])

    # State vector local static pressures
    P = helper.calculate_static_pressure(state, y)

    # State vector enthalpies
    H = np.divide(state[:, 3] + P, state[:, 0])

    # Euler equation flux vector
    fluxEuler = np.zeros((state.shape[0], 4, 2))

    # Continuity
    fluxEuler[:, 0, 0] = state[:, 1]
    fluxEuler[:, 0, 1] = state[:, 2]

    offMomentum = np.multiply(state[:, 1], v)
    # x-momentum
    fluxEuler[:, 1, 0] = np.multiply(state[:, 1], u) + P
    fluxEuler[:, 1, 1] = offMomentum

    # y-momentum
    fluxEuler[:, 2, 0] = offMomentum
    fluxEuler[:, 2, 1] = np.multiply(state[:, 2], v) + P

    # Energy
    fluxEuler[:, 3, 0] = np.multiply(state[:, 1], H)
    fluxEuler[:, 3, 1] = np.multiply(state[:, 2], H)

    # Speed of sounds
    a = np.sqrt(y * np.divide(P, state[:, 0]))

    return sqrtRho, u, v, P, H, fluxEuler, a


@njit(cache=True)
def roeEuler2DInternalEdges(IE, state, sqrtRho, u, v, H, fluxEuler, norms, lengths, y, p):
    """

    Parameters
    ----------
    state
    sqrtRho
    u
    v
    p
    fluxEuler
    indexLeft
    indexRight
    norms
    lengths
    y

    Returns
    -------

    """
    internalResiduals = np.zeros((u.shape[0], 4), dtype=np.float64)
    sumSL = np.zeros((u.shape[0]), dtype=np.float64)

    # Filter out indices for left and right states
    leftFilter = IE[:, 2]
    rightFilter = IE[:, 3]

    deltaState = state[rightFilter] - state[leftFilter]

    # "Left" is the interior state, while "right" is the freestream state
    sqrtRhoLeft = sqrtRho[leftFilter]
    sqrtRhoRight = sqrtRho[rightFilter]

    uLeft = u[leftFilter]
    uRight = u[rightFilter]

    vLeft = v[leftFilter]
    vRight = v[rightFilter]

    enthalpyLeft = H[leftFilter]
    enthalpyRight = H[rightFilter]

    # Roe-Average Quantities
    roeAvgU = np.divide(np.multiply(sqrtRhoLeft, uLeft) + np.multiply(sqrtRhoRight, uRight),
                        (sqrtRhoLeft + sqrtRhoRight))
    roeAvgV = np.divide(np.multiply(sqrtRhoLeft, vLeft) + np.multiply(sqrtRhoRight, vRight),
                        (sqrtRhoLeft + sqrtRhoRight))
    roeAvgEnthalpy = np.divide(np.multiply(sqrtRhoLeft, enthalpyLeft) + np.multiply(sqrtRhoRight, enthalpyRight),
                               (sqrtRhoLeft + sqrtRhoRight))
    qRoeSquared = np.multiply(roeAvgU, roeAvgU) + np.multiply(roeAvgV, roeAvgV)

    # Speed of Sound
    c = np.sqrt((y - 1) * (roeAvgEnthalpy - qRoeSquared / 2))

    # Local Speeds
    u = np.multiply(roeAvgU, norms[:, 0]) + np.multiply(roeAvgV, norms[:, 1])

    # System's eigenvalues
    eigenvalues = np.transpose(np.abs(np.vstack((u + c, u - c, u))))

    # Entropy fix
    for i in range(eigenvalues.shape[0]):
        for j in range(3):
            if eigenvalues[i, j] < (0.05 * c[i]):
                eigenvalues[i, j] = ((0.0025 * c[i] * c[i]) + (eigenvalues[i, j] * eigenvalues[i, j])) / (0.1 * c[i])

    # Maximum propagation speed
    s_max = c + np.abs(u)

    # Intermediate constants
    s = np.transpose(np.vstack((0.5 * (eigenvalues[:, 0] + eigenvalues[:, 1]),
                                0.5 * (eigenvalues[:, 0] - eigenvalues[:, 1]))))

    g1 = (y - 1) * (np.multiply(qRoeSquared / 2, deltaState[:, 0]) -
                    (np.multiply(roeAvgU, deltaState[:, 1]) + np.multiply(roeAvgV, deltaState[:, 2])) + deltaState[:,
                                                                                                        3])
    g2 = np.multiply(-u, deltaState[:, 0]) + (
                np.multiply(deltaState[:, 1], norms[:, 0]) + np.multiply(deltaState[:, 2], norms[:, 1]))

    c1 = np.multiply(np.divide(g1, np.multiply(c, c)), (s[:, 0] - eigenvalues[:, 2])) + np.multiply(np.divide(g2, c),
                                                                                                    s[:, 1])
    c2 = np.multiply(np.divide(g1, c), s[:, 1]) + np.multiply(s[:, 0] - eigenvalues[:, 2], g2)

    # Euler Equation Fluxes
    fluxLeft = fluxEuler[leftFilter, :, :]

    fluxRight = fluxEuler[rightFilter, :, :]

    fluxLeftx = np.transpose(np.vstack((fluxLeft[:, 0, 0] * norms[:, 0],
                                        fluxLeft[:, 1, 0] * norms[:, 0],
                                        fluxLeft[:, 2, 0] * norms[:, 0],
                                        fluxLeft[:, 3, 0] * norms[:, 0])))

    fluxLefty = np.transpose(np.vstack((fluxLeft[:, 0, 1] * norms[:, 1],
                                        fluxLeft[:, 1, 1] * norms[:, 1],
                                        fluxLeft[:, 2, 1] * norms[:, 1],
                                        fluxLeft[:, 3, 1] * norms[:, 1])))

    fluxLeft = fluxLeftx + fluxLefty

    fluxRightx = np.transpose(np.vstack((fluxRight[:, 0, 0] * norms[:, 0],
                                         fluxRight[:, 1, 0] * norms[:, 0],
                                         fluxRight[:, 2, 0] * norms[:, 0],
                                         fluxRight[:, 3, 0] * norms[:, 0])))

    fluxRighty = np.transpose(np.vstack((fluxRight[:, 0, 1] * norms[:, 1],
                                         fluxRight[:, 1, 1] * norms[:, 1],
                                         fluxRight[:, 2, 1] * norms[:, 1],
                                         fluxRight[:, 3, 1] * norms[:, 1])))

    fluxRight = fluxRightx + fluxRighty

    # Roe Flux
    flux = 0.5 * (fluxLeft + fluxRight) - \
           0.5 * np.transpose(np.vstack((np.multiply(eigenvalues[:, 2], deltaState[:, 0]) + c1,
                                         np.multiply(eigenvalues[:, 2], deltaState[:, 1]) + np.multiply(c1, roeAvgU) + np.multiply(c2, norms[:, 0]),
                                         np.multiply(eigenvalues[:, 2], deltaState[:, 2]) + np.multiply(c1, roeAvgV) + np.multiply(c2, norms[:, 1]),
                                         np.multiply(eigenvalues[:, 2], deltaState[:, 3]) + np.multiply(c1, roeAvgEnthalpy) + np.multiply(c2, u))))

    fluxResidual = np.transpose(np.vstack((np.multiply(flux[:, 0], lengths),
                                           np.multiply(flux[:, 1], lengths),
                                           np.multiply(flux[:, 2], lengths),
                                           np.multiply(flux[:, 3], lengths))))

    s_max = np.multiply(s_max, lengths)

    # TODO: Why does this have to be a loop vs just doing += on the whole array?
    for i in range(leftFilter.shape[0]):
        internalResiduals[leftFilter[i]] += fluxResidual[i]
        internalResiduals[rightFilter[i]] -= fluxResidual[i]

        sumSL[leftFilter[i]] += s_max[i]
        sumSL[rightFilter[i]] += s_max[i]

    return internalResiduals, sumSL


@njit(cache=True)
def inflowBEFlux(BE, state, u, v, sqrtRho, H, fluxEuler, norms, lengths, freestreamState, freestreamPressure, fluxFreestream, y):
    """Calculates all the wall fluxes for freestream boundary condition.

    Parameters
    ----------
    :param BE: [:, 4] Numpy array of boundary edge information [nodeA, nodeB, cell i, BE flag]
    :param u: [:, 1] Numpy array of x-dir velocities
    :param v: [:, 1] Numpy array of y-dir velocities
    :param rho: [:, 1] Numpy array of density
    :param rhoE: [:, 1] Numpy array of density*Energy
    :param a: [:, 1] Numpy array of local speed of sounds
    :param norms: [:, 2] Numpy array of normal vectors for the edges
    :param lengths: [:, 1] NUmpy array of edge lengths
    :param freestreamState: [1, 4] Numpy array of state values for the freestream condition
    :param y: Ratio of specifics heats - gamma

    Returns
    -------
    :returns: 1). wallResidual - [:, 4] array of wall BE flux, mostly zeroes except for locations that are walls, 2). sumSL -
    sum of the propagation speeds for each cell
    """
    inflowResidual = np.zeros((u.shape[0], 4), dtype=np.float64)
    sumSL = np.zeros((u.shape[0]), dtype=np.float64)

    # Down-select to only location where the supersonic exit BC applies
    if np.unique(BE[:, 3]).shape[0] == 3:
        inflowBEFilter = BE[BE[:, 3] == 2, 2]
        inflowNorms = norms[BE[:, 3] == 2, :]
        inflowLengths = lengths[BE[:, 3] == 2]
    else:
        inflowBEFilter = BE[BE[:, 3] == 3, 2]
        inflowNorms = norms[BE[:, 3] == 3, :]
        inflowLengths = lengths[BE[:, 3] == 3]

    freestreamStateArray = np.ones((inflowBEFilter.shape[0], 4), dtype=np.float64)
    freestreamStateArray[:, 0] *= freestreamState[0]
    freestreamStateArray[:, 1] *= freestreamState[1]
    freestreamStateArray[:, 2] *= freestreamState[2]
    freestreamStateArray[:, 3] *= freestreamState[3]

    deltaState = freestreamStateArray - state[inflowBEFilter]

    # "Left" is the interior state, while "right" is the freestream state
    sqrtRhoLeft = sqrtRho[inflowBEFilter]
    sqrtRhoRight = np.ones((sqrtRhoLeft.shape[0])) * np.sqrt(freestreamState[0])

    uLeft = u[inflowBEFilter]
    uRight = np.ones((sqrtRhoLeft.shape[0])) * freestreamState[1] / freestreamState[0]

    vLeft = v[inflowBEFilter]
    vRight = np.ones((sqrtRhoLeft.shape[0])) * freestreamState[2] / freestreamState[0]

    enthalpyLeft = H[inflowBEFilter]
    enthalpyRight = np.ones((sqrtRhoLeft.shape[0])) * (freestreamState[3] + freestreamPressure) / freestreamState[0]

    # Roe-Average Quantities
    roeAvgU = np.divide(np.multiply(sqrtRhoLeft, uLeft) + np.multiply(sqrtRhoRight, uRight), (sqrtRhoLeft + sqrtRhoRight))
    roeAvgV = np.divide(np.multiply(sqrtRhoLeft, vLeft) + np.multiply(sqrtRhoRight, vRight), (sqrtRhoLeft + sqrtRhoRight))
    roeAvgEnthalpy = np.divide(np.multiply(sqrtRhoLeft, enthalpyLeft) + np.multiply(sqrtRhoRight, enthalpyRight), (sqrtRhoLeft + sqrtRhoRight))
    qRoeSquared = np.multiply(roeAvgU, roeAvgU) + np.multiply(roeAvgV, roeAvgV)

    # Speed of Sound
    c = np.sqrt((y - 1) * (roeAvgEnthalpy - qRoeSquared / 2))

    # Local Speeds
    u = np.multiply(roeAvgU, inflowNorms[:, 0]) + np.multiply(roeAvgV, inflowNorms[:, 1])

    # System's eigenvalues
    eigenvalues = np.transpose(np.abs(np.vstack((u + c, u - c, u))))

    # Entropy fix
    for i in range(eigenvalues.shape[0]):
        for j in range(3):
            if eigenvalues[i, j] < (0.05 * c[i]):
                eigenvalues[i, j] = ((0.0025 * c[i] * c[i]) + (eigenvalues[i, j] * eigenvalues[i, j])) / (0.1 * c[i])

    # Maximum propagation speed
    s_max = c + np.abs(u)

    # Intermediate constants
    s = np.transpose(np.vstack((0.5 * (eigenvalues[:, 0] + eigenvalues[:, 1]),
                               0.5 * (eigenvalues[:, 0] - eigenvalues[:, 1]))))

    g1 = (y - 1) * (np.multiply(qRoeSquared / 2,  deltaState[:, 0]) -
                   (np.multiply(roeAvgU, deltaState[:, 1]) + np.multiply(roeAvgV, deltaState[:, 2])) + deltaState[:, 3])
    g2 = np.multiply(-u, deltaState[:, 0]) + (np.multiply(deltaState[:, 1], inflowNorms[:, 0]) + np.multiply(deltaState[:, 2], inflowNorms[:, 1]))

    c1 = np.multiply(np.divide(g1, np.multiply(c, c)), (s[:, 0] - eigenvalues[:, 2])) + np.multiply(np.divide(g2, c), s[:, 1])
    c2 = np.multiply(np.divide(g1, c), s[:, 1]) + np.multiply(s[:, 0] - eigenvalues[:, 2], g2)

    # Euler Equation Fluxes
    fluxLeft = fluxEuler[inflowBEFilter, :, :]

    fluxRight = np.ones((inflowBEFilter.shape[0], 4, 2))
    fluxRight[:, 0, :] *= fluxFreestream[0, :]
    fluxRight[:, 1, :] *= fluxFreestream[1, :]
    fluxRight[:, 2, :] *= fluxFreestream[2, :]
    fluxRight[:, 3, :] *= fluxFreestream[3, :]

    fluxLeftx = np.transpose(np.vstack((fluxLeft[:, 0, 0] * inflowNorms[:, 0],
                fluxLeft[:, 1, 0] * inflowNorms[:, 0] ,
                fluxLeft[:, 2, 0] * inflowNorms[:, 0] ,
                fluxLeft[:, 3, 0] * inflowNorms[:, 0])))

    fluxLefty = np.transpose(np.vstack((fluxLeft[:, 0, 1] * inflowNorms[:, 1],
                fluxLeft[:, 1, 1] * inflowNorms[:, 1],
                fluxLeft[:, 2, 1] * inflowNorms[:, 1] ,
                fluxLeft[:, 3, 1] * inflowNorms[:, 1])))

    fluxLeft = fluxLeftx + fluxLefty

    fluxRightx = np.transpose(np.vstack((fluxRight[:, 0, 0] * inflowNorms[:, 0] ,
                fluxRight[:, 1, 0] * inflowNorms[:, 0] ,
                fluxRight[:, 2, 0] * inflowNorms[:, 0] ,
                fluxRight[:, 3, 0] * inflowNorms[:, 0])))

    fluxRighty = np.transpose(np.vstack((fluxRight[:, 0, 1] * inflowNorms[:, 1] ,
                fluxRight[:, 1, 1] * inflowNorms[:, 1] ,
                fluxRight[:, 2, 1] * inflowNorms[:, 1] ,
                fluxRight[:, 3, 1] * inflowNorms[:, 1])))

    fluxRight = fluxRightx + fluxRighty

    # Roe Flux
    flux = 0.5 * (fluxLeft + fluxRight) - \
           0.5 * np.transpose(np.vstack((np.multiply(eigenvalues[:, 2], deltaState[:, 0]) + c1,
                                         np.multiply(eigenvalues[:, 2], deltaState[:, 1]) + np.multiply(c1, roeAvgU) + np.multiply(c2, inflowNorms[:, 0]),
                                         np.multiply(eigenvalues[:, 2], deltaState[:, 2]) + np.multiply(c1, roeAvgV) + np.multiply(c2, inflowNorms[:, 1]),
                                         np.multiply(eigenvalues[:, 2], deltaState[:, 3]) + np.multiply(c1, roeAvgEnthalpy) + np.multiply(c2, u))))

    fluxResidual = np.transpose(np.vstack((np.multiply(flux[:, 0], inflowLengths),
                                          np.multiply(flux[:, 1], inflowLengths),
                                          np.multiply(flux[:, 2], inflowLengths),
                                          np.multiply(flux[:, 3], inflowLengths))))

    s_max = np.multiply(s_max, inflowLengths)

    # TODO: Why does this have to be a loop vs just doing += on the whole array?
    for i in range(inflowBEFilter.shape[0]):
        inflowResidual[inflowBEFilter[i]] += fluxResidual[i]
        sumSL[inflowBEFilter[i]] += s_max[i]

    return inflowResidual, sumSL


@njit(cache=True)
def outflowBEFlux(BE, state, u, v, sqrtRho, H, fluxEuler, norms, lengths, y):
    """Calculates all the wall fluxes for a supersonic exit boundary condition.

    Parameters
    ----------
    :param BE: [:, 4] Numpy array of boundary edge information [nodeA, nodeB, cell i, BE flag]
    :param u: [:, 1] Numpy array of x-dir velocities
    :param v: [:, 1] Numpy array of y-dir velocities
    :param a: [:, 1] Numpy array of local speed of sounds
    :param norms: [:, 2] Numpy array of normal vectors for the edges
    :param lengths: [:, 1] NUmpy array of edge lengths
    :param fluxEuler: [:, 4, 2] Numpy array of euler fluxes, first col is cell index, second is flux, third is direction

    Returns
    -------
    :returns: 1). exitResidual - [:, 4] array of wall BE flux, mostly zeroes except for locations that are walls, 2). sumSL -
    sum of the propagation speeds for each cell
    """
    outflowResidual = np.zeros((u.shape[0], 4), dtype=np.float64)
    sumSL = np.zeros((u.shape[0]), dtype=np.float64)

    # Down-select to only location where the supersonic exit BC applies
    if np.unique(BE[:, 3]).shape[0] == 3:
        exitBEFilter = BE[BE[:, 3] == 1, 2]
        exitNorms = norms[BE[:, 3] == 1, :]
        exitLengths = lengths[BE[:, 3] == 1]
    else:
        jointFilter = np.logical_or(BE[:, 3] == 1, BE[:, 3] == 2)
        exitBEFilter = BE[jointFilter, 2]
        exitNorms = norms[jointFilter, :]
        exitLengths = lengths[jointFilter]

    deltaState = state[exitBEFilter] - state[exitBEFilter]

    # "Left" is the interior state, while "right" is the freestream state
    sqrtRhoLeft = sqrtRho[exitBEFilter]
    sqrtRhoRight = sqrtRho[exitBEFilter]

    uLeft = u[exitBEFilter]
    uRight = u[exitBEFilter]

    vLeft = v[exitBEFilter]
    vRight = v[exitBEFilter]

    enthalpyLeft = H[exitBEFilter]
    enthalpyRight = H[exitBEFilter]

    # Roe-Average Quantities
    roeAvgU = np.divide(np.multiply(sqrtRhoLeft, uLeft) + np.multiply(sqrtRhoRight, uRight), (sqrtRhoLeft + sqrtRhoRight))
    roeAvgV = np.divide(np.multiply(sqrtRhoLeft, vLeft) + np.multiply(sqrtRhoRight, vRight), (sqrtRhoLeft + sqrtRhoRight))
    roeAvgEnthalpy = np.divide(np.multiply(sqrtRhoLeft, enthalpyLeft) + np.multiply(sqrtRhoRight, enthalpyRight), (sqrtRhoLeft + sqrtRhoRight))
    qRoeSquared = np.multiply(roeAvgU, roeAvgU) + np.multiply(roeAvgV, roeAvgV)

    # Speed of Sound
    c = np.sqrt((y - 1) * (roeAvgEnthalpy - qRoeSquared / 2))

    # Local Speeds
    u = np.multiply(roeAvgU, exitNorms[:, 0]) + np.multiply(roeAvgV, exitNorms[:, 1])

    # System's eigenvalues
    eigenvalues = np.transpose(np.abs(np.vstack((u + c, u - c, u))))

    # Entropy fix
    for i in range(eigenvalues.shape[0]):
        for j in range(3):
            if eigenvalues[i, j] < (0.05 * c[i]):
                eigenvalues[i, j] = ((0.0025 * c[i] * c[i]) + (eigenvalues[i, j] * eigenvalues[i, j])) / (0.1 * c[i])

    # Maximum propagation speed
    s_max = c + np.abs(u)

    # Intermediate constants
    s = np.transpose(np.vstack((0.5 * (eigenvalues[:, 0] + eigenvalues[:, 1]),
                               0.5 * (eigenvalues[:, 0] - eigenvalues[:, 1]))))

    g1 = (y - 1) * (np.multiply(qRoeSquared / 2,  deltaState[:, 0]) -
                   (np.multiply(roeAvgU, deltaState[:, 1]) + np.multiply(roeAvgV, deltaState[:, 2])) + deltaState[:, 3])
    g2 = np.multiply(-u, deltaState[:, 0]) + (np.multiply(deltaState[:, 1], exitNorms[:, 0]) + np.multiply(deltaState[:, 2], exitNorms[:, 1]))

    c1 = np.multiply(np.divide(g1, np.multiply(c, c)), (s[:, 0] - eigenvalues[:, 2])) + np.multiply(np.divide(g2, c), s[:, 1])
    c2 = np.multiply(np.divide(g1, c), s[:, 1]) + np.multiply(s[:, 0] - eigenvalues[:, 2], g2)

    # Euler Equation Fluxes
    fluxLeft = fluxEuler[exitBEFilter, :, :]

    fluxRight = fluxEuler[exitBEFilter, :, :]

    fluxLeftx = np.transpose(np.vstack((fluxLeft[:, 0, 0] * exitNorms[:, 0],
                fluxLeft[:, 1, 0] * exitNorms[:, 0] ,
                fluxLeft[:, 2, 0] * exitNorms[:, 0] ,
                fluxLeft[:, 3, 0] * exitNorms[:, 0])))

    fluxLefty = np.transpose(np.vstack((fluxLeft[:, 0, 1] * exitNorms[:, 1],
                fluxLeft[:, 1, 1] * exitNorms[:, 1],
                fluxLeft[:, 2, 1] * exitNorms[:, 1] ,
                fluxLeft[:, 3, 1] * exitNorms[:, 1])))

    fluxLeft = fluxLeftx + fluxLefty

    fluxRightx = np.transpose(np.vstack((fluxRight[:, 0, 0] * exitNorms[:, 0] ,
                fluxRight[:, 1, 0] * exitNorms[:, 0] ,
                fluxRight[:, 2, 0] * exitNorms[:, 0] ,
                fluxRight[:, 3, 0] * exitNorms[:, 0])))

    fluxRighty = np.transpose(np.vstack((fluxRight[:, 0, 1] * exitNorms[:, 1] ,
                fluxRight[:, 1, 1] * exitNorms[:, 1] ,
                fluxRight[:, 2, 1] * exitNorms[:, 1] ,
                fluxRight[:, 3, 1] * exitNorms[:, 1])))

    fluxRight = fluxRightx + fluxRighty

    # Roe Flux
    flux = 0.5 * (fluxLeft + fluxRight) - \
           0.5 * np.transpose(np.vstack((np.multiply(eigenvalues[:, 2], deltaState[:, 0]) + c1,
                              np.multiply(eigenvalues[:, 2], deltaState[:, 1]) + np.multiply(c1, roeAvgU) + np.multiply(c2, exitNorms[:, 0]),
                              np.multiply(eigenvalues[:, 2], deltaState[:, 2]) + np.multiply(c1, roeAvgV) + np.multiply(c2, exitNorms[:, 1]),
                              np.multiply(eigenvalues[:, 2], deltaState[:, 3]) + np.multiply(c1, roeAvgEnthalpy) + np.multiply(c2, u))))

    fluxResidual = np.transpose(np.vstack((np.multiply(flux[:, 0], exitLengths),
                                          np.multiply(flux[:, 1], exitLengths),
                                          np.multiply(flux[:, 2], exitLengths),
                                          np.multiply(flux[:, 3], exitLengths))))

    s_max = np.multiply(s_max, exitLengths)

    # TODO: Why does this have to be a loop vs just doing += on the whole array?
    for i in range(exitBEFilter.shape[0]):
        outflowResidual[exitBEFilter[i]] += fluxResidual[i]
        sumSL[exitBEFilter[i]] += s_max[i]

    return outflowResidual, sumSL


@njit(cache=True)
def outflowBEFluxFast(BE, u, v, a, fluxEuler, norms, lengths):
    """Calculates all the wall fluxes for a supersonic exit boundary condition.

    Parameters
    ----------
    :param BE: [:, 4] Numpy array of boundary edge information [nodeA, nodeB, cell i, BE flag]
    :param u: [:, 1] Numpy array of x-dir velocities
    :param v: [:, 1] Numpy array of y-dir velocities
    :param a: [:, 1] Numpy array of local speed of sounds
    :param norms: [:, 2] Numpy array of normal vectors for the edges
    :param lengths: [:, 1] NUmpy array of edge lengths
    :param fluxEuler: [:, 4, 2] Numpy array of euler fluxes, first col is cell index, second is flux, third is direction

    Returns
    -------
    :returns: 1). exitResidual - [:, 4] array of wall BE flux, mostly zeroes except for locations that are walls, 2). sumSL -
    sum of the propagation speeds for each cell
    """
    outflowResidual = np.zeros((u.shape[0], 4), dtype=np.float64)
    sumSL = np.zeros((u.shape[0]), dtype=np.float64)

    # Down-select to only location where the supersonic exit BC applies
    if np.unique(BE[:, 3]).shape[0] == 3:
        exitBEFilter = BE[BE[:, 3] == 1, 2]
        exitNorms = norms[BE[:, 3] == 1, :]
        exitLengths = lengths[BE[:, 3] == 1]
    else:
        jointFilter = np.logical_or(BE[:, 3] == 1, BE[:, 3] == 2)
        exitBEFilter = BE[jointFilter, 2]
        exitNorms = norms[jointFilter, :]
        exitLengths = lengths[jointFilter]


    uExitBE = u[exitBEFilter]
    vExitBE = v[exitBEFilter]
    aExitBE = a[exitBEFilter]
    fluxEulerExitBE = fluxEuler[exitBEFilter]

    # Local speed
    u = np.multiply(uExitBE, exitNorms[:, 0]) + np.multiply(vExitBE, exitNorms[:, 1])

    # Maximum propagation speed
    s_max = aExitBE + np.abs(u)

    # All the simplifications show that the Roe-flux simplifies into the Euler Flux when the two states are identical
    flux = np.vstack((np.multiply(fluxEulerExitBE[:, 0, 0], exitNorms[:, 0]) +
                          np.multiply(fluxEulerExitBE[:, 0, 1], exitNorms[:, 1]),
                          np.multiply(fluxEulerExitBE[:, 1, 0], exitNorms[:, 0]) +
                          np.multiply(fluxEulerExitBE[:, 1, 1], exitNorms[:, 1]),
                          np.multiply(fluxEulerExitBE[:, 2, 0], exitNorms[:, 0]) +
                          np.multiply(fluxEulerExitBE[:, 2, 1], exitNorms[:, 1]),
                          np.multiply(fluxEulerExitBE[:, 3, 0], exitNorms[:, 0]) +
                          np.multiply(fluxEulerExitBE[:, 3, 1], exitNorms[:, 1])))
    flux = np.transpose(flux)

    fluxResidual = np.transpose(np.vstack((np.multiply(flux[:, 0], exitLengths),
                                          np.multiply(flux[:, 1], exitLengths),
                                          np.multiply(flux[:, 2], exitLengths),
                                          np.multiply(flux[:, 3], exitLengths))))

    s_max = np.multiply(s_max, exitLengths)

    # TODO: Why does this have to be a loop vs just doing += on the whole array?
    for i in range(exitBEFilter.shape[0]):
        outflowResidual[exitBEFilter[i]] += fluxResidual[i]
        sumSL[exitBEFilter[i]] += s_max[i]

    return outflowResidual, sumSL

@njit(cache=True)
def wallBEFlux(BE, u, v, rho, rhoE, a, norms, lengths, y):
    """Calculates all the wall fluxes for an inviscid wall boundary condition.

    Parameters
    ----------
    :param BE: [:, 4] Numpy array of boundary edge information [nodeA, nodeB, cell i, BE flag]
    :param u: [:, 1] Numpy array of x-dir velocities
    :param v: [:, 1] Numpy array of y-dir velocities
    :param rho: [:, 1] Numpy array of density
    :param rhoE: [:, 1] Numpy array of density*Energy
    :param a: [:, 1] Numpy array of local speed of sounds
    :param norms: [:, 2] Numpy array of normal vectors for the edges
    :param lengths: [:, 1] NUmpy array of edge lengths
    :param y: Ratio of specifics heats - gamma

    Returns
    -------
    :returns: 1). wallResidual - [:, 4] array of wall BE flux, mostly zeroes except for locations that are walls, 2). sumSL -
    sum of the propagation speeds for each cell
    """
    wallResidual = np.zeros((u.shape[0], 4), dtype=np.float64)
    sumSL = np.zeros((u.shape[0]), dtype=np.float64)

    # Filtering down to the wall boundary edges
    wallBEFilter = BE[BE[:, 3] == 0, 2]

    uWallBE = u[wallBEFilter]
    vWallBE = v[wallBEFilter]
    normWallBE = norms[BE[:, 3] == 0]
    rhoEWallBE = rhoE[wallBEFilter]
    rhoWallBE = rho[wallBEFilter]
    aWallBE = a[wallBEFilter]
    lengthsWallBE = lengths[BE[:, 3] == 0]

    # Compute the flux (pressure) to maintain the inviscid wall boundary condition
    Vplus = np.zeros((uWallBE.shape[0], 2))
    Vplus[:, 0] = uWallBE
    Vplus[:, 1] = vWallBE

    vPlusDotNorm = np.multiply(uWallBE, normWallBE[:, 0]) + np.multiply(vWallBE, normWallBE[:, 1])

    wallBEVel = Vplus - np.multiply(np.transpose(np.vstack((vPlusDotNorm, vPlusDotNorm))), normWallBE)

    magnitudeSqrdBEVel = np.multiply(wallBEVel[:, 0], wallBEVel[:, 0]) + np.multiply(wallBEVel[:, 1], wallBEVel[:, 1])

    wallBEPressure = (y - 1) * (rhoEWallBE - 0.5 * np.multiply(rhoWallBE, magnitudeSqrdBEVel))

    # Add the residual (px * l, py * l) to the residual array
    wallResidual[wallBEFilter, 1] += np.multiply(wallBEPressure, np.multiply(normWallBE[:, 0], lengthsWallBE))
    wallResidual[wallBEFilter, 2] += np.multiply(wallBEPressure, np.multiply(normWallBE[:, 1], lengthsWallBE))

    # Add the propagation speed to the propagation speed array
    sumSL[wallBEFilter] += np.multiply(np.abs(np.add(np.multiply(uWallBE, normWallBE[:, 0]),
                                                     np.multiply(vWallBE, normWallBE[:, 1])))
                                       + aWallBE, lengthsWallBE)

    return wallResidual, sumSL