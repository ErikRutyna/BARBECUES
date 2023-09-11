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
    residuals = np.zeros((state.shape[0], 4), dtype=np.float64)  # Residuals from fluxes
    sum_sl = np.transpose(np.zeros((state.shape[0]), dtype=np.float64))

    freestreamState = intlzn.init_state_mach(M, a, y)

    pressureFreestream = helper.calculate_static_pressure_single(freestreamState, y)

    sqrtRho, u, v, pressures, H, fluxEuler, mach = computeStateVals(state, y)

    # Wall boundary edge flux computations
    residualsWallBE, sumSLWallBE = wallBEFlux(BE, u, v, state[:, 0], state[:, 3], mach, be_n, be_l, y)
    residuals += residualsWallBE
    sum_sl += sumSLWallBE

    # Exit boundary edge flux computations
    residualsExitBEs, sumSLExitBE = supersonicExitBEFlux(BE,u, v, mach, be_n, be_l, fluxEuler)
    residuals += residualsExitBEs
    sum_sl += sumSLExitBE

    # fluxCheck = np.zeros((residuals.shape[0], 4))

    # Internal edges
    for i in range(IE.shape[0]):
        ie_flux, ie_smax = roeEuler2D(state[IE[i, 2]], state[IE[i, 3]],
                                      pressures[IE[i, 2]], pressures[IE[i, 3]], ie_n[i], y)

        residuals[IE[i, 2]] += ie_flux * ie_l[i]  # Summing residuals to be taken out of cell i
        residuals[IE[i, 3]] -= ie_flux * ie_l[i]  # Summing residuals to negative taken out of cell N (added to cell N)

        sum_sl[IE[i, 2]] += ie_smax * ie_l[i]
        sum_sl[IE[i, 3]] += ie_smax * ie_l[i]

    # Boundary edges with their respective conditions
    if np.unique(BE[:, 3]).shape[0] == 3:
        for i in range(BE.shape[0]):
            if BE[i, 3] == 0:
                continue

            elif BE[i, 3] == 1:
                # be_flux, be_smax = roeEuler2D(state[BE[i, 2]], state[BE[i, 2]],
                #                               pressures[BE[i, 2]], pressureFreestream, be_n[i], y)
                # fluxCheck[BE[i, 2]] += be_flux * be_l[i]
                continue

            elif BE[i, 3] == 2:
                # Apply freestream inflow boundary conditions
                be_flux, be_smax = roeEuler2D(state[BE[i, 2]], freestreamState,
                                              pressures[BE[i, 2]], pressureFreestream, be_n[i], y)

                residuals[BE[i, 2]] += be_flux * be_l[i]
                sum_sl[BE[i, 2]] += be_smax * be_l[i]
    else:
        for i in range(BE.shape[0]):
            if BE[i, 3] == 0:
                continue

            elif BE[i, 3] == 1 or BE[i, 3] == 2:
                # Apply supersonic outflow boundary conditions
                continue
            elif BE[i, 3] == 3:
                # Apply freestream inflow boundary conditions
                be_flux, be_smax = roeEuler2D(state[BE[i, 2]], freestreamState,
                                              pressures[BE[i, 2]], pressureFreestream, be_n[i], y)

                residuals[BE[i, 2]] += be_flux * be_l[i]
                sum_sl[BE[i, 2]] += be_smax * be_l[i]

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


# TODO: Write this function with the goal being to vectorize it as much as possible, ideally the whole thing can be vectorized such that no loops are needed
# @njit(cache=True)
def roeEuler2DInternalEdges(state, sqrtRho, u, v, p, fluxEuler, indexLeft, indexRight, norms, lengths, y):
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

    flux = 0
    sMax = 0
    return flux, sMax



def freestreamBEFlux(BE, u, v, rho, rhoE, a, norms, lengths, freestreamState, y):
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
    wallResidual = np.zeros((u.shape[0], 4))
    sumSL = np.zeros((u.shape[0]))

    return wallResidual, sumSL


@njit(cache=True)
def supersonicExitBEFlux(BE, u, v, a, norms, lengths, fluxEuler):
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
    exitResidual = np.zeros((u.shape[0], 4))
    sumSL = np.zeros((u.shape[0]))

    # Down-select to only location where the supersonic exit BC applies
    if np.unique(BE[:, 3]).shape[0] == 3:
        exitBEFilter = BE[BE[:, 3] == 1, 2]
    else:
        exitBEFilter = BE[np.logical_or(BE[:, 3] == 1, BE[:, 3] == 2), 2]

    uExitBE = u[exitBEFilter]
    vExitBE = v[exitBEFilter]
    aExitBE = a[exitBEFilter]
    normExitBE = norms[BE[:, 3] == 1, :]
    lengthExitBE = lengths[BE[:, 3] == 1]
    fluxEulerExitBE = fluxEuler[exitBEFilter]

    # All the simplifications show that the Roe-flux simplifies into the Euler Flux when the two states are identical
    exitFlux = np.vstack((np.multiply(fluxEulerExitBE[:, 0, 0], normExitBE[:, 0]) +
                          np.multiply(fluxEulerExitBE[:, 0, 1], normExitBE[:, 1]),
                         np.multiply(fluxEulerExitBE[:, 1, 0], normExitBE[:, 0]) +
                          np.multiply(fluxEulerExitBE[:, 1, 1], normExitBE[:, 1]),
                         np.multiply(fluxEulerExitBE[:, 2, 0], normExitBE[:, 0]) +
                          np.multiply(fluxEulerExitBE[:, 2, 1], normExitBE[:, 1]),
                         np.multiply(fluxEulerExitBE[:, 3, 0], normExitBE[:, 0]) +
                          np.multiply(fluxEulerExitBE[:, 3, 1], normExitBE[:, 1])))
    exitFlux = np.transpose(exitFlux)

    for i in range(exitFlux.shape[0]):
        exitFlux[i, :] *= lengthExitBE[i]

    exitResidual[exitBEFilter] += exitFlux

    # Local speed
    u = np.multiply(uExitBE, normExitBE[:, 0]) + np.multiply(vExitBE, normExitBE[:, 1])

    # Maximum propagation speed
    s_max = aExitBE + np.abs(u)

    sumSLTemp = np.multiply(s_max, lengthExitBE)
    sumSL[exitBEFilter] += sumSLTemp

    return exitResidual, sumSL

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
    wallResidual = np.zeros((u.shape[0], 4))
    sumSL = np.zeros((u.shape[0]))

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