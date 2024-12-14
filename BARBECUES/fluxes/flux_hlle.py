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
def hlle_euler_2d(u_l, u_r, n, y):
    """Computes the Roe Flux from the left cell into the right cell for
    the inviscid Euler Equations

    :param u_l: State vector in the left cell [rho, rho*u, rho*v, rho*E]
    :param u_r: State vector in the right cell [rho, rho*u, rho*v, rho*E]
    :param n: Unit normal vector pointing from left cell to right cell
    :param y: Ratio of specific heats of the working fluid
    :return: flux: Flux of state vector from left to right cell, s: Maximum propagation speed of the state variables
    """
    p_l = helper.calculate_static_pressure_single(u_l, y)
    F_left = stateFluxEuler2D(u_l, p_l)
    F_left = F_left[:, 0] * n[0] + F_left[:, 1] * n[1]

    p_r = helper.calculate_static_pressure_single(u_r, y)
    F_right = stateFluxEuler2D(u_r, p_r)
    F_right = F_right[:, 0] * n[0] + F_right[:, 1] * n[1]

    vel_l = np.array((u_l[1] / u_l[0], u_l[2] / u_l[0]))
    vel_r = np.array((u_r[1] / u_r[0], u_r[2] / u_r[0]))

    q_l = math.sqrt(vel_l[0]**2 + vel_l[1]**2)
    q_r = math.sqrt(vel_r[0]**2 + vel_r[1]**2)

    h_left = (u_l[3] + p_l) / u_l[0]
    h_right= (u_r[3] + p_r) / u_r[0]

    c_l = math.sqrt((y - 1) * (h_left - q_l ** 2 / 2))
    c_r = math.sqrt((y - 1) * (h_right - q_r ** 2 / 2))

    U_l = vel_l[0] * n[0] + vel_r[1] * n[1]
    U_r = vel_r[0] * n[0] + vel_r[1] * n[1]

    s_l_min = min(0, U_l - c_l)
    s_r_min = min(0, U_r - c_r)

    s_l_max = max(0, U_l + c_l)
    s_r_max = max(0, U_r + c_r)

    smin = min(s_l_min, s_r_min)
    smax = max(s_l_max, s_r_max)

    flux = 0.5 * (F_left + F_right) - 0.5 * (smax + smin) / (smax - smin) * (F_right - F_left)\
           + (smax * smin) / (smax - smin) * (u_r - u_l)

    s_max = max(abs(U_l) + c_l, abs(U_r) + c_r)

    return flux, s_max


@njit(cache=True)
def compute_residuals_hlle(IE, BE, state, be_l, be_n, ie_l, ie_n, M, a, y):
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

    freestream_state = intlzn.init_state_mach(M, a, y)

    # Internal edges
    for i in range(IE.shape[0]):
        ie_flux, ie_smax = hlle_euler_2d(state[IE[i, 2]], state[IE[i, 3]], ie_n[i], y)

        residuals[IE[i, 2]] += ie_flux * ie_l[i]  # Summing residuals to be taken out of cell i
        residuals[IE[i, 3]] -= ie_flux * ie_l[i]  # Summing residuals to negative taken out of cell N (added to cell N)

        sum_sl[IE[i, 2]] += ie_smax * ie_l[i]
        sum_sl[IE[i, 3]] += ie_smax * ie_l[i]

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
            be_pressure = (y - 1) * (state[BE[i, 2], 3] - 0.5 * state[BE[i, 2], 0] * (be_vel[0] ** 2 + be_vel[1] ** 2))

            # Enforcing no flow through with pressure condition
            be_flux = np.array((0, be_pressure * be_n[i, 0], be_pressure * be_n[i, 1], 0))

            # Calculate propagation speed
            local_p = helper.calculate_static_pressure_single(state[BE[i, 2]], y)
            h = (state[BE[i, 2], 3] + local_p) / state[BE[i, 2], 0]
            be_smax = abs(u_plus * be_n[i, 0] + v_plus * be_n[i, 1]) \
                      + math.sqrt((y - 1) * (h - np.linalg.norm(np.array((u_plus, v_plus))) ** 2 / 2))


        elif BE[i, 3] == 1 or BE[i, 3] == 2:
            # Apply supersonic outflow boundary conditions
            be_flux, be_smax = hlle_euler_2d(state[BE[i, 2]], state[BE[i, 2]], be_n[i], y)

        elif BE[i, 3] == 3:
            # Apply freestream inflow boundary conditions
            be_flux, be_smax = hlle_euler_2d(state[BE[i, 2]], freestream_state, be_n[i], y)

        residuals[BE[i, 2]] += be_flux * be_l[i]
        sum_sl[BE[i, 2]] += be_smax * be_l[i]

    return residuals, sum_sl


