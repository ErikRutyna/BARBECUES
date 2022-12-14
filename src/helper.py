import initialization as initzn
import numpy as np
import math
import cell_geometry_formulas as cgf
import compressible_flow_formulas as cff
from numba import njit


@njit(cache=True)
def calculate_atpr(V, BE, stag_pressure):
    """Calculates the average total pressure recovered (ATPR) along the exit plane of the isolator.

    :param V: Node coodinates
    :param BE: Boundary Edge information [nodeA, nodeB, state i, edge identifier]
    :param stag_pressure: Nx1 array of stagnation pressure
    :return: Double that is the average total pressure recovery (ATPR)
    """
    exit_edges = BE[BE[:, 3] == 1, :]

    # Boundary edge stagnation pressures
    boundary_stagnation = stag_pressure[exit_edges[:, 2]]

    # Freestream stagnation pressure - maximum of all values should always be freestream
    freestream_stagnation = max(stag_pressure)

    d = 0
    delta_y = np.zeros(len(exit_edges))
    for i in range(len(delta_y)):
        # delta_y = y_nodeB - y_nodeA
        delta_y[i], _ = cgf.edge_properties_calculator(V[exit_edges[i, 1]], V[exit_edges[i, 0]])
        # Total length of the exit plane
        d += delta_y[i]

    # Numerical integration via summation over exit plane edges
    atpr = 1 / d / freestream_stagnation * np.sum(np.multiply(boundary_stagnation, delta_y))

    return atpr


@njit(cache=True)
def calculate_stagnation_pressure(state, mach, y):
    """Calculates the stagnation pressure for each state vector

    :param state: Nx4 array of state variables - one per cell [rho, rho*u, rho*v, rho*E]
    :param mach: Nx1 array of the local Mach number in each cell
    :return: Nx1 array of stagnation pressures - one per cell
    """
    # Constants to be used in the stagnation pressure formula
    c_1 = (y - 1) / 2
    c_2 = y / (y - 1)

    # Local static pressure per state vector
    pressure = calculate_static_pressure(state, y)
    # Stagnation pressure
    stagnation_pressure = np.multiply(np.power(1 + c_1 * np.power(mach, 2), c_2), pressure)

    return stagnation_pressure


@njit(cache=True)
def calculate_mach(state, y):
    """Calculates the Mach number for each unique state vector.

    :param state: Local state vector in a given cell
    :return: Returns mach number
    """
    # Velocity magnitude
    q = np.sqrt(np.power(np.divide(state[:, 1], state[:, 0]), 2) + np.power(np.divide(state[:, 2], state[:, 0]), 2))

    # Use speed of sound, c = sqrt(y*p/rho)
    p = calculate_static_pressure(state, y)
    c = math.sqrt(y) * np.sqrt(np.divide(p, state[:, 0]))

    mach = np.divide(q, c)

    return mach


@njit(cache=True)
def calculate_mach_single(state, y):
    """Calculates Mach number for a single state vector slice - only exists because I was too lazy to fix the original
    mach_calc function to account for single slices and whole mesh arrays.

    :param state: Local state vector in a given cell
    :return: Returns mach number
    """
    # Velocity
    u = state[1] / state[0]
    v = state[2] / state[0]
    q = np.linalg.norm(np.array((u, v)))

    # Enthalpy
    h = (state[3] + calculate_static_pressure_single(state, y)) / state[0]

    # Speed of sound
    c = np.sqrt((y - 1) * (h - q ** 2 / 2))

    mach = q / c

    return mach


@njit(cache=True)
def calculate_static_pressure(state, y):
    """Calculates local static pressure from the given state.

    :param state: Nx4 array of Euler state variables
    :return: pressure: local static pressure for each state vector
    """
    # Leading constant term
    c = (y - 1)
    # Velocity term
    q = np.power(np.divide(state[:, 1], state[:, 0]), 2) + np.power(np.divide(state[:, 2], state[:, 0]), 2)
    # p = (y - 1) * (rho*E - 0.5 * rho * q^2)
    pressure = c * (state[:, 3] - 0.5 * np.multiply(state[:, 0], q))

    return pressure


@njit(cache=True)
def calculate_static_pressure_single(state, y):
    """Calculates local static pressure from the given state.

    :param state: Nx4 array of Euler state variables
    :param y: Ratio of specific heats - gamma
    :return: pressure: local static pressure for each state vector
    """
    # Leading constant term
    c = (y - 1)
    # Velocity term
    q = np.power(np.divide(state[1], state[0]), 2) + np.power(np.divide(state[2], state[0]), 2)
    # p = (y - 1) * (rho*E - 0.5 * rho * q^2)
    pressure = c * (state[3] - 0.5 * np.multiply(state[0], q))

    return pressure


@njit(cache=True)
def calculate_pressure_coefficient(state, M, a, y):
    """Calculate the pressure coefficient according to the formula C_p = (P - P_infty) / (P0 - P_infty)

    :param state: Nx4 state vector set
    :param M: Cell's local Mach number
    :param a: Freestream angle of attack
    :param y: Ratio of specific heats - gamma
    :return: cp - array of pressure coefficients
    """
    # Static pressures
    p = calculate_static_pressure(state, y)
    # Stagnation pressures
    p0 = calculate_stagnation_pressure(state, calculate_mach(state, y), y)
    # Freestream static pressure from freestream state
    pinf = calculate_static_pressure_single(initzn.init_state_mach(M, a, y), y)

    cp = np.divide(p - pinf, p0 - pinf)

    return cp


@njit(cache=True)
def calculate_forces_moments(V, BE, state, M, a, y):
    """Calculate the drag coefficients, lift coefficient, and the pitching moment about the origin of the domain (0,0).

    :param V: Node coodinates
    :param BE: Boundary Edge information [nodeA, nodeB, state i, edge identifier]
    :param state: Nx4 state vector set
    :param a: Angle of Attack
    :return: cd, cl, cmx (force and moment coefficients)
    """
    cp = calculate_pressure_coefficient(state, M, a, y)

    cd = 0
    cl = 0
    cmx = 0

    # Projected x-length of the object in the flow
    nodes = []
    for i in range(BE.shape[0]):
        if BE[i, 3] == 0:
            nodes.append(BE[i, 0])
            nodes.append(BE[i, 1])

    x_pos = []
    for i in range(len(nodes)):
        x_pos.append(V[nodes[i], 0])
    x_pos = np.array(x_pos)

    l_tot = np.abs(x_pos.max() - x_pos.min())

    for i in range(BE.shape[0]):
        if BE[i, 3] == 0:
            l, n = cgf.edge_properties_calculator(V[BE[i, 0]], V[BE[i, 1]])
            # Pressure integrals for lift and drag
            cl += 1 / l_tot * cp[BE[i, 2]] * -n[1] * l * math.cos(a)
            cd += 1 / l_tot * cp[BE[i, 2]] * n[0] * l

            # Moment = F x d, d is defined as distance from origin to midpoint of edge
            midpoint = (V[BE[i, 0]] + V[BE[i, 1]]) / 2
            cmx += (cp[BE[i, 2]] * n[0] * l * midpoint[0] - cp[BE[i, 2]] * n[1] * l * midpoint[1]) / l_tot ** 2

    return cd, cl, cmx


@njit(cache=True)
def calculate_plate_friction(V, BE, state, mu_inf, Rex_inf, cv, tinf, mu_ref, t_ref, S):
    """Calculate the skin drag coefficient for a flat plate (boundary section) according to the NASA paper.

    :param V: Node coodinates
    :param BE: Boundary Edge information [nodeA, nodeB, state i, edge identifier]
    :param state: Nx4 state vector set
    :return: cf_approx
    """
    cf_approx = 0

    for be in BE:
        if be[3] == 0:
            l, _ = cgf.edge_properties_calculator(V[be[0]], V[be[1]])
            # T = E / Cv
            t_local = state[be[2], 3] / state[be[2], 0] / cv
            mu_local = cff.sutherland_viscosity(t_local, mu_ref, t_ref, S)
            cw = mu_local * tinf / (mu_inf * t_local)

            # From the NASA paper in the citations
            cf_local = 0.664 / (math.sqrt(Rex_inf * l / cw))
            cf_approx += cf_local

    return cf_approx