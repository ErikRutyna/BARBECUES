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
    if exit_edges.shape[0] == 0:
        return 0

    # Freestream stagnation pressure - maximum of all values should always be freestream
    freestream_stagnation = max(stag_pressure)

    # Boundary edge stagnation pressures
    boundary_stagnation = stag_pressure[exit_edges[:, 2]]

    # Length of the exit edges where stagnation pressure is measured
    exitEdgeIndices = exit_edges[:, 0:2]
    exitEdgeLengths, _ = cgf.edgePropertiesCalculator(exitEdgeIndices, exitEdgeIndices)

    totalLength = exitEdgeLengths.sum()

    # Formula for ATPR = deltaL * P0 / sum(deltaL) / P0inf
    atpr = 1 / totalLength / freestream_stagnation * \
           np.sum(np.multiply(boundary_stagnation, exitEdgeLengths))

    return atpr


@njit(cache=True)
def calculate_stagnation_pressure(state, mach, y):
    """Calculates the stagnation pressure for each state vector

    :param state: [:, 4] Numpy array of state variables, [rho, rho*u, rho*v, rho*E]
    :param mach: [:, 1] Numpy array of the local Mach number in each cell
    :param y: Ratio of specific heats of the working fluid, gamma
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
    :param y: Ratio of specific heats of the working fluid, gamma
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
    :param y: Ratio of specific heats of the working fluid, gamma
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
    :param y: Ratio of specific heats of the working fluid, gamma
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
def calculate_forces(V, BE, state, M, a, y):
    """Calculate the drag coefficients, lift coefficient, and the pitching moment about the origin of the domain (0,0).

    :param V: Node coodinates
    :param BE: Boundary Edge information [nodeA, nodeB, state i, edge identifier]
    :param state: Nx4 state vector set
    :param M: [:] Numpy array of cell Mach numbers
    :param a: Angle of Attack
    :param y: Ratio of specific heats of the working fluid, gamma
    :return: cd, cl, cmx (force and moment coefficients)
    """
    freestream_state = initzn.init_state_mach(M, a, y)
    Pinf = calculate_static_pressure_single(freestream_state, y)
    qinf = y / 2 * Pinf * M**2

    wallEdges = BE[BE[:, 3] == 0, :]
    wallEdgeIndices = wallEdges[:, 0:2]

    wallEdgeLengths, wallEdgeNorms = cgf.edgePropertiesCalculator(wallEdgeIndices, V)

    wallPressues = calculate_static_pressure(state[wallEdges[:, 2]], y)

    cx = (np.multiply(np.multiply(wallPressues, wallEdgeNorms[:, 0]), wallEdgeLengths)).sum()
    cy = (np.multiply(np.multiply(wallPressues, wallEdgeNorms[:, 1]), wallEdgeLengths)).sum()

    cd = (cy * np.sin(a * np.pi / 180) + cx * np.cos(a * np.pi / 180)) / qinf
    cl = (cy * np.cos(a * np.pi / 180) - cx * np.sin(a * np.pi / 180)) / qinf

    return cd, cl


@njit(cache=True)
def calculate_plate_friction(V, BE, state, mu_inf, Rex_inf, cv, tinf, mu_ref, t_ref, S):
    """Calculate the skin drag coefficient for a flat plate (boundary section) according to the NASA paper.

    :param V: Node coodinates
    :param BE: Boundary Edge information [nodeA, nodeB, state i, edge identifier]
    :param state: Nx4 state vector set
    :param mu_inf: Freestream viscosity in SI units
    :param Rex_inf: Reynolds number per unit length in SI units
    :param cv: Specific volume heating coefficient in SI units
    :param tinf: Freestream temperature
    :param mu_ref: Sutherland's reference viscosity in SI units
    :param t_ref: Sutherland's reference temperature in Kelvin
    :param S: Sutherland's constant in Kelvin
    :return: cf_approx
    """
    wallEdges = BE[BE[:, 3] == 1, :]
    wallEdgeIndices = wallEdges[:, 0:2]

    wallEdgeLengths, _ = cgf.edgePropertiesCalculator(wallEdgeIndices, V)

    wallEdgeTemps = np.divide(state[wallEdges[:, 2], 3], state[wallEdges[:, 2], 0]) / cv

    wallEdgeMu = cff.sutherland_viscosity(wallEdgeTemps, mu_ref, t_ref, S)

    wallCoeff = np.divide(wallEdgeMu * tinf, mu_inf * wallEdgeLengths)

    cf_approx = (0.664 / np.sqrt(np.divide(np.multiply(Rex_inf, wallEdgeLengths), wallCoeff))).sum()

    return cf_approx