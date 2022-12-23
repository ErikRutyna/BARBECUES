import preprocess as pp
import initialization as initzn
import numpy as np
import math
import cell_geometry_formulas as cgf
import compressible_flow_formulas as cff
from numba import njit

def calculate_atpr(stag_pressure, mesh):
    """Calculates the average total pressure recovered (ATPR) along the exit plane of the isolator.

    :param stag_pressure: Nx1 array of stagnation pressure
    :param mesh: Mesh to get boundary edges
    :return: Double that is the average total pressure recovery (ATPR)
    """
    # Boundary Edges at the isolator exit
    exit_edges = mesh['BE'][mesh['BE'][:, 3] == 1, :]

    # Boundary edge stagnation pressures
    boundary_stagnation = stag_pressure[exit_edges[:, 2]]

    # Freestream stagnation pressure - maximum of all values should always be freestream
    freestream_stagnation = max(stag_pressure)

    d = 0
    delta_y = np.zeros(len(exit_edges))
    for i in range(len(delta_y)):
        # delta_y = y_nodeB - y_nodeA
        delta_y[i] = abs(mesh['V'][exit_edges[i, 1]][1] - mesh['V'][exit_edges[i, 0]][1])
        # Total length of the exit plane
        d += delta_y[i]

    # Numerical integration via summation over exit plane edges
    atpr = 1 / d / freestream_stagnation * np.sum(np.multiply(boundary_stagnation, delta_y))

    return atpr


def calculate_stagnation_pressure(state, mach):
    """Calculates the stagnation pressure for each state vector

    :param state: Nx4 array of state variables - one per cell [rho, rho*u, rho*v, rho*E]
    :param mach: Nx1 array of the local Mach number in each cell
    :return: Nx1 array of stagnation pressures - one per cell
    """
    # Constants to be used in the stagnation pressure formula
    c_1 = (pp.fluid_con['y'] - 1) / 2
    c_2 = pp.fluid_con['y'] / (pp.fluid_con['y'] - 1)

    # Local static pressure per state vector
    pressure = calculate_static_pressure(state)
    # Stagnation pressure
    stagnation_pressure = np.multiply(np.power(1 + c_1 * np.power(mach, 2), c_2), pressure)

    return stagnation_pressure


def calculate_mach(state):
    """Calculates the Mach number for each unique state vector.

    :param state: Local state vector in a given cell
    :return: Returns mach number
    """
    # Velocity magnitude
    q = np.sqrt(np.power(np.divide(state[:, 1], state[:, 0]), 2) + np.power(np.divide(state[:, 2], state[:, 0]), 2))

    # Use speed of sound, c = sqrt(y*p/rho)
    p = calculate_static_pressure(state)
    c = math.sqrt(pp.fluid_con['y']) * np.sqrt(np.divide(p, state[:, 0]))

    mach = np.divide(q, c)

    return mach


def calculate_mach_single(state):
    """Calculates Mach number for a single state vector slice - only exists because I was too lazy to fix the original
    mach_calc function to account for single slices and whole mesh arrays.

    :param state: Local state vector in a given cell
    :return: Returns mach number
    """
    # Velocity
    u = state[1] / state[0]
    v = state[2] / state[0]
    q = np.linalg.norm([u, v])

    # Enthalpy
    h = (state[3] + calculate_static_pressure(state)) / state[0]

    # Speed of sound
    c = math.sqrt((pp.fluid_con['y'] - 1) * (h - q ** 2 / 2))

    mach = q / c

    return mach


def calculate_static_pressure(state):
    """Calculates local static pressure from the given state.

    :param state: Nx4 array of Euler state variables
    :return: pressure: local static pressure for each state vector
    """
    # Leading constant term
    c = (pp.fluid_con['y'] - 1)
    # Velocity term
    q = np.power(np.divide(state[:, 1], state[:, 0]), 2) + np.power(np.divide(state[:, 2], state[:, 0]), 2)
    # p = (y - 1) * (rho*E - 0.5 * rho * q^2)
    pressure = c * (state[:, 3] - 0.5 * np.multiply(state[:, 0], q))

    return pressure


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


def calculate_pressure_coefficient(state, y):
    """Calculate the pressure coefficient according to the formula C_p = (P - P_infty) / (P0 - P_infty)

    :param state: Nx4 state vector set
    :param y: Ratio of specific heats - gamma
    :return: cp - array of pressure coefficients
    """
    # Static pressures
    p = calculate_static_pressure(state)
    # Stagnation pressures
    p0 = calculate_stagnation_pressure(state, calculate_mach(state))
    # Freestream static pressure from freestream state
    pinf = calculate_static_pressure_single(initzn.generate_freestream_state(), y)

    cp = np.divide(p - pinf, p0 - pinf)

    return cp


def calculate_forces_moments(mesh, state):
    """Calculate the drag coefficients, lift coefficient, and the pitching moment about the origin of the domain (0,0).

    :param mesh: Mesh in the loaded *.gri format
    :param state: Nx4 state vector set
    :return: cd, cl, cmx (force and moment coefficients)
    """
    cp = calculate_pressure_coefficient(state, pp.fluid_con['y'])

    cd = 0
    cl = 0
    cmx = 0

    # Projected x-length of the object in the flow
    l_tot = abs((mesh['V'][mesh['BE'][np.where(np.array(mesh['Bname']) == 'Wall')[0][0] == mesh['BE'][:, 3]][:, 0:2]][:, 0]).max() - \
                (mesh['V'][mesh['BE'][np.where(np.array(mesh['Bname']) == 'Wall')[0][0] == mesh['BE'][:, 3]][:, 0:2]][:, 0]).min())

    wall_bes = mesh['BE'][np.where(np.array(mesh['Bname']) == 'Wall')[0][0] == mesh['BE'][:, 3]]

    for be in wall_bes:
        l, n = cgf.edge_properties_calculator(mesh['V'][be[0]], mesh['V'][be[1]])
        # Pressure integrals for lift and drag
        cl += 1 / l_tot * cp[be[2]] * -n[1] * l * math.cos(pp.flight_con['angle_of_attack'])
        cd += 1 / l_tot * cp[be[2]] * n[0] * l

        # Moment = F x d, d is defined as distance from origin to midpoint of edge
        midpoint = (mesh['V'][be[0]] + mesh['V'][be[1]]) / 2
        cmx += (cp[be[2]] * n[0] * l * midpoint[0] - cp[be[2]] * n[1] * l * midpoint[1]) / l_tot ** 2

    return cd, cl, cmx

def calculate_plate_friction(mesh, state):
    """Calculate the skin drag coefficient for a flat plate (boundary section) according to the NASA paper.

    :param mesh: Mesh in the loaded *.gri format
    :param state: Nx4 state vector set
    :return: cf_approx
    """
    cf_approx = 0

    wall_bes = mesh['BE'][np.where(np.array(mesh['Bname']) == 'Wall')[0][0] == mesh['BE'][:, 3]]

    # Additional freestream quantities for calculating a viscous drag coefficient
    mu_inf = cff.sutherland_viscosity(pp.flight_con['tinf'])
    Re_x = pp.flight_con['pinf'] / (pp.flight_con['tinf'] * pp.fluid_con['R']) * \
           pp.flight_con['freestream_mach_numer'] * math.sqrt(pp.flight_con['tinf'] * pp.fluid_con['R'] * pp.fluid_con['y']) / mu_inf

    for be in wall_bes:
        l, _ = cgf.edge_properties_calculator(mesh['V'][be[0]], mesh['V'][be[1]])
        # T = E / Cv
        t_local = state[be[2], 3] / state[be[2], 0] / pp.fluid_con['cv']
        mu_local = cff.sutherland_viscosity(t_local)
        cw = mu_local * pp.flight_con['tinf'] / (mu_inf * t_local)

        # From the NASA paper in the citations
        cf_local = 0.664 / (math.sqrt(Re_x * l / cw))
        cf_approx += cf_local

    return cf_approx