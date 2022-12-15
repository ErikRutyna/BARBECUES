import math
import numpy as np
import helper
import mesh_processing


def F_euler_2d(u, fluid):
    """Computes the flux vector for the 2D compressible Euler equations from
    the state vector.

    :param u: State vector
    :param fluid: Class that contains information of the working fluid
    :return: The flux vector for the 2D compressible Euler equations, each column is a direction
    """
    F = np.zeros((4, 2))
    P = (fluid.y - 1) * (u[3] - 0.5 * (u[1] ** 2 / u[0] + u[2] ** 2 / u[0]))

    # Continuity
    F[0, 0] = u[1]
    F[0, 1] = u[2]

    # x-momentum
    F[1, 0] = u[1] ** 2 / u[0] + P
    F[1, 1] = u[1] * u[2] / u[0]

    # y-momentum
    F[2, 0] = u[1] * u[2] / u[0]
    F[2, 1] = u[2] ** 2 / u[0] + P

    # Energy
    F[3, 0] = (u[1] * u[3] + u[1] * P) / u[0]
    F[3, 1] = (u[2] * u[3] + u[2] * P) / u[0]

    return F

# TODO: See if any sort of simplification can be done here to save compute time
def roe_euler_2d(u_l, u_r, n, fluid):
    """Computes the Roe Flux from the left cell into the right cell for
    the inviscid Euler Equations

    :param u_l: State vector in the left cell [rho, rho*u, rho*v, rho*E]
    :param u_r: State vector in the right cell [rho, rho*u, rho*v, rho*E]
    :param n: Unit normal pointing from left cell to right cell
    :return: flux: Flux of state vector from left to right cell
    :return: s: Maximum propagation speed of the state variables
    """
    # Delta state
    delta_u = np.subtract(u_r, u_l)

    # Roe-average states
    roe_avg_u = (math.sqrt(u_l[0]) * u_l[1] / u_l[0] + math.sqrt(u_r[0]) * u_r[1] / u_r[0]) \
                / (math.sqrt(u_l[0]) + math.sqrt(u_r[0]))
    roe_avg_v = (math.sqrt(u_l[0]) * u_l[2] / u_l[0] + math.sqrt(u_r[0]) * u_r[2] / u_r[0]) \
                / (math.sqrt(u_l[0]) + math.sqrt(u_r[0]))
    q = np.linalg.norm(np.array([roe_avg_u, roe_avg_v]))

    # flux_left = F_euler_2d(u_l, fluid)
    # h_left = (flux_left[3][0] + flux_left[3][1]) / (flux_left[0][0] + flux_left[0][1])
    h_left = (u_l[3] + helper.calculate_static_pressure_single(u_l, fluid)) / u_l[0]

    # flux_right = F_euler_2d(u_r, fluid)
    # h_right = (flux_right[3][0] + flux_right[3][1]) / (flux_right[0][0] + flux_right[0][1])
    h_right = (u_r[3] + helper.calculate_static_pressure_single(u_r, fluid)) / u_r[0]

    roe_avg_h = (math.sqrt(u_l[0]) * h_left + math.sqrt(u_r[0]) * h_right) / (math.sqrt(u_l[0]) + math.sqrt(u_r[0]))

    c = math.sqrt((fluid.y - 1) * (roe_avg_h - (q ** 2) / 2))

    u = roe_avg_u * n[0] + roe_avg_v * n[1]

    # Eigenvalues of system
    eigens = np.abs(np.array([u + c, u - c, u, u]))

    # Entropy fix
    for i in range(len(eigens)):
        if eigens[i] < (0.05 * c):
            eigens[i] = ((0.05 * c) ** 2 + eigens[i] ** 2) / (2 * (0.05 * c))

    s_max = c + abs(u)

    # Intermediate constants
    s = np.array([0.5 * (eigens[0] + eigens[1]),
                  0.5 * (eigens[0] - eigens[1])])

    g1 = (fluid.y - 1) * (q ** 2 / 2 * delta_u[0] - (roe_avg_u * delta_u[1] + roe_avg_v * delta_u[2]) + delta_u[3])
    g2 = -u * delta_u[0] + (delta_u[1] * n[0] + delta_u[2] * n[1])

    c1 = g1 / c ** 2 * (s[0] - eigens[2]) + g2 / c * s[1]
    c2 = g1 / c * s[1] + (s[0] - eigens[2]) * g2

    # Roe Flux
    F_l = F_euler_2d(u_l, fluid)
    F_l = F_l[:, 0] * n[0] + F_l[:, 1] * n[1]
    F_r = F_euler_2d(u_r, fluid)
    F_r = F_r[:, 0] * n[0] + F_r[:, 1] * n[1]
    flux = 0.5 * (F_l + F_r) - 0.5 * np.array([eigens[2] * delta_u[0] + c1,
                                               eigens[2] * delta_u[1] + c1 * roe_avg_u + c2 * n[0],
                                               eigens[2] * delta_u[2] + c1 * roe_avg_v + c2 * n[1],
                                               eigens[2] * delta_u[3] + c1 * roe_avg_h + c2 * u])

    return flux, s_max


# TODO: Check that this flux method actually works
def hlle_euler_2d(u_l, u_r, n, fluid):
    """Computes the Roe Flux from the left cell into the right cell for
    the inviscid Euler Equations

    :param u_l: State vector in the left cell [rho, rho*u, rho*v, rho*E]
    :param u_r: State vector in the right cell [rho, rho*u, rho*v, rho*E]
    :param n: Unit normal pointing from left cell to right cell
    :return: flux: Flux of state vector from left to right cell
    :return: s: Maximum propagation speed of the state variables
    """
    # TODO: Add this flux method
    F_left = F_euler_2d(u_l, fluid)
    F_left = F_left[:, 0] * n[0] + F_left[:, 1] * n[1]

    F_right= F_euler_2d(u_r, fluid)
    F_right= F_right[:, 0] * n[0] + F_right[:, 1] * n[1]

    vel_l = [u_l[1] / u_l[0], u_l[2] / u_l[0]]
    vel_r = [u_r[1] / u_r[0], u_r[2] / u_r[0]]

    q_l = np.linalg.norm(vel_l)
    q_r = np.linalg.norm(vel_r)

    h_left = (u_l[3] + helper.calculate_static_pressure(u_l, fluid)) / u_l[0]
    h_right= (u_r[3] + helper.calculate_static_pressure(u_r, fluid)) / u_r[0]

    c_l = math.sqrt((fluid.y - 1) * (h_left - q_l ** 2 / 2))
    c_r = math.sqrt((fluid.y - 1) * (h_right - q_r ** 2 / 2))

    U_l = vel_l[0] * n[0] + vel_r[1] * n[1]
    U_r = vel_r[0] * n[0] + vel_r[1] * n[1]

    s_l_min = min(0, U_l - c_l)
    s_r_min = min(0, U_r - c_r)

    s_l_max = max(0, U_l + c_l)
    s_r_max = max(0, U_r + c_r)

    smin = min(s_l_min, s_r_min)
    smax = max(s_l_max, s_r_max)

    flux = 0.5 * (F_left + F_right)  - 0.5 * (smax + smin) / (smax - smin) * (F_right - F_left)\
           + (smax * smin) / (smax - smin) * (u_r - u_l)

    s_max = max(abs(U_l) + c_l, abs(U_r) + c_r)

    return flux, s_max


# Look for methods of vectorization of edge information, fluxes, etc
def compute_residuals_roe(config, mesh, state, be_l, be_n, ie_l, ie_n):
    """Computes the residuals and sum of speed*edge_lengths for the state on the given mesh using the Roe Flux method.

    :param config: Operating and working fluid conditions
    :param mesh: Computational domain in KFID GRI format
    :param state: Nx4 array of state values
    :param be_l: Boundary edge lengths
    :param be_n: Boundary edge normal vectors
    :param ie_l: Internal edge lengths
    :param ie_n: Internal edge normal vectors
    :return:
    """
    residuals = np.zeros((len(mesh['E']), 4))  # Residuals from fluxes
    sum_sl = np.transpose(np.zeros((len(mesh['E']))))

    # Internal edges
    for i in range(mesh['IE'].shape[0]):
        ie_flux, ie_smax = roe_euler_2d(state[mesh['IE'][i, 2]], state[mesh['IE'][i, 3]], ie_n[i], config)

        residuals[mesh['IE'][i, 2]] += ie_flux * ie_l[i]  # Summing residuals to be taken out of cell i
        residuals[mesh['IE'][i, 3]] -= ie_flux * ie_l[i]  # Summing residuals to negative taken out of cell N (added to cell N)

        sum_sl[mesh['IE'][i, 2]] += ie_smax * ie_l[i]
        sum_sl[mesh['IE'][i, 3]] += ie_smax * ie_l[i]

    # Boundary edges with their respective conditions
    for i in range(mesh['BE'].shape[0]):
        be_smax = 0  # In case there is some additional boundary condition
        be_flux = 0
        if mesh['Bname'][mesh['BE'][i, 3]] == 'Wall':
            # Apply inviscid wall boundary condition
            # Boundary velocity
            u_plus = state[mesh['BE'][i, 2], 1] / state[mesh['BE'][i, 2], 0]
            v_plus = state[mesh['BE'][i, 2], 2] / state[mesh['BE'][i, 2], 0]
            V_plus = np.array([u_plus, v_plus])
            be_vel = V_plus - np.multiply(np.dot(V_plus, be_n[i]), be_n[i])

            # Boundary pressure
            be_pressure = (config.y - 1) * (state[mesh['BE'][i, 2], 3] - 0.5 *
                                            state[mesh['BE'][i, 2], 0] * (be_vel[0] ** 2 + be_vel[1] ** 2))

            # Enforcing no flow through with pressure condition
            be_flux = [0, be_pressure * be_n[i, 0], be_pressure * be_n[i, 1], 0]
            local_p = helper.calculate_static_pressure_single(state[mesh['BE'][i, 2]], config)
            h = (state[mesh['BE'][i, 2], 3] + local_p) / state[mesh['BE'][i, 2], 0]
            be_smax = abs(u_plus * be_n[i, 0] + v_plus * be_n[i, 1]) \
                      + math.sqrt((config.y - 1) * (h - np.linalg.norm([u_plus, v_plus]) ** 2 / 2))


        elif mesh['Bname'][mesh['BE'][i, 3]] == 'Exit' or mesh['Bname'][mesh['BE'][i, 3]] == 'Outflow':
            # Apply supersonic outflow boundary conditions
            be_flux, be_smax = roe_euler_2d(state[mesh['BE'][i, 2]], state[mesh['BE'][i, 2]], be_n[i], config)

        elif mesh['Bname'][mesh['BE'][i, 3]] == 'Inflow':
            # Apply freestream inflow boundary conditions
            be_flux, be_smax = roe_euler_2d(state[mesh['BE'][i, 2]], helper.generate_freestream_state(config),
                                            be_n[i], config)

        residuals[mesh['BE'][i, 2]] += np.array(be_flux) * be_l[i]
        sum_sl[mesh['BE'][i, 2]] += be_smax * be_l[i]

    return residuals, sum_sl


# TODO: Fix Roe Flux then copy Roe flux code here and swap Roe calls to HLLE calls
def compute_residuals_hlle(config, mesh, state):
    """Computes the residuals and sum of speed*edge_lengths for the state on the given mesh using the HLLE Flux method.

    :param config: Operating and working fluid conditions
    :param mesh: Computational domain in KFID GRI format
    :param state: Nx4 array of state values
    :return:
    """
    residuals = np.zeros((len(mesh['E']), 4))  # Residuals from fluxes
    sum_sl = np.transpose(np.zeros((len(mesh['E']))))

    # Internal edges
    for ie in mesh['IE']:
        # ie = [0: node A, 1: node B, 2: cell i, 3: cell N]
        ie_l, ie_n = mesh_processing.edge_properties_calculator(mesh['V'][ie[0]], mesh['V'][ie[1]])

        ie_flux, ie_smax = hlle_euler_2d(state[ie[2]], state[ie[3]], ie_n, config)

        residuals[ie[2]] += ie_flux * ie_l  # Summing residuals to be taken out of cell i
        residuals[ie[3]] -= ie_flux * ie_l  # Summing residuals to negative taken out of cell N (added to cell N)

        sum_sl[ie[2]] += ie_smax * ie_l
        sum_sl[ie[3]] += ie_smax * ie_l

    # Boundary edges with their respective conditions
    for be in mesh['BE']:
        # be = [0: node A, 1: node B, 2: cell i, 3: boundary index]
        be_l, be_n = mesh_processing.edge_properties_calculator(mesh['V'][be[0]], mesh['V'][be[1]])
        be_smax = 0  # In case there is some additional boundary condition

        if mesh['Bname'][be[3]] == 'Engine':
            # Apply inviscid wall boundary condition
            # Boundary velocity
            u_plus = state[be[2], 1] / state[be[2], 0]
            v_plus = state[be[2], 2] / state[be[2], 0]
            V_plus = np.array([u_plus, v_plus])
            be_vel = V_plus - np.multiply(np.dot(V_plus, be_n), be_n)

            # Boundary pressure
            be_pressure = (config.y - 1) * (state[be[2], 3] - 0.5 * state[be[2], 0] * (be_vel[0] ** 2 + be_vel[1] ** 2))

            # Enforcing no flow through with pressure condition
            be_flux = [0, be_pressure * be_n[0], be_pressure * be_n[1], 0]
            local_flux = F_euler_2d(state[be[2]], config)
            h = (local_flux[3][0] + local_flux[3][1]) / (local_flux[0][1] + local_flux[0][0])
            be_smax = abs(u_plus * be_n[0] + v_plus * be_n[1]) \
                      + math.sqrt((config.y - 1) * (h - np.linalg.norm([u_plus, v_plus]) ** 2 / 2))


        elif mesh['Bname'][be[3]] == 'Exit' or mesh['Bname'][be[3]] == 'Outflow':
            # Apply supersonic outflow boundary conditions
            be_flux, be_smax = hlle_euler_2d(state[be[2]], state[be[2]], be_n, config)

        elif mesh['Bname'][be[3]] == 'Inflow':
            # Apply freestream inflow boundary conditions
            be_flux, be_smax = hlle_euler_2d(state[be[2]], helper.generate_freestream_state(config),
                                             be_n, config)

        residuals[be[2]] += np.array(be_flux) * be_l
        sum_sl[be[2]] += be_smax * be_l

    return residuals, sum_sl