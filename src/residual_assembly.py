import flux
import numpy as np
import helper
import mesh_processing
import math


# NOTE: Doesn't work - keeping for legacy reasons and proof of work
def euler_2D(sim_config, freestream_config, fluid, mesh, state):
    """Runs steady state Euler equation solver in 2D based on the given
    setup information such as mesh, fluid information, program configurations, etc.

    :param sim_config: Configuration for the simulation to run, controls things like
     adaptation, flux method, shock capturing etc.
    :param freestream_config: Freestream configuration
    :param fluid: Working fluid properties
    :param mesh: 2D mesh of the domain from the readgri format
    :param state: Initialized state for all values
    :return: Modifies the state array in place and when finished running
    """
    # Convergence and iteration information
    converged = False
    iteration_number = 1
    residuals_norm = []
    atpr = []

    cfl = 1

    for i in range(sim_config.adaptation_cycles):
        while not converged:
            residuals = np.zeros((len(mesh['E']), 4))
            s_l = np.zeros((len(mesh['E'])))
            s_l2= np.zeros((len(mesh['E'])))
            # Loop over the boundary edges for residual assembly and apply special BC's depending on boundary type
            for be in mesh['BE']:
                # be = Nx4 array: [node_a, node_b, index_left, boundary condition]
                be_length, be_normal = mesh_processing.edge_properties_calculator(mesh['V'][be[0]], mesh['V'][be[1]])

                if be[3] == 0: # Inviscid Wall on engine
                    # Boundary velocity
                    u_plus = state[be[2], 1] / state[be[2], 0]
                    v_plus = state[be[2], 2] / state[be[2], 0]
                    V_plus = np.array([u_plus, v_plus])

                    be_vel = V_plus - np.multiply(np.dot(V_plus, be_normal), be_normal)

                    # Boundary pressure
                    be_pressure = (fluid.y - 1) * (state[be[2], 3] - 0.5 * state[be[2], 0] * (be_vel[0] ** 2 + be_vel[1] ** 2))

                    # Enforcing no flow through with pressure condition
                    be_flux = [0, be_pressure * be_normal[0], be_pressure * be_normal[1], 0]

                    # I don't understand this formula for propagation speed at all
                    p = (fluid.y - 1) * (state[be[2], 3] - 0.5 * (u_plus ** 2 + v_plus ** 2))
                    h = (state[be[2], 3] + p) / state[be[2], 0]
                    q = math.sqrt((u_plus ** 2 + v_plus ** 2))
                    s = abs(u_plus * be_normal[0] + v_plus * be_normal[1]) \
                            + math.sqrt((fluid.y - 1) * (h - (q ** 2) / 2))
                    residuals[be[2]] += np.array(be_flux) * be_length
                    # print(be , '\t' , residuals[be[2]])

                elif be[3] == 1 or be[3] == 2: # Inlet exit & outflow exit conditions
                    # Outflow conditions have flow into "zero" cell
                    be_flux, s = flux.roe_euler_2d(state[be[2]], state[be[2]], be_normal, fluid)
                    be_flux2, s2 = flux.compFlux(be_normal, state[be[2]], state[be[2]])
                    residuals[be[2]] += np.array(be_flux2) * be_length
                    # print(be , '\t' , residuals[be[2]].round(14))

                else: # Inflow boundary condition
                    # Inflow conditions use a "freestream" cell
                    be_flux, s = flux.roe_euler_2d(state[be[2]], helper.generate_freestream_state(freestream_config, fluid), be_normal, fluid)
                    be_flux2, s2 = flux.compFlux(be_normal, state[be[2]], helper.generate_freestream_state(freestream_config, fluid))

                    residuals[be[2]] += np.array(be_flux2) * be_length
                    # print(be , '\t' , residuals[be[2]].round(14))
                s_l[be[2]] += s * be_length
                s_l2[be[2]] += s2 * be_length
            norm_diff = []
            # Loop over the internal edges for residual assembly
            for ie in mesh['IE']:
                # ie = Nx4 array: [node_a, node_b, index_left, index_right]
                ie_length, ie_normal = mesh_processing.edge_properties_calculator(mesh['V'][ie[0]], mesh['V'][ie[1]])
                _, ie_normal2, _ = mesh_processing.getNormal([ie[0]], [ie[1]], ie[2], mesh['V'], mesh['E'])
                norm_diff.append(np.abs(ie_normal - np.transpose(ie_normal2)))
                ie_flux, s = flux.roe_euler_2d(state[ie[2]], state[ie[3]], ie_normal, fluid)
                ie_flux2, s2 = flux.compFlux(ie_normal, state[ie[2]], state[ie[3]])

                residuals[ie[2]] += ie_flux2 * ie_length
                residuals[ie[3]] -= ie_flux2 * ie_length

                s_l[ie[2]] += s * ie_length
                s_l[ie[3]] += s * ie_length

                s_l2[ie[2]] += s2 * ie_length
                s_l2[ie[3]] += s2 * ie_length

            # Calculate delta_t and timestep forward the local states
            # deltat_deltaa = np.divide(2 * cfl, s_l)
            # state = np.subtract(state, np.transpose(np.multiply(deltat_deltaa, np.transpose(state))))
            # u-np.transpose(2/(delt_delA)*np.transpose(R))
            state = state - np.transpose((2 / s_l * np.transpose(residuals)))

            residuals_norm.append(np.linalg.norm(residuals, ord=1))

            # Check for convergence, either "smart" or standard
            if sim_config.smart_convergence:
                pass
            else:
                if residuals_norm[iteration_number - 1] < 1e-5:
                    converged = True

            # Print out a small tracking statement every 10 iterations
            if iteration_number % 10 == 0:
                print('Iteration: {0}\t L1 Residual Norm: {1}'.format(iteration_number, residuals_norm[iteration_number - 1]))

            iteration_number += 1
    return

def euler_2D_v2(config, mesh, state):
    """Runs steady state Euler equation solver in 2D based on the given
    setup information such as mesh, fluid information, program configurations, etc.

    :param config: Class containing fluid and simulation configuration information
    :param mesh: 2D mesh of the domain from the readgri format
    :param state: Initialized state for all values
    :return: Modifies the state array in place and when finished running
    """
    # Convergence and iteration information
    converged = False
    iteration_number = 1
    residuals_norm = []
    atrp = []

    be_l, be_n = np.zeros((mesh['BE'].shape[0])), np.zeros((mesh['BE'].shape[0], 2))
    ie_l, ie_n = np.zeros((mesh['IE'].shape[0])), np.zeros((mesh['IE'].shape[0], 2))

    for i in range(mesh['BE'].shape[0]):
        be_l[i], be_n[i] = mesh_processing.edge_properties_calculator(mesh['V'][mesh['BE'][i, 0]],
                                                                      mesh['V'][mesh['BE'][i, 1]])
    for i in range(mesh['IE'].shape[0]):
        ie_l[i], ie_n[i] = mesh_processing.edge_properties_calculator(mesh['V'][mesh['IE'][i, 0]],
                                                                      mesh['V'][mesh['IE'][i, 1]])
    # Time stepping information
    cfl = 1

    atrp_temp = []
    r_temp = []

    while not converged:
        if iteration_number % 10 == 0:
            # Print out a small tracking statement every 10 iterations to watch the program
            print('Iteration: {0}\t L1 Residual Norm: {1}'.format(iteration_number,
                                                                  residuals_norm[iteration_number - 2]))
        # Reset the residual and timestep arrays
        residuals = np.zeros((len(mesh['E']), 4))  # Residuals from fluxes
        sum_sl = np.transpose(np.zeros((len(mesh['E']))))  # s*l vector for computing time steps

        if config.flux_method == 'roe':
            residuals, sum_sl = flux.compute_residuals_roe(config, mesh, state, be_l, be_n, ie_l, ie_n)
        elif config.flux_method == 'hlle':
            residuals, sum_sl = flux.compute_residuals_hlle(config, mesh, state)
            pass

        # Calculate delta_t and timestep forward the local states
        deltat_deltaa = np.divide(2 * cfl, sum_sl)
        state -= np.transpose(np.multiply(deltat_deltaa, np.transpose(residuals)))

        residuals_norm.append(np.linalg.norm(residuals, ord=1))

        stagnation_pressure_temp = helper.calculate_stagnation_pressure(state, helper.calculate_mach(state, config), config)
        atrp_temp.append(helper.calculate_atpr(stagnation_pressure_temp, mesh))

        # Check for convergence, either "smart" or standard
        if config.smart_convergence:
            # Require some minimum degree of convergence
            if residuals_norm[iteration_number - 1] < 1e-2:
                # Calculate ATRP and add to back of array, if above size - pop first value off
                if len(atrp) == config.sma_counter:
                    atrp.pop(0)
                stagnation_pressure = helper.calculate_stagnation_pressure(state, helper.calculate_mach(state, config), config)
                atrp.append(helper.calculate_atpr(stagnation_pressure, mesh))

                # Check if recent result difference is less than simple moving average - if so we can call it converged
                if len(atrp) == config.sma_counter and \
                        (abs(atrp[config.sma_counter - 1] - np.average(atrp)) / np.average(atrp)) < config.error_percent:
                    return
        else:
            if residuals_norm[iteration_number - 1] < 1e-5:
                print('Iteration: {0}\t L1 Residual Norm: {1}'.format(iteration_number,
                                                                      residuals_norm[iteration_number - 2]))
                return
        iteration_number += 1
    return