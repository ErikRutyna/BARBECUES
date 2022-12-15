import flux
import numpy as np
import helper
import mesh_processing
import math


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

    # TODO: Vectorize this part
    be_l, be_n = np.zeros((mesh['BE'].shape[0])), np.zeros((mesh['BE'].shape[0], 2))
    ie_l, ie_n = np.zeros((mesh['IE'].shape[0])), np.zeros((mesh['IE'].shape[0], 2))

    for i in range(mesh['BE'].shape[0]):
        be_l[i], be_n[i] = mesh_processing.edge_properties_calculator(mesh['V'][mesh['BE'][i, 0]],
                                                                      mesh['V'][mesh['BE'][i, 1]])
    for i in range(mesh['IE'].shape[0]):
        ie_l[i], ie_n[i] = mesh_processing.edge_properties_calculator(mesh['V'][mesh['IE'][i, 0]],
                                                                      mesh['V'][mesh['IE'][i, 1]])
    # TODO: Add variable CFL that increases as residuals decrease
    # Time stepping information
    cfl = 1

    atrp_temp = []

    while not converged:
        if iteration_number % 10 == 0:
            # Print out a small tracking statement every 10 iterations to watch the program
            print('Iteration: {0}\t L1 Residual Norm: {1}'.format(iteration_number,
                                                                  residuals_norm[iteration_number - 2]))
        # Reset the residual and timestep arrays
        residuals = np.zeros((len(mesh['E']), 4))  # Residuals from fluxes
        sum_sl = np.transpose(np.zeros((len(mesh['E']))))  # s*l vector for computing time steps

        # TODO: Reconfigure this to be in case format or something
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

        # TODO: Reformat this convergence section as its ugly as hell and confusing
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