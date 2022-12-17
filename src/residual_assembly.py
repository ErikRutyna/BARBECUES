import flux
import numpy as np
import helper
import cell_geometry_formulas as cgf
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
    atpr = []
    # TODO: Investigate dynamic CFL numbers
    cfl = 1

    be_l, be_n = np.zeros((mesh['BE'].shape[0])), np.zeros((mesh['BE'].shape[0], 2))
    ie_l, ie_n = np.zeros((mesh['IE'].shape[0])), np.zeros((mesh['IE'].shape[0], 2))

    for i in range(mesh['BE'].shape[0]):
        be_l[i], be_n[i] = cgf.edge_properties_calculator(mesh['V'][mesh['BE'][i, 0]],
                                                                      mesh['V'][mesh['BE'][i, 1]])
    for i in range(mesh['IE'].shape[0]):
        ie_l[i], ie_n[i] = cgf.edge_properties_calculator(mesh['V'][mesh['IE'][i, 0]],
                                                                      mesh['V'][mesh['IE'][i, 1]])

    while not converged:
        if iteration_number % 10 == 0:
            # Print out a small tracking statement every 10 iterations to watch the program
            print('Iteration: {0}\t L1 Residual Norm: {1}'.format(iteration_number,
                                                                  residuals_norm[iteration_number - 2]))
        # Reset the residual and timestep arrays
        residuals = np.zeros((len(mesh['E']), 4))  # Residuals from fluxes
        sum_sl = np.transpose(np.zeros((len(mesh['E']))))  # s*l vector for computing time steps

        # If-elif-else tree for flux method selection
        if config.flux_method == 'roe':
            residuals, sum_sl = flux.compute_residuals_roe(config, mesh, state, be_l, be_n, ie_l, ie_n)
        elif config.flux_method == 'hlle':
            residuals, sum_sl = flux.compute_residuals_hlle(config, mesh, state)
        # If it cannot find the right flux method, it will default to the Roe Flux
        else:
            residuals, sum_sl = flux.compute_residuals_roe(config, mesh, state, be_l, be_n, ie_l, ie_n)
        # Residual tracking
        residuals_norm.append(np.linalg.norm(residuals, ord=1))

        # Calculate delta_t and timestep forward the local states
        deltat_deltaa = np.divide(2 * cfl, sum_sl)
        state -= np.transpose(np.multiply(deltat_deltaa, np.transpose(residuals)))

        # Apply the right convergence method depending on the configuration
        if config.smart_convergence:
            # Require some minimum degree of convergence to ensure proper physics
            if residuals_norm[-1] < config.smart_convergence_minimum:
                # Calculate ATPR and add to back of array
                stagnation_pressure = helper.calculate_stagnation_pressure(state, helper.calculate_mach(state, config),
                                                                           config)
                # If the array is already at counter length - pop off the first value as we don't want it in the counter
                if len(atpr) == config.sma_counter:
                    # If we've hit length, check to see if the last value is close to the average
                    if (abs(atpr[-1] - np.average(atpr)) / np.average(atpr)) < config.error_percent:
                        return
                    else:
                        atpr.pop(0)
                # Append the newly calculated value for ATPR
                atpr.append(helper.calculate_atpr(stagnation_pressure, mesh))
        else:
            if residuals_norm[iteration_number - 1] < 1e-5:
                print('Iteration: {0}\t L1 Residual Norm: {1}'.format(iteration_number,
                                                                      residuals_norm[iteration_number - 2]))
                return


        iteration_number += 1
    return