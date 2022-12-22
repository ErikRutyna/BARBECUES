import flux
import numpy as np
import helper
import cell_geometry_formulas as cgf
import preprocess as pp


def euler_2D_v2(mesh, state):
    """Runs steady state Euler equation solver in 2D based on the given
    setup information such as mesh, fluid information, program configurations, etc.

    :param mesh: 2D mesh of the domain from the readgri format
    :param state: Initialized state for all values
    :return: Modifies the state array in place and when finished running
    """
    # Convergence and iteration information
    converged = False
    iteration_number = 1
    # All residual coefficient tracking for standard convergence/plotting purposes
    residual_norms = np.empty((0, 5))
    coefficients = np.empty((0, 4))
    # Asymptotic Convergence Criteria for Smart Convergence
    ascs = np.empty((0, 4))
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
            print('Iteration: {0}\t L1 Residual Norm: {1}'.format(iteration_number, residual_norms[-1, 4]))

        # Reset the residual and timestep arrays
        residuals = np.zeros((len(mesh['E']), 4))  # Residuals from fluxes
        sum_sl = np.transpose(np.zeros((len(mesh['E']))))  # s*l vector for computing time steps

        # If-elif-else tree for flux method selection
        if pp.sim_con['flux_method'] == 'roe':
            residuals, sum_sl = flux.compute_residuals_roe(mesh, state, be_l, be_n, ie_l, ie_n)
        elif pp.sim_con['flux_method'] == 'hlle':
            residuals, sum_sl = flux.compute_residuals_hlle(mesh, state, be_l, be_n, ie_l, ie_n)
        # If it cannot find the right flux method, it will default to the Roe Flux
        else:
            residuals, sum_sl = flux.compute_residuals_roe(mesh, state, be_l, be_n, ie_l, ie_n)

        # Residual tracking - L1 norms of [continuity, x-moment, y-momentum, energy, all]
        residual_norms = np.vstack((residual_norms,
                                    (np.linalg.norm(residuals[:, 0], ord=1), np.linalg.norm(residuals[:, 1], ord=1),
                                     np.linalg.norm(residuals[:, 2], ord=1), np.linalg.norm(residuals[:, 3], ord=1),
                                     np.linalg.norm(residuals, ord=1))))
        # Coefficient tracking - exported for plotting purposes
        stagnation_pressure = helper.calculate_stagnation_pressure(state, helper.calculate_mach(state))
        atpr = helper.calculate_atpr(stagnation_pressure, mesh)
        cd, cl, cmx = helper.calculate_forces_moments(mesh, state)
        coefficients = np.vstack((coefficients, (cd, cl, cmx, atpr)))

        # Calculate delta_t and timestep forward the local states
        deltat_deltaa = np.divide(2 * cfl, sum_sl)
        state -= np.transpose(np.multiply(deltat_deltaa, np.transpose(residuals)))

        # Apply the right convergence method depending on the configuration
        if pp.conv_con['convergence_method'] == 'smart':
            # Require some minimum degree of convergence to ensure proper physics
            if residual_norms[-1, 4] < pp.conv_con['smart_convergence_minimum']:
                # Check the ASC quantities and add to back of array that tracks them
                converge_check = []

                # If the array is already at counter length - pop off the first value as we don't want it in the counter
                if ascs.shape[0] == pp.conv_con['smart_convergence_length']:
                    # If we've hit length, check to see if the last value is close to the average for the desired ASCs
                    for asc in pp.conv_con['smart_convergence_ascs']:
                        if asc == 'cd':
                            if abs(ascs[-1, 0] - np.mean(ascs[:, 0])) / np.mean(ascs[:, 0]) < pp.conv_con['smart_convergence_error_tol']:
                                converge_check.append(True)
                        if asc == 'cl':
                            if abs(ascs[-1, 1] - np.mean(ascs[:, 1])) / np.mean(ascs[:, 1]) < pp.conv_con['smart_convergence_error_tol']:
                                converge_check.append(True)
                        if asc == 'cmx':
                            if abs(ascs[-1, 2] - np.mean(ascs[:, 2])) / np.mean(ascs[:, 2]) < pp.conv_con['smart_convergence_error_tol']:
                                converge_check.append(True)
                        if asc == 'atpr':
                            if abs(ascs[-1, 3] - np.mean(ascs[:, 3])) / np.mean(ascs[:, 3]) < pp.conv_con['smart_convergence_error_tol']:
                                converge_check.append(True)
                    # If all checks pass, then its converged
                    if np.all(np.array(converge_check)):
                        print('Iteration: {0}\t L1 Residual Norm: {1}'.format(iteration_number, residual_norms[-1, 4]))
                        return residual_norms, coefficients
                    else:
                        # Remove the first entry
                        np.delete(ascs, 0, axis=1)
                # Append the newly calculated values for the ASCs
                ascs = np.vstack((ascs, (cd, cl, cmx, atpr)))
        else:
            if residual_norms[-1, 4] < pp.conv_con['convergence minimum']:
                print('Iteration: {0}\t L1 Residual Norm: {1}'.format(iteration_number, residual_norms[-1, 4]))
                return residual_norms, coefficients
        iteration_number += 1
    return residual_norms, coefficients