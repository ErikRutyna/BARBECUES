import flux
import numpy as np
import helper
import cell_geometry_formulas as cgf
from numba import njit
from numba.typed import List


# @njit(cache=True)
def euler_2D_v2(E, V, BE, IE, state, M, a, y, f_method, c_method, c_tol, s_tol, s_len, s_e_tol, asc_check):
    """Runs steady state Euler equation solver in 2D based on the given
    setup information such as mesh, fluid information, program configurations, etc.

    :param E: Element-2-Node array
    :param V: Node Coordinate array
    :param BE: Boundary Edge Matrix [nodeA, nodeB, cell, boundary flag]
    :param IE: Internal Edge Matrix [nodeA, nodeB, cell left, cell right]
    :param state: State vector array
    :param M: Freestream Mach number
    :param a: Freestream angle of attack
    :param y: Ratio of specific heats - gamma
    :param f_method: Flux method identifier
    :param c_method: Convergence method identifier
    :param c_tol: Standard/backup convergence method tolerance
    :param s_tol: Smart convergence minimum tolerence
    :param s_len: Smart convergence ASC array length
    :param s_e_tol: Smart convergence running average tolerance
    :param asc_check: Smart convergence asymptotic convergence criteria
    """
    # Convergence and iteration information
    converged = False
    iteration_number = 1
    # All residual coefficient tracking for standard convergence/plotting purposes
    residual_norms = np.zeros((0, 5))
    coefficients = np.zeros((0, 4))
    # Asymptotic Convergence Criteria for Smart Convergence
    ascs = np.empty((0, 4))
    # TODO: Investigate dynamic CFL numbers, HoM (gradient-based), and HoI schemes (RK2, RK4)
    cfl = 1

    be_l, be_n = np.zeros((BE.shape[0])), np.zeros((BE.shape[0], 2))
    ie_l, ie_n = np.zeros((IE.shape[0])), np.zeros((IE.shape[0], 2))

    for i in range(BE.shape[0]):
        be_l[i], be_n[i] = cgf.edge_properties_calculator(V[BE[i, 0]], V[BE[i, 1]])
    for i in range(IE.shape[0]):
        ie_l[i], ie_n[i] = cgf.edge_properties_calculator(V[IE[i, 0]], V[IE[i, 1]])

    while not converged:
        if iteration_number % 10 == 0:
            # Print out a small tracking statement every 10 iterations to watch the program
            print_resi = float(residual_norms[-1, 4])
            print("Iteration: ", iteration_number, "\t L1 Residual Norm:", print_resi)

        # Reset the residual and timestep arrays
        residuals = np.zeros((E.shape[0], 4))  # Residuals from fluxes
        sum_sl = np.transpose(np.zeros((E.shape[0])))  # s*l vector for computing time steps

        # If-elif-else tree for flux method selection
        if f_method == 1:
            residuals, sum_sl = flux.compute_residuals_roe(IE, BE, state, be_l, be_n, ie_l, ie_n, M, a, y)
        elif f_method == 2:
            residuals, sum_sl = flux.compute_residuals_hlle(IE, BE, state, be_l, be_n, ie_l, ie_n, M, a, y)
        # If it cannot find the right flux method, it will default to the Roe Flux
        else:
            residuals, sum_sl = flux.compute_residuals_roe(IE, BE, state, be_l, be_n, ie_l, ie_n, M, a, y)

        cont_norm = np.linalg.norm(residuals[:, 0], ord=1)
        xmom_norm = np.linalg.norm(residuals[:, 1], ord=1)
        ymom_norm = np.linalg.norm(residuals[:, 2], ord=1)
        ener_norm = np.linalg.norm(residuals[:, 3], ord=1)
        glob_norm = np.linalg.norm(residuals, ord=1)
        new_norms = np.array((cont_norm, xmom_norm, ymom_norm, ener_norm, glob_norm))

        # Residual tracking - L1 norms of [continuity, x-moment, y-momentum, energy, all]
        residual_norms = np.vstack((residual_norms, np.reshape(new_norms, (1, 5))))

        # Coefficient tracking - exported for plotting purposes
        stagnation_pressure = helper.calculate_stagnation_pressure(state, helper.calculate_mach(state, y), y)
        atpr = helper.calculate_atpr(V, BE, stagnation_pressure)
        cd, cl, cmx = helper.calculate_forces_moments(V, BE, state, M, a, y)
        coefficients = np.vstack((coefficients, np.reshape(np.array((cd, cl, cmx, atpr)), (1, 4))))

        # Calculate delta_t and timestep forward the local states
        deltat_deltaa = np.divide(2 * cfl, sum_sl)
        state -= np.transpose(np.multiply(deltat_deltaa, np.transpose(residuals)))

        # Apply the right convergence method depending on the configuration
        if c_method == 1:
            # Require some minimum degree of convergence to ensure proper physics
            if residual_norms[-1, 4] < s_tol:
                # Check the ASC quantities and add to back of array that tracks them
                converge_check = []

                # If the array is already at counter length - pop off the first value as we don't want it in the counter
                if ascs.shape[0] >= s_len:
                    # If we've hit length, check to see if the last value is close to the average for the desired ASCs
                    for asc in asc_check:
                        if asc == 0:
                            if abs(ascs[-1, 0] - np.mean(ascs[-s_len+1::, 0])) / np.mean(ascs[-s_len+1::, 0]) < s_e_tol:
                                converge_check.append(True)
                        if asc == 1:
                            if abs(ascs[-1, 1] - np.mean(ascs[-s_len+1::, 1])) / np.mean(ascs[-s_len+1::, 1]) < s_e_tol:
                                converge_check.append(True)
                        if asc == 2:
                            if abs(ascs[-1, 2] - np.mean(ascs[-s_len+1::, 2])) / np.mean(ascs[-s_len+1::, 2]) < s_e_tol:
                                converge_check.append(True)
                        if asc == 3:
                            if abs(ascs[-1, 3] - np.mean(ascs[-s_len+1::, 3])) / np.mean(ascs[-s_len+1::, 3]) < s_e_tol:
                                converge_check.append(True)
                    # If all checks pass, then its converged
                    if np.all(np.array(converge_check)):
                        print_resi = float(residual_norms[-1, 4])
                        print("Iteration: ", iteration_number, "\t L1 Residual Norm:", print_resi)
                        return residual_norms, coefficients
                # Append the newly calculated values for the ASCs
                ascs = np.vstack((ascs, np.reshape(np.array((cd, cl, cmx, atpr)), (1, 4))))
            # If Smart convergence somehow fails, employ standard convergence as a backup method
            elif residual_norms[-1, 4] < c_tol:
                print_resi = float(residual_norms[-1, 4])
                print("Iteration: ", iteration_number, "\t L1 Residual Norm:", print_resi)
                return residual_norms, coefficients
        else:
            if residual_norms[-1, 4] < c_tol:
                print_resi = float(residual_norms[-1, 4])
                print("Iteration: ", iteration_number, "\t L1 Residual Norm:", print_resi)
                return residual_norms, coefficients
        iteration_number += 1
    return residual_norms, coefficients