import cell_geometry_formulas as cgf
from numba import njit
import numpy as np
import helper
import time_integration

@njit(cache=True)
def euler_2D_v2(E, V, BE, IE, state, M, a, y, fluxMethod, timeMethod, convergenceMethod, convergenceTolerance, smartMinConvTol,
                smartConvAscLen, smartConvASCAvgTol, smartASCsToCheck):
    """Runs steady state Euler equation solver in 2D based on the given
    setup information such as mesh, fluid information, program configurations, etc.

    :param E: [:, 3] Numpy array of the Element-2-Node hashing
    :param V: [:, 2] Numpy array of x-y coordinates of node locations
    :param BE: [:, 4] Numpy array boundary Edge Matrix [nodeA, nodeB, cell, boundary flag]
    :param IE: [:, 4] Numpy array internal Edge Matrix [nodeA, nodeB, cell left, cell right]
    :param state: [:, 4] Numpy array of state vectors, each row is 1 cell's state [rho, rho*u, rho*v, rho*E]
    :param M: Freestream Mach number
    :param a: Freestream angle of attack
    :param y: Ratio of specific heats - gamma
    :param fluxMethod: Flux method identifier
    :param timeMethod: Time integration method
    :param convergenceMethod: Convergence method identifier
    :param convergenceTolerance: Standard/backup convergence method tolerance
    :param smartMinConvTol: Smart convergence minimum tolerance
    :param smartConvAscLen: Smart convergence ASC array length
    :param smartConvASCAvgTol: Smart convergence running average tolerance
    :param smartASCsToCheck: Smart convergence asymptotic convergence criteria
    :returns: Nothing, modifies the state vector array in place.
    """
    # Convergence and iteration information
    converged = False
    iterationNumber = 0

    # All residual coefficient tracking for standard convergence/plotting purposes
    residualNorms = np.zeros((0, 5))
    coefficients = np.zeros((0, 3))

    # Asymptotic Convergence Criteria for Smart Convergence
    ascs = np.empty((0, 3))

    # TODO: Investigate dynamic CFL numbers, HoM (gradient-based), and HoI schemes (RK2, RK4, RK45)
    # This will likely involve two convergence possibilities, # of iterations or until fully/smart converged
    cfl = 1

    # TODO: Apply vectorized form in locations where un-vectorized form exists
    beLength, beNorm = cgf.edgePropertiesCalculator(BE[:, 0:2], V)
    ieLength, ieNorm = cgf.edgePropertiesCalculator(IE[:, 0:2], V)
    areas = cgf.areaCalculator(E, V)

    # Reset the residual and timestep arrays
    residuals = np.zeros((E.shape[0], 4))  # Residuals from fluxes
    sumSL = np.transpose(np.zeros((E.shape[0])))  # s*l vector for computing time steps

    while not converged:

        # Iteration and residual norm information print-outs
        iterationNumber += 1

        if iterationNumber % 25 == 0:
            # Print out a small tracking statement every so many iterations to watch the program in command line
            printResidual = float(residualNorms[-1, 4])
            print("Iteration: ", iterationNumber, "\t L1 Residual Norm:", printResidual)

        # If-elif-else tree for flux method selection
        if timeMethod == 1:
            residuals = \
                time_integration.updateStateRK1(IE, BE, state, beLength, beNorm, ieLength, ieNorm, areas, cfl, M, a, y)
        elif timeMethod == 2:
            residuals = \
                time_integration.updateStateRK2(IE, BE, state, beLength, beNorm, ieLength, ieNorm, areas, cfl, M, a, y)
        else:
            residuals = \
                time_integration.updateStateRK4(IE, BE, state, beLength, beNorm, ieLength, ieNorm, areas, cfl, M, a, y)

        cont_norm = (np.abs(residuals[:, 0])).sum()
        xmom_norm = (np.abs(residuals[:, 1])).sum()
        ymom_norm = (np.abs(residuals[:, 2])).sum()
        ener_norm = (np.abs(residuals[:, 3])).sum()
        glob_norm = (np.abs(residuals)).sum()
        new_norms = np.array((cont_norm, xmom_norm, ymom_norm, ener_norm, glob_norm))

        # Residual tracking - L1 norms of [continuity, x-moment, y-momentum, energy, all]
        residualNorms = np.vstack((residualNorms, np.reshape(new_norms, (1, 5))))

        # Coefficient tracking - exported for plotting purposes
        local_Mach = helper.calculate_mach(state, y)
        stagnation_pressure = helper.calculate_stagnation_pressure(state, local_Mach, y)
        atpr = helper.calculate_atpr(V, BE, stagnation_pressure)
        cd, cl = helper.calculate_forces(V, BE, state, M, a, y)
        coefficients = np.vstack((coefficients, np.reshape(np.array((cd, cl, atpr)), (1, 3))))

        # Check for convergence, if not converged update the ASCs and iterate
        converged, ascs = checkConvergence(convergenceMethod, residualNorms, cl, cd, atpr, ascs, smartASCsToCheck,
                                           smartConvAscLen, smartConvASCAvgTol, smartMinConvTol, convergenceTolerance)

    printResidual = float(residualNorms[-1, 4])
    print("Iteration: ", iterationNumber, "\t L1 Residual Norm:", printResidual)
    return residualNorms, coefficients


@njit(cache=True)
def checkConvergence(convergenceMethod, residualNorms, cl, cd, atpr, ascs, ascsToCheck, lengthSmartConvAvg,
                     smartConvAvgTol, smartConvTol, residualTol):
    """
    Checks if the simulation is converged based on the specified convergence method and parameters set by the method.

    Parameters
    ----------
    :param convergenceMethod: Value to determine convergence method, 1 implies smart convergence,
                              anything else implies standard convergence
    :param residualNorms: [:, 5] Numpy array of residual norms [continuity, x-momentum, y-momentum, energy, total]
    :param cl: Lift coefficient from the most recent iteration
    :param cd: Drag coefficient from the most recent iteration
    :param atpr: Average total stagnation pressure recovered at "Exit" boundary condition from the most recent iteration
    :param ascs: [:, 3] Array of asymptotic convergence criteria (cd, cl, atpr)
    :param ascsToCheck: 1-D Numpy array of the ascs to check
    :param lengthSmartConvAvg: Number of values to use in the smart convergence running average
    :param smartConvAvgTol: Tolerance use in the smart convergence method running average
    :param smartConvTol: Minimum convergence of the residuals before smart convergence activates
    :param residualTol: Minimum convergence for the default convergence method, as well as a fallback value for smart
                        convergence

    Returns
    -------
    :returns: True or false depending on if the simulation is converged as well as the asymptotic convergence criteria
    array with the set of parameters appended to the end if the convergence method returns false
    """

    # Apply the right convergence method depending on the configuration
    if convergenceMethod == 1:
        # Require some minimum degree of convergence to ensure proper physics
        if residualNorms[-1, 4] < smartConvTol:
            # Check the ASC quantities and add to back of array that tracks them
            converge_check = []

            # If the array is already at counter length - pop off the first value as we don't want it in the counter
            if ascs.shape[0] >= lengthSmartConvAvg:
                # If we've hit length, check to see if the last value is close to the average for the desired ASCs
                for asc in ascsToCheck:
                    if asc == 0:
                        if abs(ascs[-1, 0] - np.mean(ascs[-lengthSmartConvAvg + 1::, 0])) / \
                                np.mean(ascs[-lengthSmartConvAvg + 1::, 0]) < smartConvAvgTol:
                            converge_check.append(True)
                    if asc == 1:
                        if abs(ascs[-1, 1] - np.mean(ascs[-lengthSmartConvAvg + 1::, 1])) / \
                                np.mean(ascs[-lengthSmartConvAvg + 1::, 1]) < smartConvAvgTol:
                            converge_check.append(True)
                    if asc == 2:
                        if abs(ascs[-1, 2] - np.mean(ascs[-lengthSmartConvAvg + 1::, 2])) / \
                                np.mean(ascs[-lengthSmartConvAvg + 1::, 2]) < smartConvAvgTol:
                            converge_check.append(True)
                # If all checks pass, then the simulation has converged
                if np.all(np.array(converge_check)) or residualNorms[-1, 4] < residualTol:
                    return True, ascs

            # Append the newly calculated values for the ASCs and exit out without being converged
            ascs = np.vstack((ascs, np.reshape(np.array((cd, cl, atpr)), (1, 3))))
            return False, ascs

        # If Smart convergence is unable to use ASCs, then use standard convergence as a backup method
        else:
            # Append the newly calculated values for the ASCs and exit out without being converged
            ascs = np.vstack((ascs, np.reshape(np.array((cd, cl, atpr)), (1, 3))))
            return False, ascs
    else:
        # Standard convergence - if residuals are below some global minimum then the simulation is physically converged
        if residualNorms[-1, 4] < residualTol:
            return True, ascs
        else:
            return False, ascs