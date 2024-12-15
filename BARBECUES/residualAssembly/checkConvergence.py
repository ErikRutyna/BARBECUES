from numba import njit
import numpy as np

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