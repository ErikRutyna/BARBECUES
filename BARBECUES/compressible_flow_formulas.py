from numba import njit
import math
import numpy as np


# A collection of compressible flow formulas coded while taking Aerospace 520 @ University of Michigan that are used
# once/twice in the code base.
def f(turnAngle, rampAngle, machNumber, gamma):
    """theta-beta-Mach formula except the returned result is the LHS - RHS.

    :param turnAngle: turning angle in radians
    :param rampAngle: flow turning angle in radians
    :param machNumber: upstream of shock Mach number
    :param gamma: ratio of specific heats
    :return: fout = LHS - RHS of the theta-beta-Mach formula
    """
    numerator = machNumber ** 2 * np.sin(turnAngle) ** 2 - 1
    denominator = machNumber ** 2 * (gamma + np.cos(2 * turnAngle)) + 2

    fout = 2 * (np.tan(turnAngle)) ** -1 * numerator / denominator - np.tan(rampAngle)

    return fout


def fp(turnAngle, machNumber, gamma):
    """Derivative of the LHS - RHS of the theta-beta-Mach formula.

    :param turnAngle: turning angle in radians
    :param machNumber: upstream of shock Mach number
    :param gamma: ratio of specific heats
    :return: fout = LHS - RHS of the derivative theta-beta-Mach formula
    """

    h = 2 * np.tan(turnAngle) ** -1
    hp = -2 * math.sin(turnAngle) ** -2

    u = machNumber ** 2 * math.sin(turnAngle) ** 2 - 1
    up = machNumber ** 2 * math.sin(2 * turnAngle)

    v = machNumber ** 2 * (gamma + np.cos(2 * turnAngle)) + 2
    vp = -2 * machNumber ** 2 * math.sin(2 * turnAngle)

    fout = hp * u / v + h * (v * up - vp * u) / v ** 2

    return fout


def solvebeta(rampAngle, machNumber, gamma):
    """Solves for the flow turning angle of the theta-beta-Mach formula.

    :param rampAngle: turning angle
    :param machNumber: upstream of shock Mach number
    :param gamma: ratio of specific heats
    :return: turning angle (beta)
    """
    # Edge case - no turning angle -> normal shock
    if rampAngle == 0:
        beta = np.pi / 2
        return beta

    # Initial guess for beta
    beta = rampAngle * np.pi / 180

    # 100 iterations of Newton-Raphson for root finder to get the value of beta that best fits the t-b-M relationship
    for i in range(100):
        beta = beta - f(beta, rampAngle, machNumber, gamma) / fp(beta, machNumber, gamma)

    return beta


def pratio(machNumber, gamma):
    """Normal shock pressure ratio (p2/p1 - downstream/upstream)

    :param machNumber: upstream Mach number
    :param gamma: ratio of specific heats
    :return: pressure ratio of downstream/upstream pressures (p2/p1)
    """
    pr = 1 + 2 * gamma / (gamma + 1) * (machNumber ** 2.0 - 1)
    return pr


def rratio(machNumber, gamma):
    """Normal shock density ratio (rho2/rho1 - downstream/upstream)

    :param machNumber: upstream Mach number
    :param gamma: ratio of specific heats
    :return: Density ratio of downstream/upstream densities (rho2/rho1)
    """
    rr = (gamma + 1) * machNumber ** 2.0 / ((gamma - 1) * machNumber ** 2.0 + 2.0)
    return rr


def tratio(machNumber, gamma):
    """Normal shock temperature ratio (t2/t1 - downstream/upstream)

    :param machNumber: upstream Mach number
    :param gamma: ratio of specific heats
    :return: Temperature ratio of downstream/upstream temperatures (t2/t1)
    """
    pr = pratio(machNumber, gamma)
    rr = rratio(machNumber, gamma)
    tr = pr/rr
    return tr


def Mpost(machNumber, gamma):
    """Mach number post normal shock

    :param machNumber: upstream Mach number
    :param gamma: ratio of specific heats
    :return: Mach number post normal shock
    """
    num = machNumber ** 2.0 + (2.0 / (gamma - 1))
    den = 2.0 * gamma / (gamma-1) * machNumber ** 2.0 - 1
    return (num/den)**0.5


def p0(staticPressure, machNumber, gamma):
    """Stagnation pressure calculator, use a value of p=1 to return stagnation pressure ratio (p0/p - stagnation/static)

    :param staticPressure: local static pressure
    :param machNumber: local static Mach number
    :param gamma: ratio of specific heats
    :return: Stagnation pressure p0
    """
    ratio = (1 + (gamma - 1) / 2.0 * machNumber ** 2.0) ** (gamma / (gamma - 1))
    return ratio*staticPressure


def T0(staticTemperature, machNumber, gamma):
    """Stagnation temperature calculator, use a value of T=1 to return stagnation temperature ratio
    (T0/T - stagnation/static)

    :param staticTemperature: local static temperature
    :param machNumber: local static Mach number
    :param gamma: ratio of specific heats
    :return: Stagnation temperature T0
    """
    ratio = (1 + (gamma-1) / 2.0 * machNumber ** 2.0)
    return ratio*staticTemperature


def r0(staticDensity, machNumber, gamma):
    """Stagnation density calculator, use a value of r=1 to return stagnation density ratio (r0/r - stagnation/static)

    :param staticDensity: local static density
    :param machNumber: local static Mach number
    :param gamma: ratio of specific heats
    :return: Stagnation density r0
    """
    ratio = (1 + (gamma-1) / 2.0 * machNumber ** 2.0) ** (1.0 / (gamma - 1))
    return ratio*staticDensity


def obliqueshock(rampAngle, machNumber, upstreamStaticPressure, upstreamStaticTemperature, upstreamStaticDensity, gamma):
    """Given an initial fluid dynamic state (M1, p1, T1, r1) and a ramp angle (theta), return a dictionary consisting of
    the post-oblique shock state (M2, p2, T2, r2, p02, T02), as well as normal Mach numbers (Mn1 and Mn2).

    :param rampAngle: ramp angle in radians
    :param machNumber: upstream Mach number
    :param upstreamStaticPressure: upstream static pressure
    :param upstreamStaticTemperature: upstream static temperature
    :param upstreamStaticDensity: upstream static density
    :param gamma: ratio of specific heats
    :return: result - A dictionary consisting of the listed state values that can be indexed with result["state"]
    """
    result = {}

    # Solve for turning angle for given upstream state
    beta = solvebeta(rampAngle, machNumber, gamma)
    result['beta'] = beta

    # Normal Mach #
    Mn1 = machNumber * math.sin(beta)

    # Post shock Mach
    Mn2 = Mpost(Mn1, gamma)
    M2 = Mn2 / math.sin(beta - rampAngle)

    # Post shock state
    p2 = pratio(Mn1, gamma) * upstreamStaticPressure
    T2 = tratio(Mn1, gamma) * upstreamStaticTemperature
    r2 = rratio(Mn1, gamma) * upstreamStaticDensity

    # Stagnation State
    p01 = p0(upstreamStaticPressure, machNumber, gamma)
    p02 = p0(p2, M2, gamma)

    T01 = T0(upstreamStaticTemperature, machNumber, gamma)
    T02 = T0(T2, M2, gamma)

    result['p2'] = p2
    result['T2'] = T2
    result['r2'] = r2

    result['Mn1'] = Mn1
    result['Mn2'] = Mn2

    result['M1'] = machNumber
    result['M2'] = M2

    result['p01'] = p01
    result['p02'] = p02

    result['T01'] = T01
    result['T02'] = T02

    return result


def findtheta(upstreamMachNumber, downstreamMachNumber, gamma):
    """Returns the ramp angle, theta, needed to form an oblique shock to go from M1 to M2 assuming that the t-b-M
    relationship is valid.

    :param upstreamMachNumber: upstream Mach
    :param downstreamMachNumber: downstream Mach
    :param gamma: ratio of specific heats
    :return: theta - ramp angle in radians that would result in a valid oblique shock to go from M1 to M2
    """
    # Note that the results do not depend on pressure or temperature, but you need to use the pressure/temperature ratio
    # functions defined before. For this, you need to send some dummy values.

    # Dummy values for ratios
    P1 = 101325
    T1 = 298.15
    r1 = 1.225

    # Initial value for theta
    theta = 25 * np.pi / 180

    # Tiny wiggle for numerical derivative
    epi = 0.0001

    # 100 Newton-Raphson iterations to solve for turn angle, theta; uses numerical central difference for derivative
    for i in range(100):
        num = obliqueshock(theta, upstreamMachNumber, P1, T1, r1, gamma)
        num = num['M2'] - downstreamMachNumber

        Upper = obliqueshock(theta + epi, upstreamMachNumber, P1, T1, r1, gamma)
        Upper = Upper['M2']

        Lower = obliqueshock(theta - epi, upstreamMachNumber, P1, T1, r1, gamma)
        Lower = Lower['M2']

        denom = (Upper - Lower) / (2 * epi)

        theta = theta - num / denom

    return theta


def pmfunction(machNumber, gamma):
    """Solves the Prandtl-Meyer formula for nu.

    :param machNumber: local Mach number
    :param gamma: ratio of specific heats
    :return: nu from the Prandtl-Meyer formula
    """

    term1 = math.sqrt((gamma + 1) / (gamma - 1)) * math.atan(math.sqrt((gamma - 1) / (gamma + 1) * (machNumber ** 2 - 1)))
    term2 = math.atan(math.sqrt(machNumber ** 2 - 1))

    nu = term1 - term2
    return nu


def pmfunctionp(machNumber, gamma):
    """Derivative of the Prandtl-Meyer formula used in solving for Mach number after an expansion fan.

    :param machNumber: local Mach number
    :param gamma: ratio of specific heats
    :return: nu prime

    """
    C1 = ((gamma - 1) / (gamma + 1)) ** 0.5 * (machNumber ** 2 - 1) ** 0.5
    C2 = (machNumber ** 2 - 1) ** 0.5

    dC1 = ((gamma - 1) / (gamma + 1)) ** 0.5 * machNumber * (machNumber ** 2 - 1) ** -0.5
    dC2 = machNumber * (machNumber ** 2 - 1) ** -0.5

    constant = ((gamma + 1) / (gamma - 1)) ** 0.5
    term1 = constant * dC1 / (1 + C1 ** 2)
    term2 = dC2 / (1 + C2 ** 2)

    nup = constant * term1 - term2
    return nup


def PMsolveM(rampAngle, gamma):
    """Solves for the Mach number after an expansion with turn angle of theta in radians

    :param rampAngle: turn angle in radians
    :param gamma: ratio of specific heats
    :return: Mnew - Mach number after the turn angle
    """
    # Mnew >= 1 in order for PM-Fans to exist, so start here
    Mnew = 1.01

    # Newton-Raphson as the solver for the post expansion Mach number
    for i in range(100):
        Mnew = Mnew - (pmfunction(Mnew, gamma) - rampAngle) / pmfunctionp(Mnew, gamma)

    return Mnew


def PMexpansion(rampAngle, upstreamMachNumber, upstreamStaticTemperature, upstreamStaticPressure, gamma):
    """Solves for the state after an expansion with turn angle of theta in radians

    :param rampAngle: turn angle in radians
    :param upstreamMachNumber: upstream Mach number
    :param upstreamStaticTemperature: upstream static temperature
    :param upstreamStaticPressure: upstream static pressure
    :param gamma: ratio of specific heats
    :return: result - Dictionary that can be indexed to get M2, T2, and p2
    """
    # Solve for the Mach number
    M2 = PMsolveM(rampAngle + pmfunction(upstreamMachNumber, gamma), gamma)

    # Temperature
    T1_T2 = (1 + (gamma - 1) / 2 * M2 ** 2) / (1 + (gamma - 1) / 2 * upstreamMachNumber ** 2)
    T2 = T1_T2 ** -1 * upstreamStaticTemperature

    # Pressure
    P1_P2 = T1_T2 ** (gamma / (gamma - 1))
    p2 = P1_P2 ** -1 * upstreamStaticPressure

    result = {'M2': M2, 'T2': T2, 'p2': p2}
    return result


@njit(cache=True)
def sutherland_viscosity(temperature, referenceViscosity, referenceTemperature, sutherlandConstant=111):
    """Calculates the viscosity using Sutherland's model for a given temperature.

    :param temperature: Temperature to evaluate viscosity at
    :param referenceViscosity: Reference viscosity (1.716e-5 Pa*s for CPG air)
    :param referenceTemperature: Reference temperature (273 K for CPG air)
    :param sutherlandConstant: Sutherland reference constant (111 K for CPG air)
    :return: mu - kinematic viscosity at temperature t
    """
    mu = np.multiply(np.multiply(referenceViscosity, np.power(temperature / referenceTemperature, 1.5)), np.divide((referenceTemperature + sutherlandConstant), (temperature + sutherlandConstant)))

    return mu