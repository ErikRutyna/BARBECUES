from numba import njit
import math
import numpy as np



def f(beta, theta, M1, gamma):
    """theta-beta-Mach formula except the returned result is the LHS - RHS.

    :param beta: turning angle in radians
    :param theta: flow turning angle in radians
    :param M1: upstream of shock Mach number
    :param gamma: ratio of specific heats
    :return: fout = LHS - RHS of the theta-beta-Mach formula
    """
    numerator = M1 ** 2 * np.sin(beta) ** 2 - 1
    denominator = M1 ** 2 * (gamma + np.cos(2 * beta)) + 2

    fout = 2 * (np.tan(beta)) ** -1 * numerator / denominator - np.tan(theta)

    return fout


def fp(beta, theta, M1, gamma):
    """Derivative of the LHS - RHS of the theta-beta-Mach formula.

    :param beta: turning angle in radians
    :param theta: flow turn angle in radians
    :param M1: upstream of shock Mach number
    :param gamma: ratio of specific heats
    :return: fout = LHS - RHS of the derivative theta-beta-Mach formula
    """

    h = 2 * np.tan(beta) ** -1
    hp = -2 * math.sin(beta) ** -2

    u = M1 ** 2 * math.sin(beta) ** 2 - 1
    up = M1 ** 2 * math.sin(2 * beta)

    v = M1 ** 2 * (gamma + np.cos(2 * beta)) + 2
    vp = -2 * M1 ** 2 * math.sin(2 * beta)

    fout = hp * u / v + h * (v * up - vp * u) / v ** 2

    return fout


def solvebeta(theta, M1, gamma):
    """Solves for the flow turning angle of the theta-beta-Mach formula.

    :param theta: turning angle
    :param M1: upstream of shock Mach number
    :param gamma: ratio of specific heats
    :return: turning angle (beta)
    """
    # Edge case - no turning angle -> normal shock
    if theta == 0:
        beta = np.pi / 2
        return beta

    # Initial guess for beta
    beta = theta * np.pi / 180

    # 100 iterations of Newton-Raphson for root finder to get the value of beta that best fits the t-b-M relationship
    for i in range(100):
        beta = beta - f(beta, theta, M1, gamma) / fp(beta, theta, M1, gamma)

    return beta


def pratio(M, gamma):
    """Normal shock pressure ratio (p2/p1 - downstream/upstream)

    :param M: upstream Mach number
    :param gamma: ratio of specific heats
    :return: pressure ratio of downstream/upstream pressures (p2/p1)
    """
    pr = 1 + 2 * gamma / (gamma + 1) * ( M**2.0 - 1)
    return pr


def rratio(M, gamma):
    """Normal shock density ratio (rho2/rho1 - downstream/upstream)

    :param M: upstream Mach number
    :param gamma: ratio of specific heats
    :return: Density ratio of downstream/upstream densities (rho2/rho1)
    """
    rr = (gamma + 1) * M**2.0 / ((gamma - 1) * M**2.0 + 2.0)
    return rr


def tratio(M, gamma):
    """Normal shock temperature ratio (t2/t1 - downstream/upstream)

    :param M: upstream Mach number
    :param gamma: ratio of specific heats
    :return: Temperature ratio of downstream/upstream temperatures (t2/t1)
    """
    pr = pratio(M,gamma)
    rr = rratio(M,gamma)
    tr = pr/rr
    return tr


def Mpost(M, gamma):
    """Mach number post normal shock

    :param M: upstream Mach number
    :param gamma: ratio of specific heats
    :return: Mach number post normal shock
    """
    num = M**2.0 + (2.0/(gamma-1))
    den = 2.0*gamma/(gamma-1)*M**2.0 -1
    return (num/den)**0.5


def p0(p, M, gamma):
    """Stagnation pressure calculator, use a value of p=1 to return stagnation pressure ratio (p0/p - stagnation/static)

    :param p: local static pressure
    :param M: local static Mach number
    :param gamma: ratio of specific heats
    :return: Stagnation pressure p0
    """
    ratio = (1 + (gamma - 1) / 2.0 * M ** 2.0) ** (gamma / (gamma - 1))
    return ratio*p


def T0(T, M, gamma):
    """Stagnation temperature calculator, use a value of T=1 to return stagnation temperature ratio
    (T0/T - stagnation/static)

    :param T: local static temperature
    :param M: local static Mach number
    :param gamma: ratio of specific heats
    :return: Stagnation temperature T0
    """
    ratio = (1 + (gamma-1)/2.0*M**2.0)
    return ratio*T


def r0(r, M, gamma):
    """Stagnation density calculator, use a value of r=1 to return stagnation density ratio (r0/r - stagnation/static)

    :param r: local static density
    :param M: local static Mach number
    :param gamma: ratio of specific heats
    :return: Stagnation density r0
    """
    ratio = (1 + (gamma-1)/2.0*M**2.0)**(1.0/(gamma-1))
    return ratio*r


def obliqueshock(theta, M1, p1, T1, r1, gamma):
    """Given an initial fluid dynamic state (M1, p1, T1, r1) and a ramp angle (theta), return a dictionary consisting of
    the post-oblique shock state (M2, p2, T2, r2, p02, T02), as well as normal Mach numbers (Mn1 and Mn2).

    :param theta: ramp angle in radians
    :param M1: upstream Mach number
    :param p1: upstream static pressure
    :param T1: upstream static temperature
    :param r1: upstream static density
    :param gamma: ratio of specific heats
    :return: result - A dictionary consisting of the listed state values that can be indexed with result["state"]
    """
    result = {}

    # Solve for turning angle for given upstream state
    beta = solvebeta(theta, M1, gamma)
    result['beta'] = beta

    # Normal Mach #
    Mn1 = M1 * math.sin(beta)

    # Post shock Mach
    Mn2 = Mpost(Mn1, gamma)
    M2 = Mn2 / math.sin(beta - theta)

    # Post shock state
    p2 = pratio(Mn1, gamma) * p1
    T2 = tratio(Mn1, gamma) * T1
    r2 = rratio(Mn1, gamma) * r1

    # Stagnation State
    p01 = p0(p1, M1, gamma)
    p02 = p0(p2, M2, gamma)

    T01 = T0(T1, M1, gamma)
    T02 = T0(T2, M2, gamma)

    result['p2'] = p2
    result['T2'] = T2
    result['r2'] = r2

    result['Mn1'] = Mn1
    result['Mn2'] = Mn2

    result['M1'] = M1
    result['M2'] = M2

    result['p01'] = p01
    result['p02'] = p02

    result['T01'] = T01
    result['T02'] = T02

    return result


def findtheta(M1, M2, gamma):
    """Returns the ramp angle, theta, needed to form an oblique shock to go from M1 to M2 assuming that the t-b-M
    relationship is valid.

    :param M1: upstream Mach
    :param M2: downstream Mach
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
        num = obliqueshock(theta, M1, P1, T1, r1, gamma)
        num = num['M2'] - M2

        Upper = obliqueshock(theta + epi, M1, P1, T1, r1, gamma)
        Upper = Upper['M2']

        Lower = obliqueshock(theta - epi, M1, P1, T1, r1, gamma)
        Lower = Lower['M2']

        denom = (Upper - Lower) / (2 * epi)

        theta = theta - num / denom

    return theta


def pmfunction(M, gamma):
    """Solves the Prandtl-Meyer formula for nu.

    :param M: local Mach number
    :param gamma: ratio of specific heats
    :return: nu from the Prandtl-Meyer formula
    """

    term1 = math.sqrt((gamma + 1) / (gamma - 1)) * math.atan(math.sqrt((gamma - 1) / (gamma + 1) * (M ** 2 - 1)))
    term2 = math.atan(math.sqrt(M ** 2 - 1))

    nu = term1 - term2
    return nu


def pmfunctionp(M, gamma):
    """Derivative of the Prandtl-Meyer formula used in solving for Mach number after an expansion fan.

    :param M: local Mach number
    :param gamma: ratio of specific heats
    :return: nu prime

    """
    C1 = ((gamma - 1) / (gamma + 1)) ** 0.5 * (M ** 2 - 1) ** 0.5
    C2 = (M ** 2 - 1) ** 0.5

    dC1 = ((gamma - 1) / (gamma + 1)) ** 0.5 * M * (M ** 2 - 1) ** -0.5
    dC2 = M * (M ** 2 - 1) ** -0.5

    constant = ((gamma + 1) / (gamma - 1)) ** 0.5
    term1 = constant * dC1 / (1 + C1 ** 2)
    term2 = dC2 / (1 + C2 ** 2)

    nup = constant * term1 - term2
    return nup


def PMsolveM(theta, gamma):
    """Solves for the Mach number after an expansion with turn angle of theta in radians

    :param theta: turn angle in radians
    :param gamma: ratio of specific heats
    :return: Mnew - Mach number after the turn angle
    """
    # Mnew >= 1 in order for PM-Fans to exist, so start here
    Mnew = 1.01

    # Newton-Raphson as the solver for the post expansion Mach number
    for i in range(100):
        Mnew = Mnew - (pmfunction(Mnew, gamma) - theta) / pmfunctionp(Mnew, gamma)

    return Mnew


def PMexpansion(theta, M1, T1, p1, gamma):
    """Solves for the state after an expansion with turn angle of theta in radians

    :param theta: turn angle in radians
    :param M1: upstream Mach number
    :param T1: upstream static temperature
    :param p1: upstream static pressure
    :param gamma: ratio of specific heats
    :return: result - Dictionary that can be indexed to get M2, T2, and p2
    """
    # Solve for the Mach number
    M2 = PMsolveM(theta + pmfunction(M1, gamma), gamma)

    # Temperature
    T1_T2 = (1 + (gamma - 1) / 2 * M2 ** 2) / (1 + (gamma - 1) / 2 * M1 ** 2)
    T2 = T1_T2 ** -1 * T1

    # Pressure
    P1_P2 = (T1_T2) ** (gamma / (gamma - 1))
    p2 = P1_P2 ** -1 * p1

    result = {'M2': M2, 'T2': T2, 'p2': p2}
    return result


@njit(cache=True)
def sutherland_viscosity(t, mu_ref, t_ref, S):
    """Calculates the viscosity using Sutherland's model for a given temperature.

    :param t: Temperature to evaluate viscosity at
    :param mu_ref: Reference viscosity (1.716e-5 Pa*s for CPG air)
    :param t_ref: Reference temperature (273 K for CPG air)
    :param S: Sutherland reference constant (111 K for CPG air)
    :return: mu - kinematic viscosity at temperature t
    """
    mu = np.multiply(np.multiply(mu_ref, np.power(t / t_ref, 1.5)), np.divide((t_ref + S), (t + S)))

    return mu