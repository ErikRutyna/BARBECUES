import math
import numpy as np


def f(beta, theta, M1, gamma):
    # This function refers to the relation between M1, beta and theta and is written as
    # fout(beta,theta, M1, gamma) =  f(M1, beta, gamma) - tan(theta)

    numerator = M1 ** 2 * np.sin(beta) ** 2 - 1
    denomenator = M1 ** 2 * (gamma + np.cos(2 * beta)) + 2

    fout = 2 * (np.tan(beta)) ** -1 * numerator / denomenator - np.tan(theta)

    return fout


def fp(beta, theta, M1, gamma):
    # If you are using Newton-Raphson method, you will need f' = df/dbeta
    # Write this analytically here

    h = 2 * np.tan(beta) ** -1
    hp = -2 * math.sin(beta) ** -2

    u = M1 ** 2 * math.sin(beta) ** 2 - 1
    up = M1 ** 2 * math.sin(2 * beta)

    v = M1 ** 2 * (gamma + np.cos(2 * beta)) + 2
    vp = -2 * M1 ** 2 * math.sin(2 * beta)

    fout = hp * u / v + h * (v * up - vp * u) / v ** 2

    return fout

def solvebeta(theta, M1, gamma):
    # Given a theta, M1 and gamma, this function solver for beta using f and f'. Returns beta as solution

    if theta == 0:
        beta = np.pi / 2
        return beta

    # Initial guess for beta
    beta = theta * np.pi / 180

    # Its either going to converge within 10 and be near machine precision,
    # or it won't converge at all at this point
    for i in range(100):
        beta = beta - f(beta, theta, M1, gamma) / fp(beta, theta, M1, gamma)

        # print("Iteration: {0 \t beta = {1}".format(i+1, beta))

    # print("Beta is {0}".format(beta))
    return beta

def pratio(M,gamma):
    pr = 1 +2*gamma/(gamma+1)*(M**2.0-1)
    return pr

def rratio(M,gamma):
    rr = (gamma+1)*M**2.0/((gamma-1)*M**2.0+2.0)
    return rr

def tratio(M,gamma):
    pr = pratio(M,gamma)
    rr = rratio(M,gamma)
    tr = pr/rr
    return tr

def Mpost(M,gamma):
    num = M**2.0 + (2.0/(gamma-1))
    den = 2.0*gamma/(gamma-1)*M**2.0 -1
    return (num/den)**0.5

def p0(p,M,gamma):
    ratio = (1 + (gamma-1)/2.0*M**2.0)**(gamma/(gamma-1))
    return ratio*p

def T0(T,M,gamma):
    ratio = (1 + (gamma-1)/2.0*M**2.0)
    return ratio*T

def r0(r,M,gamma):
    ratio = (1 + (gamma-1)/2.0*M**2.0)**(1.0/(gamma-1))
    return ratio*r


def obliqueshock(theta, M1, p1, T1, r1, gamma):
    result = {}

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
    # print(p2/p1)

    # Stagnation State
    p01 = p0(p1, M1, gamma)
    p02 = p0(p2, M2, gamma)

    T01 = T0(T1, M1, gamma)
    T02 = T0(T2, M2, gamma)

    # You can store results in paired list called result, and input numbers as follows

    result['p2'] = p2
    result['T2'] = T2
    result['r2'] = r2

    result['Mn1'] = Mn1
    result['Mn2'] = Mn2

    result['M2'] = M2

    result['p01'] = p01
    result['p02'] = p02

    result['T01'] = T01
    result['T02'] = T02

    return result


def findtheta(M1, M2, gamma):
    # Note that the results do not depend on pressure or temperature, but you need to use the pressure/temperature ratio
    # functions defined before. For this, you need to send some dummy values.

    # The function should return the angle theta

    # Dummy values for ratios
    P1 = 101325
    T1 = 298.15
    r1 = 1.225

    # Initial value for theta
    theta = 25 * np.pi / 180

    # Tiny wiggle for derivative
    epi = 0.0001

    for i in range(100):
        num = obliqueshock(theta, M1, P1, T1, R1, gamma)
        num = num['M2'] - M2

        Upper = obliqueshock(theta + epi, M1, P1, T1, R1, gamma)
        Upper = Upper['M2']
        # print(Upper)

        Lower = obliqueshock(theta - epi, M1, P1, T1, R1, gamma)
        Lower = Lower['M2']
        # print(Lower)

        denom = (Upper - Lower) / (2 * epi)

        theta = theta - num / denom

    return theta


def pmfunction(M, gamma):
    term1 = math.sqrt((gamma + 1) / (gamma - 1)) * math.atan(math.sqrt((gamma - 1) / (gamma + 1) * (M ** 2 - 1)))
    term2 = math.atan(math.sqrt(M ** 2 - 1))

    nu = term1 - term2
    return nu


def pmfunctionp(M, gamma):
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
    # Assumes theta is always given in radians

    # Mnew >= 1 in order for PM-Fans to exist, so start here
    Mnew = 1.01

    # Its either going to converge within 10 and be near machine precision,
    # or it won't converge at all at this point
    for i in range(100):
        Mnew = Mnew - (pmfunction(Mnew, gamma) - theta) / pmfunctionp(Mnew, gamma)

    return Mnew


def PMexpansion(theta, M1, T1, p1, gamma):
    # Have to account for nu1 no longer being zero
    M2 = PMsolveM(theta + pmfunction(M1, gamma), gamma)

    T1_T2 = (1 + (gamma - 1) / 2 * M2 ** 2) / (1 + (gamma - 1) / 2 * M1 ** 2)
    # print(T1_T2)
    T2 = T1_T2 ** -1 * T1

    P1_P2 = (T1_T2) ** (gamma / (gamma - 1))
    # print(P1_P2)
    p2 = P1_P2 ** -1 * p1

    result = {'M2': M2, 'T2': T2, 'p2': p2}
    return result