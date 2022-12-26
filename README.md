# B.A.R.B.E.C.U.E.S.
B.A.R.B.E.C.U.E.S. (Basically Another Really Badly Enhanced Compressible Unstructured Euler Solver) is a 2D CFD code developed in Python that solves the compressible Euler equations. The code is meant to be a testing ground for novel ideas and features in an effort to push the limits of traditional CFD codes by doing standard processes (i.e. initialization or convergence) in non-standard methods. More information about such features can be found in the Wiki.

Boundary conditions must follow this convention in order for the NUMBA JIT to properly function as it cannot easily do string-based comparisons. The boundary condition identifiers are as follows:
0: Inviscid Wall

1: Supersonic Exit (Used in tracking pressure @ exit of intakes)

2: Supersonic Outflow (Dumptank style outflow)

3: Supersonic Inflow (Freestream @ Mach Number)


For the Flux Method use the following options

1: Upwind Roe Flux

2: HLLE Flux


For Convergence use the following options:

0/1: For Standard/Smart convergence w/ ASCs for Smart convergence as follows:

0: Drag

1: Lift

2: Pitching Moment

3: ATPR (Only useful if Supersonic Exist BC exists)


# Changelog
V1.3 Numba JIT integration for continued speed increases.

V1.2 Conversion of simulation input parameters to a JSON format and source code reconfiguration.

V1.1 Continued bug fixes, further performance increases via vectorization of operations and further in-code documentation.

V1.0.1 Minor bug fixes, performance optimizations, and in-code documentation.

V1.0. Initial commit of the base code.
