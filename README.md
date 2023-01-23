# B.A.R.B.E.C.U.E.S. V1.2.7
B.A.R.B.E.C.U.E.S. (Basically Another Really Badly Enhanced Compressible Unstructured Euler Solver) is a 2D CFD code developed in Python that solves the compressible Euler equations. The code is meant to be a testing ground for novel ideas and features in an effort to push the limits of traditional CFD codes by doing standard processes (i.e. initialization or convergence) in non-standard ways. More information about such features can be found in the Wiki (WIP).


The simulation is controlled through a JSON file called config.json and an example can be found in the source code; a demonstration mesh is currently not provided, and the only meshes accepted are those in *.gri format.

## Boundary Conditions
BCs must follow this convention in order for the NUMBA JIT to properly function. The identifiers are as follows:

| Boundary Condition | Flag Number |
|--------------------|-------------|
| Inviscid Wall | 0 |
| (Supersonic) Exit | 1 |
| Supersonic Outflow | 2 |
| Supersonic Inflow | 3 |


## Flux Methods
The inviscid flux method must follow this convention in order for the NUMBA JIT to properly function. The identifiers are as follows:

| Flux Method | Flag Number |
|--------------------|-------------|
| Roe Flux | 1 |
| HLLE Flux | 2 |

If no number is specified, or an invalid number is provided, the solver will default to the Roe Flux.



## Convergence Methods
There are two options for convergence: Smart and Standard. Smart convergence detects when certain solution-dependent variables (ASCs) become asymptotic and terminate the simulation early. Standard convergence uses the L1 residual norm of the Euler equations to determine convergence. The flag for these is "0" for Standard Convergence, and "1" for Smart Convergence. If using smart convergence you need some number of the following ASCs:

| ASC | Flag Number |
|--------------------|-------------|
| Drag | 0 |
| Lift | 1 |
| Pitching Moment | 2 |
| Average Total Pressure Recovery Factor (@ (Supersonic) Exits) | 3 |

# Changelog
V1.2.7 Added additional initialization method ("exp") that uses expoential scaling for scaling initial freestream state at each cell. In some cases this is more optimal (less iterations til converged) than other methods. Configured simulation to generate an "Output" directory that contains all output files. Also clarified plotting labels and fixed small error regarding output file where the Lift and Drag coefficient values were swapped.

V1.2.6 Original AMR algorithm (mesh_refinement.refine_interp_uniform()) works, also newer and better refinement algorithm (mesh_refinement.adapt_mesh()) that is fully Numba compatible, so it is both faster and produces better quality meshes.

V1.2.5 Numba JIT integration for continued speed increases. AMR & MOC-based initialization are temporarily broken until it has been updated to work with Numba JIT.

V1.2 Conversion of simulation input parameters to a JSON format and source code reconfiguration.

V1.1 Continued bug fixes, further performance increases via vectorization of operations and further in-code documentation.

V1.0.1 Minor bug fixes, performance optimizations, and in-code documentation.

V1.0. Initial commit of the base code.
