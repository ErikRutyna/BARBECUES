# B.A.R.B.E.C.U.E.S. & GRILS V1.4.0
B.A.R.B.E.C.U.E.S. (Basically Another Really Badly Enhanced Compressible 
Unstructured Euler Solver) is a 2D CFD code developed in Python that solves the
compressible Euler equations. The code is meant to be a testing ground for 
novel ideas and features to push the limits of traditional CFD codes by doing 
standard processes (i.e., initialization or convergence) in non-standard ways. 
More information about such features can be found in the Wiki (implemented 
"soon"). The simulation is controlled through a JSON file called config.json, 
and an example can be found in the source code. 

The solver also has another utility that can be used to generate meshes called
the G.R.I.L.S Mesher (*.GRI Level Set Mesher). The mesher  is similar in 
architecture to the DistMesh (in that it runs the core underlying level-set 
based method for producing high quality meshes), but has some of its own 
features and changes. Some examples of how to use GRIFT can be seen in 
the "mesh_generation.py" script.

NOTE: There appear to be some issues regarding freestream conditions with 
different initialization methods and getting the flowfield to converge. It is
recommended to use either "freestream" or "exp" based initialization methods. 
Eventually all methods for initialization will be replaced with either a level 
set method or strict freestream method.

## Boundary Conditions
BCs must follow this convention in order for the NUMBA JIT to properly 
function. The identifiers are:

| Boundary Condition | Flag Number |
|--------------------|-------------|
| Inviscid Wall      | 0           |
| (Supersonic) Exit  | 1           |
| Supersonic Outflow | 2           |
| Supersonic Inflow  | 3           |


## Flux Methods
The inviscid flux method must follow this convention in order for the NUMBA JIT
to properly function. The identifiers are :

| Flux Method | Flag Number |
|-------------|-------------|
| Roe Flux    | 1           |
| HLLE Flux   | 2           |

If no number is specified, or an invalid number is provided, the solver will 
default to the Roe Flux.


## Convergence Methods
There are two options for convergence: Smart and Standard. Smart convergence 
detects when certain solution-dependent variables (ASCs) become asymptotic and
terminate the simulation early. Standard convergence uses the L1 residual norm 
of the Euler equations to determine convergence. The flag for this is “0” for 
Standard Convergence, and “1” for Smart Convergence. If using smart convergence
you need a minimum of one of the following ASCs:

| ASC                                                           | Flag Number |
|---------------------------------------------------------------|-------------|
| Drag Coefficient $$(C_D)$$                                    | 0           |
| Lift Coefficient $(C_L)$                                      | 1           |
| Pitching Moment Coefficient $(C_MX)$                          | 2           |
| Average Total Pressure Recovery Factor (@ (Supersonic) Exits) | 3           |


## Initialization Methods
There are 3 unique initialization methods for initializing the state variables in 
the cells of the domain.

| Initialization Method | Effect                                                                                                                                                                                                                                                                  |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| freestream            | All cells have freestream initial condition.                                                                                                                                                                                                                            |
| sdf                   | The Mach number is scaled via the minimum signed distance value to all internal walls and then that Mach number is used to compute the freestream state.                                                                                                                |
| moc                   | Propagates characteristic lines from the inflow and reflects <br/> them off of the inviscid walls. Uses aforementioned reflections in order to create zones for oblique shock trains which can scale Mach number down according to the location within the shock train. |


# Changelog
V1.4.0 GRILS Mesher has had its documentation updated and and default values optimized using particle swarm 
optimization. All the things I wished to accomplish with GRILS have been done and it will be moving into a finished 
state. V1.5.0 will contain a manual with in-depth information regarding the program, configurations, examples, and 
information regarding its optimization process. Old initialization methods have been depreciated and removed, the only 
two remaining methods are the "freestream", "sdf", and "moc" based methods. 

V1.3.1.1 Small change to flux.py to avoid extra computations resulting in 
~4.5% increase in solver speed. Also fixed *.out files using the wrong value
for the pressure drag when computing approximated total drag.

V1.3.1 Additional geometries added to GRILS.  

V1.3.0 A new meshing utility available under GRILS, which can 
be used to generate basic meshes of shapes object in some flow-field. V1.3.0 
can only produce flat plates and circles, and a combination of them can be 
modeled in a flow-field. More complex geometries are going to be added in 
next version.

To generate a mesh, simply execute the “mesh_generation.py“ script. The config 
file provided under /src/ is already set up to work with this demo mesh.

NOTE: If simulation is struggling to converge past the first few iterations 
with a mesh generated by this script, it is recommended to change the 
initialization condition. The solver runs First-Order with Forward-Euler and 
no limiter, as such the solver can struggle to "find its feet" while simulating
the flow field.

V1.2.8 Fixed the Characteristic-line-like initialization method that 
initializes the initial state by running characteristic lines from the inflow 
to outflow and looking at possible location for oblique shock trains in the 
domain.

V1.2.7 Added additional initialization method (“exp”) that uses exponential 
scaling for scaling initial freestream state at each cell. Sometimes this 
is more optimal (fewer iterations til convergence) than other methods. 
Configured simulation to generate an “Output” directory that contains all 
output files. Also clarified plotting labels and fixed small error regarding 
output file where the Lift and Drag coefficient values were swapped.

V1.2.6 Original AMR algorithm (mesh_refinement.refine_interp_uniform()) works, 
also newer and better refinement algorithm (mesh_refinement.adapt_mesh()) 
that is fully Numba compatible, so it is both faster and produces better 
quality meshes.

V1.2.5 Numba JIT integration for continued speed increases. AMR & MOC-based 
initialization are temporarily broken.

V1.2 Conversion of simulation input parameters to a JSON format and source code
reconfiguration.

V1.1 Continued bug fixes, further performance increases via vectorization of 
operations and further in-code documentation.

V1.0.1 Minor bug fixes, performance optimizations, and in-code documentation.

V1.0. Initial commit of the base code.
