import moc_preproc as mpp
import readgri
import mesh_refinement as meshref
from operating_conditions import *
import residual_assembly as ra
import plotmesh
import numpy as np
import timeit
import initialization as intlzn
import preprocess as pp


def main():
    # Configuration for the CFD run
    op_con = operating_conditions(30000)

    # Grid pre-processing
    mesh = readgri.readgri(pp.sim_con['mesh_path']) # Mesh loading

    # Checking Characteristic Lines
    moc_lines = mpp.moc_inflow(mesh, op_con)
    moc_lines = mpp.moc_reflect(mesh, op_con, moc_lines)
    plotmesh.plot_moc(mesh, moc_lines, 'mesh_moc.png')

    start_time = timeit.default_timer()

    # Solution Initialization
    # state_variables = mpp.initialize_moc(mesh)  # [rho, rho*u, rho*v, rho*E]
    state_variables = intlzn.initialize_boundary(mesh) # [rho, rho*u, rho*v, rho*E]

    # Pre-solving plotting and visualization
    plotmesh.plot_config(mesh, state_variables, pp.data_con['filename'], -1)


    # Print which adaptation cycle we're on
    print('\nPerforming solve on adapted mesh #: 0.\n')
    # Run the actual residual assembly and solve for the flow field
    residuals = ra.euler_2D_v2(mesh, state_variables)
    # If we want restart files of the solved flow field, save them here
    if pp.sim_con['restart_files']: np.savetxt("{0}_state0.csv".format(pp.data_con['filename']), state_variables)
    if pp.sim_con['adaptive_cycles'] == 0: plotmesh.plot_config(mesh, state_variables, pp.data_con['filename'], 0)

    for i in range(pp.sim_con['adaptive_cycles']):
        # Refine the mesh and interpolate solution to the refined grid
        mesh, state_variables = meshref.refine_interp_uniform(mesh, state_variables,
                                                              pp.data_con['filename'] + '{0}.gri'.format(i + 1))
        # Plot according to the data plotting of the simulation configuration
        plotmesh.plot_config(mesh, state_variables, pp.data_con['filename'], i+1)
        # Solve on the new computational domain
        print('\nPerforming solve on adapted mesh #: {0}.\n'.format(i+1))
        residuals = ra.euler_2D_v2(mesh, state_variables)
        if pp.sim_con['restart_files']: np.savetxt("{0}_state{1}.csv".format(pp.data_con['filename'], i+1), state_variables)

    # Post-Processing - Plotting
    if pp.sim_con['adaptive_cycles'] != 0: plotmesh.plot_config(mesh, state_variables, pp.data_con['filename'] + '_solved', i)
    print('Simulation complete - check results files. Time to run: {0} seconds'.format(timeit.default_timer() - start_time))


if __name__ == '__main__':
    main()
