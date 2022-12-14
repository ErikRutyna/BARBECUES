import os
import datetime
import moc_preproc
import moc_preproc as mpp
import readgri
import mesh_processing
from operating_conditions import *
import residual_assembly as ra
import plotmesh
import numpy as np
import timeit


def main():
    # Configuration for the CFD run
    op_con = operating_conditions(30000)

    mesh_folder = 'Inlets'
    mesh_file = 'mesh3.gri'

    run_dir = os.getcwd()
    mesh_dir = os.path.join(run_dir, 'Meshes')

    # Grid pre-processing
    mesh = readgri.readgri(os.path.join(mesh_dir, mesh_folder, mesh_file)) # Mesh loading
    mesh_filename = 'inlet' + '.png'
    plotmesh.plotmesh(mesh, mesh_filename)

    # Checking Characteristic Lines
    # moc_lines = mpp.moc_inflow(mesh, op_con)
    # moc_lines = mpp.moc_reflect(mesh, op_con, moc_lines)
    # plotmesh.plot_moc(mesh, moc_lines, 'mesh_moc.png')

    start_time = timeit.default_timer()
    # Solution Initialization
    # state_variables = moc_preproc.initialize_moc(mesh, op_con)  # [rho, rho*u, rho*v, rho*E]
    state_variables = mesh_processing.initialize_boundary(mesh, op_con) # [rho, rho*u, rho*v, rho*E]

    plotmesh.plotmesh_values(mesh, state_variables, op_con, 'inlet_{0}_M{1}_a{2}_not_solved.png'
                             .format(op_con.flux_method, op_con.M, op_con.a))
    if op_con.adaptation:
        # Initial solve
        print('\nPerforming initial solve on initial mesh\n')
        ra.euler_2D_v2(op_con, mesh, state_variables)
        np.savetxt("init_solve_state.txt", state_variables)
        state_variables = np.loadtxt('init_solve_state.txt')

        for i in range(op_con.adaptation_cycles):
            print('\nRefining mesh: {0}\n'.format(i))
            # Refine the mesh and interpolate solution to the refined grid
            mesh, state_variables = mesh_processing.refine_interp_uniform(mesh, state_variables,
                                                                          'mesh{0}.gri'.format(i+1), op_con)

            # Save the new mesh and interpolated state figures
            mesh_filename = 'mesh' + str(i+1) + '.png'
            plotmesh.plotmesh(mesh, mesh_filename)
            plotmesh.plotmesh_values(mesh, state_variables, op_con, '{0}_M{1}_a{2}_interpolated_{3}.png'
                                     .format(op_con.flux_method, op_con.M, op_con.a, i))

            # Solve on the new computational domain
            print('\nSolving on mesh: {0}\n'.format(i+1))
            ra.euler_2D_v2(op_con, mesh, state_variables)
    else:
        pass
        ra.euler_2D_v2(op_con, mesh, state_variables)
        # Save the mesh/state variables
        if op_con.restart_files:
            np.savetxt("mesh7_state.txt", state_variables)

    # Post-Processing - Plotting
    plotmesh.plotmesh_values(mesh, state_variables, op_con, 'inlet_{0}_M{1}_a{2}_solved.png'
                             .format(op_con.flux_method, op_con.M, op_con.a))
    print('Simulation complete - check results files. Time to run: {0} seconds'.
          format(timeit.default_timer() - start_time))


if __name__ == '__main__':
    main()
