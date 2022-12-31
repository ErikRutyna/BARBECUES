import moc_preproc as mpp
import readgri
import mesh_refinement as meshref
import residual_assembly as ra
import plotting
import numpy as np
import timeit
import initialization as intlzn
import preprocess as pp
import postprocess as pop


# # Configuration for the CFD run
# op_con = operating_conditions(30000)

# # Checking Characteristic Lines
# moc_lines = mpp.moc_inflow(mesh, op_con)
# moc_lines = mpp.moc_reflect(mesh, op_con, moc_lines)
# plotting.plot_moc(mesh, moc_lines, 'mesh_moc.png')
# state_variables = mpp.initialize_moc(mesh)  # [rho, rho*u, rho*v, rho*E]

def main():
    # Setup configuration and tracking variables
    start_time = timeit.default_timer()

    config = pp.preprocess()
    residuals = np.empty((0, 5))
    coefficients = np.empty((0, 4))

    # Grid Loading
    mesh = readgri.readgri(config['mesh_path'])

    # TODO: Solution initialization driver function
    # Solution Initialization
    # state_vectors = intlzn.initialize_boundary_dist(mesh['E'], mesh['V'],
    #                                                 config['freestream_mach_numer'],
    #                                                 config['angle_of_attack'],
    #                                                 config['y'])
    state_vectors = intlzn.initialize_boundary_weak(mesh['E'].shape[0],
                                                      config['freestream_mach_numer'],
                                                      config['angle_of_attack'],
                                                      config['y'])

    # Pre-solution visualization
    plotting.plot_config(mesh, state_vectors, residuals, coefficients, config, -1)

    # Print current adaptation cycle
    print('\nPerforming solve on adapted mesh #: 0.\n')

    # Run residual assembly and solve for the flow field
    residuals_temp, coefficients_temp = ra.euler_2D_v2(mesh['E'], mesh['V'], mesh['BE'], mesh['IE'], state_vectors,
                                                       config['freestream_mach_numer'],
                                                       config['angle_of_attack'],
                                                       config['y'],
                                                       config['flux_method'],
                                                       config['convergence_method'],
                                                       config['convergence_minimum'],
                                                       config['smart_convergence_minimum'],
                                                       config['smart_convergence_length'],
                                                       config['smart_convergence_error_tol'],
                                                       np.array(config['smart_convergence_ascs']))
    residuals = np.vstack((residuals, residuals_temp))
    coefficients = np.vstack((coefficients, coefficients_temp))

    # If we want restart files of the solved flow field, save them here
    if config['restart_files']: np.savetxt("{0}_state0.csv".format(config['filename']), state_vectors)
    # If no adaptive cycles, plot the solution & generate output file
    if config['adaptive_cycles'] == 0:
        runtime = timeit.default_timer() - start_time
        pop.postprocess(mesh, state_vectors, coefficients, residuals, config, 0, runtime)
        print('Simulation complete - check results files.')
        # return
    foo = meshref.find_flagged_edges(state_vectors, mesh['E'], mesh['V'], mesh['IE'], mesh['BE'], config['adaptation_percentage'], config['y'])
    return
    # If adaptive cycles, being the adaptation process
    for i in range(config['adaptive_cycles']):
        # Refine the mesh and interpolate solution to the refined grid
        mesh, state_vectors = meshref.refine_interp_uniform(mesh, state_vectors,
                                                              config['filename'] + '{0}.gri'.format(i + 1))
        # Plot according to the data plotting of the simulation configuration
        plotting.plot_config(mesh, state_vectors, residuals, coefficients, config['filename'], i+1)
        # Solve on the new computational domain
        print('\nPerforming solve on adapted mesh #: {0}.\n'.format(i+1))
        residuals_temp, coefficients_temp = ra.euler_2D_v2(mesh, state_vectors, config)
        residuals = np.vstack((residuals, residuals_temp))
        coefficients = np.vstack((coefficients, coefficients_temp))
        if config['restart_files']: np.savetxt("{0}_state{1}.csv".format(config['filename'], i+1), state_vectors)

    # Plot the solution & generate output file
    runtime = timeit.default_timer() - start_time
    pop.postprocess(mesh, state_vectors, coefficients, residuals, config, i+1, runtime)
    print('Simulation complete - check results files.')
    return



if __name__ == '__main__':
    main()
