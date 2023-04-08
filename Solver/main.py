import os
import readgri
import mesh_refinement as meshref
import residual_assembly as ra
import plotting as plot
import numpy as np
import timeit
import initialization as intlzn
import preprocess as pp
import postprocess as pop
import shutil


# TODO: Comb over the entire code and look for optimizations
# TODO: Comb over the code and apply Python linting style where appropriate
def main():
    """This main function executes the BARBEQUES Euler Solver. General process
    follows this pattern:

    1). Read the config file and load in the mesh
    2). Initialize the state based on the method specified on the config
    2a). Plot the pre-solved flow-field and mesh
    3). Solve for the flow field using residual assembly
    4). Check if adaptation is to be run, if so go back to step 3, if not, then
        post-process the solution on the solved flow field.
    """
    # Begin tracking time and setup the simulation variables
    start_time = timeit.default_timer()

    config = pp.preprocess()
    residuals = np.empty((0, 5))
    coefficients = np.empty((0, 3))

    # Grid Loading
    mesh = readgri.readgri(config['mesh_path'])

    # Initialization is done based on the method supplied by the config
    state_vectors = intlzn.init_state(mesh, config)

    # Check and change to output directory
    if not os.path.isdir(os.path.join(os.getcwd(), '../Output/')):
        os.mkdir(os.path.join(os.getcwd(), '../Output/'))
    os.chdir(os.path.join(os.getcwd(), '../Output/'))

    # Move the MoC plot file to this directory if it exists
    if os.path.exists(os.path.join(os.path.dirname(os.getcwd()),
                                   'src/moc_lines.png')):
        shutil.move(os.path.join(os.path.dirname(os.getcwd()),
                                 'src/moc_lines.png'),
                    os.path.join(os.getcwd(), 'moc_lines.png'))

    # Pre-solution visualization
    plot.plot_config(mesh, state_vectors, residuals, coefficients, config, -1)

    # Print current adaptation cycle
    print('\nPerforming solve on adapted mesh #: 0.\n')

    # Run residual assembly and solve for the flow field
    flowfield_residuals, aerodynamic_coefficients = ra.euler_2D_v2(mesh['E'], mesh['V'],
                                                                   mesh['BE'], mesh['IE'],
                                                                   state_vectors,
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
    residuals = np.vstack((residuals, flowfield_residuals))
    coefficients = np.vstack((coefficients, aerodynamic_coefficients))

    # If we want restart files of the solved flow field, save them here
    if config['restart_files']:
        np.savetxt("{0}_state0.csv".format(config['filename']), state_vectors)

    # If no adaptive cycles, plot the solution & generate output file
    if config['adaptive_cycles'] == 0:
        runtime = timeit.default_timer() - start_time
        pop.postprocess(mesh, state_vectors, coefficients, residuals, config,
                        0, runtime)
        print('Simulation complete - check results files.')
        return

    # If adaptive cycles, being the adaptation process
    for i in range(config['adaptive_cycles']):
        # Plot the local adaptation cycle & solution state
        plot.plot_config(mesh, state_vectors, residuals, coefficients, config,
                         i)

        # Refine the mesh and interpolate solution to the refined grid
        state_vectors, mesh = meshref.adapt_mesh(state_vectors, mesh['E'],
                                                 mesh['V'], mesh['IE'],
                                                 mesh['BE'],
                                                 mesh['Bname'],
                                                 config['adaptation_percentage'],
                                                 config['y'],
                                                 config['filename'] + '{0}.gri'.format(i + 1))



        # Solve on the new computational domain
        print('\nPerforming solve on adapted mesh #: {0}.\n'.format(i+1))
        flowfield_residuals, aerodynamic_coefficients = ra.euler_2D_v2(mesh['E'], mesh['V'], mesh['BE'], mesh['IE'], state_vectors,
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
        residuals = np.vstack((residuals, flowfield_residuals))
        coefficients = np.vstack((coefficients, aerodynamic_coefficients))

        # Save the state if requested
        if config['restart_files']:
            np.savetxt("{0}_state{1}.csv".format(config['filename'], i+1),
                       state_vectors)

    # Plot the solution & generate output file
    runtime = timeit.default_timer() - start_time
    pop.postprocess(mesh, state_vectors, coefficients, residuals, config, i+1,
                    runtime)

    print('Simulation complete - check results files.')
    return


if __name__ == '__main__':
    main()
