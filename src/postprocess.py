import numpy as np
import plotting
import helper


def postprocess(mesh, state, coefficients, residuals, config, i, t):
    """Performs to post-process applications (plotting and generating output file).

    :param mesh: Mesh in dictionary format
    :param state: Nx4 state vector
    :param coefficients: Performance coefficients Nx4
    :param residuals: Flow residuals
    :param config: Simulation json in dictionary format
    :param i: Identifier for plotting
    :param t: Simulation runtime
    """
    # Reference length calc
    ref_length = abs((mesh['V'][mesh['BE'][np.where(np.array(mesh['Bname']) == 'Wall')[0][0] == mesh['BE'][:, 3]][:, 0:2]][:, 0]).max() - \
                (mesh['V'][mesh['BE'][np.where(np.array(mesh['Bname']) == 'Wall')[0][0] == mesh['BE'][:, 3]][:, 0:2]][:, 0]).min())

    # Plotting handles all plotting commands
    plotting.plot_config(mesh, state, residuals, coefficients, config, i)

    # Output file writing
    f = open(config['filename'] + '.out', 'w')
    f.write('Simulation Runtime: {0}s\n'.format(t))
    f.write('Number of iterations: {0}\n'.format(residuals[:, 0].shape[0]))
    f.write('Time per iteration: {0}\n\n'.format(t / residuals[:, 0].shape[0]))

    # Simulation specific settings
    f.write('Simulation Configuration Parameters:\n')
    f.write('Simulation Name: {0}\n'.format(config['filename']))

    #Flux Method de-flagging
    if config['flux_method'] == 1: f.write('Flux Method: Roe\n')
    if config['flux_method'] == 2: f.write('Flux Method: HLLE\n')

    f.write('# of Adaptation Cycles: {0}\n'.format(config['adaptive_cycles']))
    f.write('Final Cell Count: {0}\n'.format(mesh['E'].shape[0]))

    # Convergence method de-flagging
    if config['convergence_method'] == 0:
        f.write('Convergence Method: Standard\n')
        f.write('ASCs Used: None\n\n')
    if config['convergence_method'] == 1:
        f.write('Convergence Method: Smart\n')
        asc_str = ''
        for asc in config['smart_convergence_ascs']:
            if asc == 0: asc_str += 'Drag, '
            if asc == 1: asc_str += 'Lift, '
            if asc == 2: asc_str += 'Pitch Moment, '
            if asc == 3: asc_str += 'ATPR,'
        f.write('ASCs Used: ' + asc_str + '\n\n')


    # Flight control conditions
    f.write('Flight Conditions:\n')
    f.write('Freestream Mach Number, M_inf: {0}\n'.format(config['freestream_mach_numer']))
    f.write('Freestream Velocity, U_inf: {0} m/s\n'.format(config['U_inf']))
    f.write('Freestream AoA, alpha: {0}\n'.format(config['angle_of_attack']))
    f.write('Freestream Altitude, h: {0}\n\n'.format(config['altitude']))


    # Flight performance quantities
    cf = helper.calculate_plate_friction(mesh['V'], mesh['BE'], state, config['mu_inf'], config['Rex_inf'],
                                         config['cv'], config['tinf'], config['viscosity_ref'],
                                         config['viscosity_ref_temp'], config['viscosity_ref_S'])
    f.write('Flight Performance:\n')
    f.write('Lift Coefficient, C_L: {0}\n'.format(coefficients[-1, 0]))
    f.write('Pressure Drag Coefficient, C_D: {0}\n'.format(coefficients[-1, 1]))
    f.write('Pitching Moment Coefficient, C_mx: {0}\n'.format(coefficients[-1, 2]))
    f.write('Average Total Pressure Recovery Factor (@ Exit BCs), ATPR: {0}\n'.format(coefficients[-1, 3]))
    f.write('Approx. Viscous Drag Coefficient, C_f: {0}\n'.format(cf))
    f.write('Approx. Total Drag Coefficient, C_D_tot: {0}\n\n'.format(cf + coefficients[-1, 1]))


    # Various reference and freestream quantities
    f.write('Reference Quantities:\n')
    f.write('Fluid: {0}\n'.format(config['name']))
    f.write('Fluid Cp: {0}\n'.format(config['cp']))
    f.write('Fluid Cv: {0}\n'.format(config['cv']))
    f.write('P_inf: {0} Pa\n'.format(config['pinf']))
    f.write('T_inf: {0} K\n'.format(config['tinf']))
    f.write('Rex_inf: {0} 1/m\n'.format(config['Rex_inf']))
    f.write('mu_inf: {0} Pa/s\n'.format(config['mu_inf']))
    f.write('Reference Length L: {0} m\n'.format(ref_length))
    f.close()