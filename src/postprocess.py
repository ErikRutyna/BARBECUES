import plotting
import numpy as np
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
    # TODO: Better format this and have it translate numbers -> what was used in the *.out file
    # Reference length calc
    ref_length = abs((mesh['V'][mesh['BE'][np.where(np.array(mesh['Bname']) == 'Wall')[0][0] == mesh['BE'][:, 3]][:, 0:2]][:, 0]).max() - \
                (mesh['V'][mesh['BE'][np.where(np.array(mesh['Bname']) == 'Wall')[0][0] == mesh['BE'][:, 3]][:, 0:2]][:, 0]).min())

    # Plotting handles all plotting commands
    plotting.plot_config(mesh, state, residuals, coefficients, config, i)

    # Output file writing
    f = open(config['filename'] + '.out', 'w')

    # Simulation specific settings
    f.write('Simulation Configuration Parameters:\n')
    f.write('Simulation Name: \t {0}\n'.format(config['filename']))
    f.write('Flux Method: \t {0}\n'.format(config['flux_method']))
    f.write('# of Adaptation Cycles: \t {0}\n'.format(config['adaptive_cycles']))
    f.write('Final Cell Count: \t {0}\n'.format(mesh['E'].shape[0]))
    f.write('Convergence Method: \t {0}\n'.format(config['convergence_method']))
    if config['convergence_method'] == 1: f.write('ASCs Used: \t {0}\n'.format(config['smart_convergence_ascs']))
    if config['convergence_method'] == 0: f.write('ASCs Used: \t None\n')

    # Flight control conditions
    f.write('Flight Conditions:\n')
    f.write('Freestream Mach Number, M_inf: \t {0}\n'.format(config['freestream_mach_numer']))
    f.write('Freestream Velocity, U_inf: \t {0}\n'.format(config['U_inf']))
    f.write('Freestream AoA, alpha: \t {0}\n'.format(config['angle_of_attack']))
    f.write('Freestream Altitude, h: \t {0}\n'.format(config['altitude']))

    # Flight performance quantities
    cf = helper.calculate_plate_friction(mesh['V'], mesh['BE'], state, config['mu_inf'], config['Rex_inf'],
                                         config['cv'], config['tinf'], config['viscosity_ref'],
                                         config['viscosity_ref_temp'], config['viscosity_ref_S'])
    f.write('Flight Performance:\n')
    f.write('Lift Coefficient, C_L: \t {0}\n'.format(coefficients[-1, 0]))
    f.write('Pressure Drag Coefficient, C_D: \t {0}\n'.format(coefficients[-1, 1]))
    f.write('Pitching Moment Coefficient, C_mx: \t {0}\n'.format(coefficients[-1, 2]))
    f.write('Average Total Pressure Recovery Factor (@ Exit BCs), ATPR: \t {0}\n'.format(coefficients[-1, 3]))
    f.write('Approx. Viscous Drag Coefficient, C_f: \t {0}\n'.format(cf))
    f.write('Approx. Total Drag Coefficient, C_D_tot: \t {0}\n'.format(cf + coefficients[-1, 1]))

    # Various reference and freestream quantities
    f.write('Reference Quantities:\n')
    f.write('Fluid: \t {0}\n'.format(config['name']))
    f.write('Fluid Cp: \t {0}\n'.format(config['cp']))
    f.write('Fluid Cv: \t {0}\n'.format(config['cv']))
    f.write('P_inf: \t {0} Pa\n'.format(config['pinf']))
    f.write('T_inf: \t {0} K\n'.format(config['tinf']))
    f.write('Rex_inf: \t {0} 1/m\n'.format(config['Rex_inf']))
    f.write('mu_inf: \t {0} Pa/s\n'.format(config['mu_inf']))
    f.write('Reference Length L: \t {0} m\n'.format(ref_length))
    f.close()