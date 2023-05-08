from skaero.atmosphere import coesa
import compressible_flow_formulas as cff
import math
import json


def preprocess():
    """Sets up the simulation pre-processing by reading the given json file and dumping its contents into the static
    Numba dictionary.

    :returns: Merged - a merged dictionary of various configurations setup in the config.json file.
    """
    # Read in the configuration file
    f = open('config.json')
    configuration = json.load(f)
    # Split the list into dictionaries for each configuration section
    sim_con = configuration['simulation'][0]
    conv_con = configuration['convergence'][0]
    init_con = configuration['initialization'][0]
    data_con = configuration['data_processing'][0]
    fluid_con = configuration['fluid'][0]
    flight_con = configuration['flight'][0]

    # Additional information about the working fluid
    fluid_con['y'] = fluid_con['cp'] / fluid_con['cv']
    fluid_con['R'] = 8.31446261815324 / fluid_con['MW']

    # Add freestream atmospheric conditions to the flight_con
    _, temp, pres, _ = coesa.table(flight_con['altitude'])
    flight_con['tinf'] = temp
    flight_con['pinf'] = pres
    flight_con['r_inf'] = flight_con['pinf'] / (fluid_con['R'] * flight_con['tinf'])
    flight_con['U_inf'] = flight_con['freestream_mach_numer'] * math.sqrt(
        fluid_con['y'] * fluid_con['R'] * flight_con['tinf'])
    flight_con['mu_inf'] = cff.sutherland_viscosity(flight_con['tinf'], fluid_con['viscosity_ref'], fluid_con['viscosity_ref_temp'], fluid_con['viscosity_ref_S'])
    flight_con['Rex_inf'] = flight_con['U_inf'] * flight_con['r_inf'] / flight_con['mu_inf']

    # Merge all dictionaries for numba support
    merged = sim_con | conv_con | init_con | data_con | fluid_con | flight_con

    return merged
