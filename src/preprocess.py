from skaero.atmosphere import coesa
import json


"""Sets up the simulation pre-processing by reading the given json file.

:param fname: The json filename/filename path.
:return *_con: configuration for that section of the program, i.e. simulation controls simulation parameters (i.e.
flux method), or fluid_con controls information about the working fluid (i.e. CPG air has gamma = 1.4)
"""
# TODO: Have the pre-process make a specialized mega numba typed dictionary that contains everything of all the other dictionaries
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

