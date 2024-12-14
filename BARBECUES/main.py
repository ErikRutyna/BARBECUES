from BARBECUES.core import Simulation
import yaml
import timeit


def main():
    """This main function executes the BARBEQUES Euler Solver. General process
    follows this pattern:

    1). Read the config file and load in the mesh
    2). Initialize the state based on the method specified on the config
    2a). Plot the pre-solved flow-field and mesh
    3). Solve for the flow field using residual assembly
    4). Check if adaptation is to be run, if so go back to step 3, if not, then
        post-process the solution on the solved flow field.
    5). Generate a *.out file that contains simulation runtime information and some
        output data
    """
    # Begin tracking time and set up the simulation environment
    timeSimulationStart = timeit.default_timer()

    # All control parameters are held in the "config.yml" file - edit that to change simulation runtime
    with open('config.yml', 'r') as file:
            config = yaml.safe_load(file)

    # Initialize the simulation "Simulation" controller
    sim = Simulation.Simulation(config)

    runtime = timeit.default_timer() - timeSimulationStart


    print('Simulation complete after {0} seconds - check results files.'.format(runtime))
    return


if __name__ == '__main__':
    main()
