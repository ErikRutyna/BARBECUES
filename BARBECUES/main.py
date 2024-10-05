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
import unstructuredMesh
import flowfield

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
    # Time tracking for performance and general informational assessment
    timeSimulationStart = timeit.default_timer()

    # Read the configuration file
    config = pp.preprocess()

    # Grid Loading
    demoMesh = unstructuredMesh.UnstructuredMesh(config['mesh_path'])

    # Flow field creation and initialization
    demoFlowfield = flowfield.flowfield(
        demoMesh.elements.shape[0],
        config['freestream_mach_numer'],
        config['angle_of_attack'],
        config['y'])
    demoFlowfield.initializeFreestream()

    # Link the mesh and flow field so they can see each other
    demoMesh.addFlowfield(demoFlowfield)
    demoFlowfield.addUnstructuredMesh(demoMesh)


    # Plot the solution & generate output file
    runtime = timeit.default_timer() - timeSimulationStart

    print('Simulation complete - check results files.')
    return


if __name__ == '__main__':
    main()
