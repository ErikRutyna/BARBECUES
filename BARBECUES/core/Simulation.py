from BARBECUES.mesh import UnstructuredMesh
from BARBECUES.flowfield import UnstructuredFlowfield

class Simulation:

    def __init__(self, config):
        # Single simulation parameter set
        self.config = config

        # Mesh is loaded and all mesh-related routines (edge length calcs, cell area calcs, etc.) are executed within
        # the constructor when it runs
        self.loadMesh()

        # Flow field generation and initialization
        self.generateFlowfield()
        self.unstructuredMesh.unstructuredFlowfield = self.unstructuredFlowfield

        # Preprocessing routine

        # Simulation execution

        # Postprocessing routine



    # Grab the mesh and have it self-initialize with all the necessary information
    def loadMesh(self):
        self.unstructuredMesh = UnstructuredMesh.UnstructuredMesh(self)

    # Generate the array that represents the flow field and initializes it based upon the method supplied by the config
    def generateFlowfield(self):
        self.unstructuredFlowfield = UnstructuredFlowfield.UnstructuredFlowfield(self)


    def executeSimulation(self):
        pass