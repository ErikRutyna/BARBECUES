from BARBECUES.mesh import UnstructuredMesh
from BARBECUES.flowfield import UnstructuredFlowfield
from BARBECUES.utilities.plotting import *


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
        self.preprocess()

        # Simulation execution
        self.executeSimulation()

        # Postprocessing routine
        self.postprocess()

    # Grab the mesh and have it self-initialize with all the necessary information
    def loadMesh(self):
        self.unstructuredMesh = UnstructuredMesh.UnstructuredMesh(self)

    # Generate the array that represents the flow field and initializes it based upon the method supplied by the config
    def generateFlowfield(self):
        self.unstructuredFlowfield = UnstructuredFlowfield.UnstructuredFlowfield(self)

    def preprocess(self):
        if self.config['postprocessing'][0]['plots']['mesh']:
            plotMesh.plotMesh(self.unstructuredMesh.nodes, self.unstructuredMesh.elements,
                              self.unstructuredMesh.boundaryEdges, self.unstructuredMesh.boundaryName,
                              self.config['project'][0]['filename'] + "_mesh_0.png")
        if self.config['postprocessing'][0]['plots']['Mach']:
            plotMach.plotMach(self.unstructuredMesh.nodes, self.unstructuredMesh.elements,
                              self.unstructuredFlowfield.stateVectors, self.unstructuredFlowfield.gamma,
                              self.config['project'][0]['filename'] + "_Mach_0.png")

    def executeSimulation(self):
        pass

    def postprocess(self):
        pass