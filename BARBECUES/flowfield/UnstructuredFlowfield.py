from BARBECUES.flowfield.initialization import initializeFreestream
import math
import numpy as np

class UnstructuredFlowfield:

    def __init__(self, sim):
        # Self-link the sim to the flowfield for data access
        self.simulation = sim
        self.unstructuredMesh = self.simulation.unstructuredMesh

        # Cell Information
        self.cellCount = self.unstructuredMesh.cellArea.size

        # Empty state vectors, gradients, fluxes, and transport speed arrays
        self.stateVectors               = np.zeros((self.cellCount, 4))
        self.stateGradients             = np.zeros((self.cellCount, 4, 2))
        self.stateFluxes                = np.zeros((self.cellCount, 4))
        self.residuals                  = np.zeros((self.cellCount, 4))
        self.statePropagationSpeed      = np.zeros(self.cellCount)

        # Farfield flight conditions
        self.machNumber                 = sim.config['freestream'][0]['mach']
        self.angleOfAttack              = sim.config['freestream'][0]['angle_of_attack']

        # Working fluid information
        self.fluidName                  = sim.config['fluid'][0]['name']
        self.cp                         = sim.config['fluid'][0]['cp']
        self.cv                         = sim.config['fluid'][0]['cv']
        self.gamma                      = self.cp / self.cv
        self.molecularWeight            = sim.config['fluid'][0]['MW']
        self.refViscosity               = sim.config['fluid'][0]['reference_viscosity']
        self.refViscosityTemperature    = sim.config['fluid'][0]['reference_viscosity_temperature']
        self.refViscosityConstant       = sim.config['fluid'][0]['reference_viscosity_constant']

        # Freestream state conditions
        self.freestreamState = np.array([
            1,
            self.machNumber * math.cos(self.angleOfAttack * math.pi / 180),
            self.machNumber * math.sin(self.angleOfAttack * math.pi / 180),
            1 / (self.gamma - 1) / self.gamma + self.machNumber ** 2 / 2])

        # Initialize the flow field with respect to what's stated in the config
        self.initialize()

    def initialize(self):
        match self.simulation.config['preprocessing'][0]['initialization']:
            case "freestream":
                initializeFreestream.initializeFreestream(self.stateVectors, self.machNumber, self.angleOfAttack, self.gamma)
            case "weak":
                pass
            case "linear":
                pass
            case "exponential":
                pass
            case "scaled":
                pass
            case "moc":
                pass