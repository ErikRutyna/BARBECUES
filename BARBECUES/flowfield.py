import math
import numpy as np



class flowfield:

    def __init__(self, n, M, aoa, y):

        # Number of cells in the flow field
        self.cellCount = n

        # Freestream Mach number of the flow field
        self.machNumber = M

        # Freestream angle of attack of the flow field
        self.angleOfAttack = aoa

        # Ratio of specific heats of the working fluid of the flow field
        self.gamma = y

        # Pointer towards the corresponding unstructured mesh
        self.unstructuredMesh = None

        # Initial empty array of state vectors
        self.stateVectors = np.zeros((self.cellCount, 4))

        # Freestream condition used as a reference
        self.freestreamState = np.array([
            1,
            self.machNumber * math.cos(self.angleOfAttack * math.pi / 180),
            self.machNumber * math.sin(self.angleOfAttack * math.pi / 180),
            1 / (self.gamma - 1) / self.gamma + self.machNumber ** 2 / 2])

    def addUnstructuredMesh(self, newUnstructuredMesh):
        self.unstructuredMesh = newUnstructuredMesh

    def initializeFreestream(self, scale=1):
        # Initialize the freestream condition based upon an optional scale factor
        self.stateVectors[:, 0] = 1
        self.stateVectors[:, 1] = scale * self.machNumber * math.cos(self.angleOfAttack * math.pi / 180)
        self.stateVectors[:, 2] = scale * self.machNumber * math.sin(self.angleOfAttack * math.pi / 180)
        self.stateVectors[:, 3] = 1 / (self.gamma - 1) / self.gamma + (scale * self.machNumber) ** 2 / 2