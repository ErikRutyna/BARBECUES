from skaero.atmosphere import coesa

# TODO: Convert this to JSON format
class operating_conditions:
    def __init__(self, h):
        # Program control flags
        self.restart_files =        False # Toggle on/off making restart files
        self.adaptation =           False # Toggle on/off adaptation algorithm
        self.adaptation_percentage = 0.05 # Toggle on/off adaptation refinement limits
        self.smart_convergence =    True # Toggle on/off smart convergence criteria
        self.flux_method =          'roe' # Which flux method used in the solver - 'roe'/'hlle'
        self.smart_convergence_minimum = 1e-2

        if self.smart_convergence:
            self.sma_counter = 42
            self.error_percent = 0.025

        if not self.adaptation:
            self.adaptation_cycles = 1
            pass
        else:
            self.adaptation_cycles = 2    # If we are adapting, the number of adaptation cycles

        # Initial conditions
        self.M = 2    # Freestream Mach number
        self.a = 10.0    # Vehicle AoA
        # Altitude, temperature, pressure, and density
        self.h, self.temp, self.pres, self.rho = coesa.table(h)

        # CPG Air Constants/Values
        self.cp = 1005              # J/kg-K
        self.cv = 718               # J/kg-K
        self.mw = 28.9647 / 1000    # kg/mol
        self.r  = 8.314 / self.mw   # J/kg-K
        self.y  = 1.40              # Ratio of specific heats