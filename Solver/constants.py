# These are all conversion factors to go from SI units to imperial units or vice versa.

# Default is in SI units
g = 9.8067

# Pressure
# Pascals -> PSI and PSI -> Pascals
pa2psi = 0.000145038
psi2pa = 1 / pa2psi

# Temperature
# Kelvin -> Rankine and Rankine -> Kelvin
k2r = 1.8
r2k = 1 / k2r
# Kelvin -> Celsius and Celsius -> Kelvin
k2c = 273.15
c2k = -273.15
# Rankine -> Fahrenheit and Fahrenheit -> Rankine
r2f = -458.67
f2r = 458.67

# Density
# kg/m^3 -> lbm/ft^3 and lbm/ft^3 -> kg/m^3
kgcm2lbmcf = 0.062428
lbmcf2kgcm = 1 / kgcm2lbmcf

# Force
# Newton -> lbf and lbf -> Newton
n2lbf = 0.2248090795
lbf2n = 1 / n2lbf

# Mass
# kg -> lbm and lbm -> kg
kg2lbm = 2.20462
lbm2kg = 1 / kg2lbm

# Velocity
# m/s -> mph and mph -> m/s
mps2mph = 2.23694
mph2mps = 1 / mps2mph

# m/s -> ft/s and ft/s -> m/s
mps2ftps = 3.28084
ftps2mps = 1 / mps2ftps
