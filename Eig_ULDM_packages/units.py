import math
import os
def convert(value, unit, type):
    converted = 0
    if type == 'l':
        if unit == '':
            converted = value
        elif unit == 'm' or unit == 'SI':
            converted = value / length_unit
        elif unit == 'km':
            converted = value * 1e3 / length_unit
        elif unit == 'pc':
            converted = value * parsec / length_unit
        elif unit == 'kpc':
            converted = value * 1e3 * parsec / length_unit
        elif unit == 'Mpc':
            converted = value * 1e6 * parsec / length_unit
        elif unit == 'ly':
            converted = value * light_year / length_unit
        else:
            raise NameError('Unsupported LENGTH unit used')

    elif type == 'm':
        if unit == '':
            converted = value
        elif unit == 'kg' or unit == 'SI':
            converted = value / mass_unit
        elif unit == 'solar_masses':
            converted = value * solar_mass / mass_unit
        elif unit == 'M_solar_masses':
            converted = value * solar_mass * 1e6 / mass_unit
        else:
            raise NameError('Unsupported MASS unit used')

    elif type == 't':
        if unit == '':
            converted = value
        elif unit == 's' or unit == 'SI':
            converted = value / time_unit
        elif unit == 'yr':
            converted = value * 60 * 60 * 24 * 365 / time_unit
        elif unit == 'kyr':
            converted = value * 60 * 60 * 24 * 365 * 1e3 / time_unit
        elif unit == 'Myr':
            converted = value * 60 * 60 * 24 * 365 * 1e6 / time_unit
        elif unit == 'Gyr':
            converted = value * 60 * 60 * 24 * 365 * 1e9 / time_unit
        else:
            raise NameError('Unsupported TIME unit used')

    elif type == 'v':
        if unit == '':
            converted = value
        elif unit == 'm/s' or unit == 'SI':
            converted = value * time_unit / length_unit
        elif unit == 'km/s':
            converted = value * 1e3 * time_unit / length_unit
        elif unit == 'km/h':
            converted = value * 1e3 / (60 * 60) * time_unit / length_unit
        elif unit == 'c':
            converted = value * time_unit / length_unit * 299792458
        else:
            raise NameError('Unsupported SPEED unit used')

    elif type == 'd':
        if unit == '':
            converted = value
        elif unit == 'Crit':
            converted = value / omega_m0
        elif unit == 'MSol/pc3':
            converted = value * solar_mass / mass_unit * length_unit**3 / parsec**3
        elif unit == 'MMSol/kpc3':
            converted = value * solar_mass / mass_unit * length_unit**3 / parsec**3 / 1000
        elif unit == 'kg/m3' or unit == 'SI':
            converted = value / mass_unit * length_unit**3
        else:
            raise NameError('Unsupported DENSITY unit used')

    elif type == 'a':
        if unit == '':
            converted = value
        elif unit == 'm/s2' or unit == 'SI':
            converted = value / length_unit * time_unit**2
        else:
            raise NameError('Unsupported ACCELERATION unit used')
    
    elif type == 'p':
        if unit == '':
            converted = value
        elif unit == 'm^2/s^2' or unit == 'SI':
            converted = value / length_unit**2 * time_unit**2
        else:
            raise NameError('Unsupported POTENTIAL unit used')
    else:
        raise TypeError('Unsupported conversion type')

    return converted


def convert_back(value, unit, type):
    converted = 0
    if type == 'l':
        if unit == '':
            converted = value
        elif unit == 'm' or unit == 'SI':
            converted = value * length_unit
        elif unit == 'km':
            converted = value / 1e3 * length_unit
        elif unit == 'pc':
            converted = value / parsec * length_unit
        elif unit == 'kpc':
            converted = value / (1e3 * parsec) * length_unit
        elif unit == 'Mpc':
            converted = value / (1e6 * parsec) * length_unit
        elif unit == 'ly':
            converted = value / light_year * length_unit
        else:
            raise NameError('Unsupported LENGTH unit used')

    elif type == 'm':
        if unit == '':
            converted = value
        elif unit == 'kg' or unit == 'SI':
            converted = value * mass_unit
        elif unit == 'solar_masses':
            converted = value / solar_mass * mass_unit
        elif unit == 'M_solar_masses':
            converted = value / (solar_mass * 1e6) * mass_unit
        else:
            raise NameError('Unsupported MASS unit used')

    elif type == 't':
        if unit == '':
            converted = value
        elif unit == 's' or unit == 'SI':
            converted = value * time_unit
        elif unit == 'yr':
            converted = value / (60 * 60 * 24 * 365) * time_unit
        elif unit == 'kyr':
            converted = value / (60 * 60 * 24 * 365 * 1e3) * time_unit
        elif unit == 'Myr':
            converted = value / (60 * 60 * 24 * 365 * 1e6) * time_unit
        elif unit == 'Gyr':
            converted = value / (60 * 60 * 24 * 365 * 1e9) * time_unit
        else:
            raise NameError('Unsupported TIME unit used')

    elif type == 'v':
        if unit == '':
            converted = value
        elif unit == 'm/s' or unit == 'SI':
            converted = value / time_unit * length_unit
        elif unit == 'km/s':
            converted = value / (1e3) / time_unit * length_unit
        elif unit == 'km/h':
            converted = value / (1e3) * (60 * 60) / time_unit * length_unit
        elif unit == 'c':
            converted = value * time_unit / length_unit / 299792458
        else:
            raise NameError('Unsupported SPEED unit used')

    elif type == 'd':
        if unit == '':
            converted = value
        elif unit == 'Crit':
            converted = value * omega_m0
        elif unit == 'MSol/pc3':
            converted = value / solar_mass * mass_unit / length_unit**3 * parsec**3
        elif unit == 'MMSol/kpc3':
            converted = value / solar_mass * mass_unit / length_unit**3 * parsec**3 * 1000
        elif unit == 'kg/m3' or unit == 'SI':
            converted = value * mass_unit / length_unit**3
        else:
            raise NameError('Unsupported DENSITY unit used')

    elif type == 'a':
        if unit == '':
            converted = value
        elif unit == 'm/s2' or unit == 'SI':
            converted = value * length_unit / time_unit**2
        else:
            raise NameError('Unsupported ACCELERATION unit used')
    
    elif type == 'p':
        if unit == '':
            converted = value
        elif unit == 'm^2/s^2' or unit == 'SI':
            converted = value * length_unit**2 / time_unit**2
        else:
            raise NameError('Unsupported POTENTIAL unit used')

    else:
        raise TypeError('Unsupported conversion type')

    return converted

def convert_between(value, oldunit, newunit, type):
    return convert_back(convert(value, oldunit, type), newunit, type)

# Constant Definitions
hbar = 1.0545718e-34  # m^2 kg/s
parsec = 3.0857e16  # m
light_year = 9.4607e15  # m
solar_mass = 1.989e30  # kg
G = 6.67e-11  # m^3 kg^-1 s^-2
omega_m0 = 0.31
H_0 = 67.7 / (parsec * 1e3)  # s^-1
pi = math.pi
#axion_E = 1e-21  # Example value in eV, replace with actual value
axion_E = float(os.environ.get("PYUL_AXION_MASS_EV", "1e-22"))
eV = 1.78266191e-36  # kg
axion_mass = axion_E * eV

# unit of measure
CritDens = 3 * H_0 ** 2 / (8 * pi * G)

time_unit = (3 * H_0 ** 2 * omega_m0 / (8 * pi)) ** -0.5
length_unit = (8 * pi * hbar ** 2 / (3 * axion_mass ** 2 * H_0 ** 2 * omega_m0)) ** 0.25
mass_unit = (3 * H_0 ** 2 * omega_m0 / (8 * pi)) ** 0.25 * hbar ** 1.5 / (axion_mass ** 1.5 * G)
energy_unit = mass_unit * length_unit ** 2 / (time_unit**2)

# Example usage
'''
density_dimless = 100
density_SI = convert_back(density_dimless, 'kg/m3', 'd')
print(f"Density in SI units (kg/m^3): {density_SI}")

potential_dimless = -300
potential_SI = convert_back(potential_dimless, 'm^2/s^2', 'p')
print(f"Gravitational Potential in SI units (m^2/s^2): {potential_SI}")

potential_SI = -9.8e8  # Example value in m^2/s^2
potential_dimless = convert(potential_SI, 'm^2/s^2', 'p')
print(f"Gravitational Potential in dimensionless units: {potential_dimless}")
'''