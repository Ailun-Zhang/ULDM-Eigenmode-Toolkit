"""PyUL code-unit conversions (G = hbar/m_a = 1), identical to PyUL_SK Current_SK.py.

The axion mass is read from the environment variable ``PYUL_AXION_MASS_EV``
**when this module is first imported** (default 1e-22 eV).  The paper runs
used 1e-21 eV.  If the variable is not set, the first conversion emits one
warning.  :func:`set_axion_mass` changes the mass afterwards (all
conversion functions of this module use the new value; names copied
earlier with ``from units import *`` are not updated).
"""
import math
import os
import warnings


def _warn_if_default_axion_mass():
    global _default_mass_warned
    if _AXION_MASS_SOURCE == "default" and not _default_mass_warned:
        _default_mass_warned = True
        warnings.warn(
            "PYUL_AXION_MASS_EV is not set, so Eig_ULDM_packages.units uses the default axion mass "
            f"{axion_E:g} eV. The Stone Skipping paper runs used 1e-21 eV. Set PYUL_AXION_MASS_EV "
            "before the first import of this module, or call units.set_axion_mass(value_in_eV).",
            UserWarning,
            stacklevel=3,
        )


def convert(value, unit, type):
    _warn_if_default_axion_mass()
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
    _warn_if_default_axion_mass()
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
            # Inverse of convert(value, 'c', 'v') (fixed in 0.3.0; it used
            # value * time_unit / length_unit / c before).
            converted = value / time_unit * length_unit / 299792458
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
_AXION_MASS_SOURCE = "env" if os.environ.get("PYUL_AXION_MASS_EV", "").strip() else "default"
_default_mass_warned = False
axion_E = float(os.environ.get("PYUL_AXION_MASS_EV", "").strip() or "1e-22")
eV = 1.78266191e-36  # kg
axion_mass = axion_E * eV

# unit of measure
CritDens = 3 * H_0 ** 2 / (8 * pi * G)

time_unit = (3 * H_0 ** 2 * omega_m0 / (8 * pi)) ** -0.5
length_unit = (8 * pi * hbar ** 2 / (3 * axion_mass ** 2 * H_0 ** 2 * omega_m0)) ** 0.25
mass_unit = (3 * H_0 ** 2 * omega_m0 / (8 * pi)) ** 0.25 * hbar ** 1.5 / (axion_mass ** 1.5 * G)
energy_unit = mass_unit * length_unit ** 2 / (time_unit**2)


def set_axion_mass(axion_mass_ev):
    """Set the axion mass (in eV) used by all conversions of this module.

    Recomputes ``axion_E``, ``axion_mass``, ``length_unit``, ``mass_unit``
    and ``energy_unit`` with exactly the import-time formulas, so
    ``set_axion_mass(1e-21)`` gives bitwise the same units as importing with
    ``PYUL_AXION_MASS_EV=1e-21``.  ``time_unit`` does not depend on the mass.
    """
    global axion_E, axion_mass, length_unit, mass_unit, energy_unit, _AXION_MASS_SOURCE
    value = float(axion_mass_ev)
    if not (value > 0 and math.isfinite(value)):
        raise ValueError(f"axion mass must be a positive finite number of eV, got {axion_mass_ev!r}")
    axion_E = value
    axion_mass = axion_E * eV
    length_unit = (8 * pi * hbar ** 2 / (3 * axion_mass ** 2 * H_0 ** 2 * omega_m0)) ** 0.25
    mass_unit = (3 * H_0 ** 2 * omega_m0 / (8 * pi)) ** 0.25 * hbar ** 1.5 / (axion_mass ** 1.5 * G)
    energy_unit = mass_unit * length_unit ** 2 / (time_unit**2)
    _AXION_MASS_SOURCE = "set_axion_mass"
    return axion_E


def get_axion_mass_ev():
    """Axion mass (eV) currently used by this module."""
    return axion_E


def axion_mass_source():
    """Where the axion mass came from: 'env', 'default' or 'set_axion_mass'."""
    return _AXION_MASS_SOURCE

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