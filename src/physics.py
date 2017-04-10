import numpy as np


def cyclotron_frequency(magnetic_field, charge=1, element=1):
    proton_charge = -1.6 * 10 ** -19
    proton_mass = 1.67 * 10 ** -27
    coef = (charge * proton_charge) / (2 * np.pi * element * proton_mass)
    cyclotron_data = np.abs(coef * (magnetic_field * 10 ** -9))
    return cyclotron_data
