"""
Name: Chengjun Lu, Victoria Liu, Yudi Wu, Eve Zhang
Date: 10/22/2026
"""

import math

kB = 1.380649e-23   # Boltzmann constant
NA = 6.02214076e23  # Avogadro constant

def radius_from_molecular_weight(mw: float, density=1.3) -> float:
    """
    Estimate the radius of a molecule from its molecular weight.

    V = MW / (rho * N_A)
    r = (3 * V / (4 * pi)) ** (1/3)

    Args: 
        mw (float): molecular weight in g/mol
        density (float): density of the molecule
    
    Returns: 
        float: the estimated molecular radius in meters. 
    """
    rho = density * 1e6
    vol = mw / (rho * NA)
    r = (3 * vol / (4 * math.pi)) ** (1 / 3)
    return r


def stokes_einstein_mm2_s(mw: float, mu: float, T=310.15, radius_nm=None) -> float:
    """
    Compute the diffusion coefficient of a molecule.

    D = k_B * T / (6 * pi * mu * r)

    Args:
        mw (float): molecular weight in g/mol
        mu (float): viscosity 
        T (float): temperature in kelvin
        radius_nm (float): diffusion coefficient
    """
    if radius_nm is not None:
        r = radius_nm * 1e-9
    else:
        r = radius_from_molecular_weight(mw)

    D = kB * T / (6 * math.pi * mu * r)
    return D * 1e6