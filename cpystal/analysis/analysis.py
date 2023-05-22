"""`cpystal.analysis` is a module for theoretical calculations and making some useful data files for numerical analysis.

Functions:
    `brillouin`
        -Brillouin function B_J(x).
    `paramagnetization_curie`
        -Magnetization from Curie paramagnetism.
    `fit_paramagnetism`
        -Fitting magnetic field dependence of magnetic moment to theoretical paramagnetism.
    `demagnetizing_factor_ellipsoid`
        -Calculating demagnetizing factor of ellipsoid 2a x 2b x 2c.
    `demagnetizing_factor_rectangular_prism`
        -Calculating demagnetizing factor of rectangular prism axbxc.

"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit # type: ignore
from scipy import integrate # type: ignore

from ..core import Crystal



def brillouin(x: float, J: float) -> float:
    """Brillouin function B_J(x).

    Args:
        x (float): Real parameter.
        J (float): Integer of half integer (corresponds to total momentum quantum number).
    
    Returns:
        (float): B_J(x).
    """
    return (2*J+1) / (2*J) / np.tanh(x*(2*J+1)/(2*J)) - 1 / (2*J) / np.tanh(x/(2*J))

def paramagnetization_curie(H: float, T: float, g: float, J: float, n: int) -> float:
    """Magnetization from Curie paramagnetism.

    Note:
        M = n g J B_J(g muB J H/kB T) [muB/f.u.],
        where
            n = number of magnetic atom per formula unit,
            g = g factor,
            J = total angular momentum quantum number,
            B_J = Brillouin function,
            kB = Boltzmann constant (J/K),
            H = magnetic field (Oe),
            T = temperature (K).
    
    Args:
        H (float): Magnetic field (Oe).
        T (float): Temperature (K).
        g (float): g-factor.
        J (float): Total angular momentum quantum number.
        n (float): Number of magnetic atom per formula unit (1/f.u.).
    
    Returns:
        (float): Magnetization from Curie paramagnetism (muB/f.u.).
    """
    muB: float = 9.27401e-21 * 1.0e-7 # Bohr磁子 [emu = erg/Oe = 10^(-7) J/Oe]
    kB: float = 1.380649e-23 # Boltzmann定数 [J/K]
    return n * g * J * brillouin(g*J*muB*H/(kB*T), J)

def fit_paramagnetism(material: Crystal, H: list[float], moment: list[float], T: float) -> tuple[float, float]:
    """Fitting magnetic field dependence of magnetic moment to theoretical paramagnetism.

    Note:
        M = n g J B_J(g muB J H/kB T) [muB/f.u.],
        where
            n = number of magnetic atom per formula unit,
            g = g factor,
            J = total angular momentum quantum number,
            B_J = Brillouin function,
            kB = Boltzmann constant,
            H = magnetic field,
            T = temperature.
    Args:
        material (Crystal): Crystal instance.
        H (list[float]): Magnetic field (Oe).
        moment (list[float]): Magnetic moment (emu).

    Returns:
        (tuple[float, float]): g and J.
    """
    n: int = material.num_magnetic_ion
    magnetization = lambda h, g, J: paramagnetization_curie(h, T, g, J, n)
    popt, pcov = curve_fit(magnetization, np.array(H), moment)
    return popt

def demagnetizing_factor_ellipsoid(a: float, b: float, c: float) -> tuple[float, float, float]:
    """Calculating demagnetizing factor of ellipsoid 2a x 2b x 2c.

    Args:
        a (float): Length of an edge (arb. unit).
        b (float): Length of an edge (arb. unit).
        c (float): Length of an edge (arb. unit).
    
    Returns:
        (tuple[float]): Demagnetizing factor Nx, Ny, Nz.
    """
    a, b, c = a/(a+b+c), b/(a+b+c), c/(a+b+c)
    def D(u: float) -> float:
        return (a**2+u) * (b**2+u) * (c**2+u)
    
    def fx(u: float) -> float:
        return 1 / ((a**2+u) * np.sqrt(D(u)))
    def fy(u: float) -> float:
        return 1 / ((b**2+u) * np.sqrt(D(u)))
    def fz(u: float) -> float:
        return 1 / ((c**2+u) * np.sqrt(D(u)))
    
    Nx: float = a*b*c/2 * integrate.quad(fx, 0, np.inf)[0]
    Ny: float = a*b*c/2 * integrate.quad(fy, 0, np.inf)[0]
    Nz: float = a*b*c/2 * integrate.quad(fz, 0, np.inf)[0]
    return Nx, Ny, Nz

def demagnetizing_factor_rectangular_prism(a: float, b: float, c: float) -> float:
    """Calculating demagnetizing factor of rectangular prism axbxc.

    Thesis:
        A. Aharoni et al., Journal of Applied Physics 83, 3432 (1998).
        (See also: http://www.magpar.net/static/magpar/doc/html/demagcalc.html)

    Args:
        a (float): Length of an edge (arb. unit).
        b (float): Length of an edge (arb. unit).
        c (float): Length of an edge (arb. unit).
    
    Returns:
        (float): Demagnetizing factor.
    """
    abc_root: float = np.sqrt(a**2 + b**2 + c**2)
    ab_root: float = np.sqrt(a**2 + b**2)
    bc_root: float = np.sqrt(b**2 + c**2)
    ca_root: float = np.sqrt(a**2 + c**2)

    F1: float = (b**2-c**2) / (2*b*c) * np.log((abc_root-a) / (abc_root+a))
    F2: float = (a**2-c**2) / (2*a*c) * np.log((abc_root-b) / (abc_root+b))
    F3: float = b / (2*c) * np.log((ab_root+a) / (ab_root-a))
    F4: float = a / (2*c) * np.log((ab_root+b) / (ab_root-b))
    F5: float = c / (2*a) * np.log((bc_root-b) / (bc_root+b))
    F6: float = c / (2*b) * np.log((ca_root-a) / (ca_root+a))
    F7: float = 2 * np.arctan(a*b/(c*abc_root))
    F8: float = (a**3 + b**3 - 2*c**3) / (3 * a * b * c)
    F9: float = (a**2 + b**2 - 2*c**2) / (3 * a * b * c) * abc_root
    F10: float = c / (a*b) * (ca_root + bc_root)
    F11: float = - (ab_root**3 + bc_root**3 + ca_root**3) / (3 * a * b * c)
    Dz: float = (F1 + F2 + F3 + F4 + F5 + F6 + F7 + F8 + F9 + F10 + F11) / np.pi
    return Dz


def main() -> None:
    pass
    return

if __name__ == "__main__":
    main()

