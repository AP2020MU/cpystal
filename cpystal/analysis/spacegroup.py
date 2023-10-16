"""`cpystal.analysis.spacegroup` is a module for analyzing space group of crystal.

Classes:
    `REF`
    `MatrixREF`
    `SymmetryOperation`
    `PhysicalPropertyTensorAnalyzer`
    `Atom`
    `SpaceGroupSymmetryOperation`
    `CrystalStructure`
    `SpinStructure`

Functions:
    `spacegroup_to_pointgroup`
    `pointgroup_to_laueclass`
    `pointgroup_to_crystal_system`
    `spacegroup_to_bravais_lattice`
    `is_polar_point_group`
    `is_polar_space_group`
    `is_chiral_point_group`
    `is_chiral_space_group`
    `is_centrosymmetric_point_group`
    `is_centrosymmetric_space_group`
    `is_enantiomorphic_space_group`
    `is_sohncke_space_group`
    `is_symmorphic_space_group`
    `generate_point_group`
    `crystal_system_to_symmetry_directions`
    `spacegroup_to_symmetry_directions`
    `unit_vector`
    `circular_mean`
    `circular_diff`
"""
from __future__ import annotations

from collections import defaultdict, deque
from fractions import Fraction
from functools import reduce
from itertools import combinations, product
from math import gcd
import os
import re
from typing import Any, List, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

from ..mathematics.algebra.core import REF, MatrixREF


SPACE_GROUPS: list[str] = [
    "P1",
    "P-1",
    "P2",
    "P2_1",
    "C2",
    "Pm",
    "Pc",
    "Cm",
    "Cc",
    "P2/m",
    "P2_1/m",
    "C2/m",
    "P2/c",
    "P2_1/c",
    "C2/c",
    "P222",
    "P222_1",
    "P2_12_12",
    "P2_12_12_1",
    "C222_1",
    "C222",
    "F222",
    "I222",
    "I2_12_12_1",
    "Pmm2",
    "Pmc2_1",
    "Pcc2",
    "Pma2",
    "Pca2_1",
    "Pnc2",
    "Pmn2_1",
    "Pba2",
    "Pna2_1",
    "Pnn2",
    "Cmm2",
    "Cmc2_1",
    "Ccc2",
    "Amm2",
    "Aem2",
    "Ama2",
    "Aea2",
    "Fmm2",
    "Fdd2",
    "Imm2",
    "Iba2",
    "Ima2",
    "Pmmm",
    "Pnnn",
    "Pccm",
    "Pban",
    "Pmma",
    "Pnna",
    "Pmna",
    "Pcca",
    "Pbam",
    "Pccn",
    "Pbcm",
    "Pnnm",
    "Pmmn",
    "Pbcn",
    "Pbca",
    "Pnma",
    "Cmcm",
    "Cmce",
    "Cmmm",
    "Cccm",
    "Cmme",
    "Ccce",
    "Fmmm",
    "Fddd",
    "Immm",
    "Ibam",
    "Ibca",
    "Imma",
    "P4",
    "P4_1",
    "P4_2",
    "P4_3",
    "I4",
    "I4_1",
    "P-4",
    "I-4",
    "P4/m",
    "P4_2/m",
    "P4/n",
    "P4_2/n",
    "I4/m",
    "I4_1/a",
    "P422",
    "P42_12",
    "P4_122",
    "P4_12_12",
    "P4_222",
    "P4_22_12",
    "P4_322",
    "P4_32_12",
    "I422",
    "I4_122",
    "P4mm",
    "P4bm",
    "P4_2cm",
    "P4_2nm",
    "P4cc",
    "P4nc",
    "P4_2mc",
    "P4_2bc",
    "I4mm",
    "I4cm",
    "I4_1md",
    "I4_1cd",
    "P-42m",
    "P-42c",
    "P-42_1m",
    "P-42_1c",
    "P-4m2",
    "P-4c2",
    "P-4b2",
    "P-4n2",
    "I-4m2",
    "I-4c2",
    "I-42m",
    "I-42d",
    "P4/mmm",
    "P4/mcc",
    "P4/nbm",
    "P4/nnc",
    "P4/mbm",
    "P4/mnc",
    "P4/nmm",
    "P4/ncc",
    "P4_2/mmc",
    "P4_2/mcm",
    "P4_2/nbc",
    "P4_2/nnm",
    "P4_2/mbc",
    "P4_2/mnm",
    "P4_2/nmc",
    "P4_2/ncm",
    "I4/mmm",
    "I4/mcm",
    "I4_1/amd",
    "I4_1/acd",
    "P3",
    "P3_1",
    "P3_2",
    "R3",
    "P-3",
    "R-3",
    "P312",
    "P321",
    "P3_112",
    "P3_121",
    "P3_212",
    "P3_221",
    "R32",
    "P3m1",
    "P31m",
    "P3c1",
    "P31c",
    "R3m",
    "R3c",
    "P-31m",
    "P-31c",
    "P-3m1",
    "P-3c1",
    "R-3m",
    "R-3c",
    "P6",
    "P6_1",
    "P6_5",
    "P6_2",
    "P6_4",
    "P6_3",
    "P-6",
    "P6/m",
    "P6_3/m",
    "P622",
    "P6_122",
    "P6_522",
    "P6_222",
    "P6_422",
    "P6_322",
    "P6mm",
    "P6cc",
    "P6_3cm",
    "P6_3mc",
    "P-6m2",
    "P-6c2",
    "P-62m",
    "P-62c",
    "P6/mmm",
    "P6/mcc",
    "P6_3/mcm",
    "P6_3/mmc",
    "P23",
    "F23",
    "I23",
    "P2_13",
    "I2_13",
    "Pm-3",
    "Pn-3",
    "Fm-3",
    "Fd-3",
    "Im-3",
    "Pa-3",
    "Ia-3",
    "P432",
    "P4_232",
    "F432",
    "F4_132",
    "I432",
    "P4_332",
    "P4_132",
    "I4_132",
    "P-43m",
    "F-43m",
    "I-43m",
    "P-43n",
    "F-43c",
    "I-43d",
    "Pm-3m",
    "Pn-3n",
    "Pm-3n",
    "Pn-3m",
    "Fm-3m",
    "Fm-3c",
    "Fd-3m",
    "Fd-3c",
    "Im-3m",
    "Ia-3d",
]

SPACE_GROUPS_FULL_SYMBOL: list[str] = [
    "P1",
    "P-1",
    "P121", "P112",
    "P12_11", "P112_1",
    "C121", "A121",
    "P1m1", "P11m",
    "P1c1", "P11a",
    "C1m1", "A11m",
    "C1c1", "A11a",
    "P12/m1", "P112/m",
    "P12_1/m1", "P112_1/m",
    "C12/m1", "A112/m",
    "P12/c1", "P112/a",
    "P12_1/c1", "P112_1/a",
    "C12/c1", "C112/a",
    "P222",
    "P222_1",
    "P2_12_12",
    "P2_12_12_1",
    "C222_1",
    "C222",
    "F222",
    "I222",
    "I2_12_12_1",
    "Pmm2",
    "Pmc2_1",
    "Pcc2",
    "Pma2",
    "Pca2_1",
    "Pnc2",
    "Pmn2_1",
    "Pba2",
    "Pna2_1",
    "Pnn2",
    "Cmm2",
    "Cmc2_1",
    "Ccc2",
    "Amm2",
    "Aem2",
    "Ama2",
    "Aea2",
    "Fmm2",
    "Fdd2",
    "Imm2",
    "Iba2",
    "Ima2",
    "P2/m2/m2/m",
    "P2/n2/n2/n",
    "P2/c2/c2/m",
    "P2/b2/a2/n",
    "P2_1/m2/m2/a",
    "P2/n2_1/n2/a",
    "P2/m2/n2_1/a",
    "P2_1/c2/c2/a",
    "P2_1/b2_1/a2/m",
    "P2_1/c2_1/c2/n",
    "P2_1/b2_1/c2_1/m",
    "P2_1/n2_1/n2/m",
    "P2_1/m2_1/m2/n",
    "P2_1/b2/c2_1/n",
    "P2_1/b2_1/c2_1/a",
    "P2_1/n2_1/m2_1/a",
    "C2/m2/c2_1/m",
    "C2/m2/c2_1/e",
    "C2/m2/m2/m",
    "C2/c2/c2/m",
    "C2/m2/m2/e",
    "C2/c2/c2/e",
    "F2/m2/m2/m",
    "F2/d2/d2/d",
    "I2/m2/m2/m",
    "I2/b2/a2/m",
    "I2_1/b2_1/c2_1/a",
    "I2_1/m2_1/m2_1/a",
    "P4",
    "P4_1",
    "P4_2",
    "P4_3",
    "I4",
    "I4_1",
    "P-4",
    "I-4",
    "P4/m",
    "P4_2/m",
    "P4/n",
    "P4_2/n",
    "I4/m",
    "I4_1/a",
    "P422",
    "P42_12",
    "P4_122",
    "P4_12_12",
    "P4_222",
    "P4_22_12",
    "P4_322",
    "P4_32_12",
    "I422",
    "I4_122",
    "P4mm",
    "P4bm",
    "P4_2cm",
    "P4_2nm",
    "P4cc",
    "P4nc",
    "P4_2mc",
    "P4_2bc",
    "I4mm",
    "I4cm",
    "I4_1md",
    "I4_1cd",
    "P-42m",
    "P-42c",
    "P-42_1m",
    "P-42_1c",
    "P-4m2",
    "P-4c2",
    "P-4b2",
    "P-4n2",
    "I-4m2",
    "I-4c2",
    "I-42m",
    "I-42d",
    "P4/m2/m2/m",
    "P4/m2/c2/c",
    "P4/n2/b2/m",
    "P4/n2/n2/c",
    "P4/m2_1/b2/m",
    "P4/m2_1/n2/c",
    "P4/n2_1/m2/m",
    "P4/n2/c2/c",
    "P4_2/m2/m2/c",
    "P4_2/m2/c2/m",
    "P4_2/n2/b2/c",
    "P4_2/n2/n2/m",
    "P4_2/m2_1/b2/c",
    "P4_2/m2_1/n2/m",
    "P4_2/n2_1/m2/c",
    "P4_2/n2_1/c2/m",
    "I4/m2/m2/m",
    "I4/m2/c2/m",
    "I4_1/a2/m2/d",
    "I4_1/a2/c2/d",
    "P3",
    "P3_1",
    "P3_2",
    "R3",
    "P-3",
    "R-3",
    "P312",
    "P321",
    "P3_112",
    "P3_121",
    "P3_212",
    "P3_221",
    "R32",
    "P3m1",
    "P31m",
    "P3c1",
    "P31c",
    "R3m",
    "R3c",
    "P-312/m",
    "P-312/c",
    "P-32/m1",
    "P-32/c1",
    "R-32/m",
    "R-32/c",
    "P6",
    "P6_1",
    "P6_5",
    "P6_2",
    "P6_4",
    "P6_3",
    "P-6",
    "P6/m",
    "P6_3/m",
    "P622",
    "P6_122",
    "P6_522",
    "P6_222",
    "P6_422",
    "P6_322",
    "P6mm",
    "P6cc",
    "P6_3cm",
    "P6_3mc",
    "P-6m2",
    "P-6c2",
    "P-62m",
    "P-62c",
    "P6/m2/m2/m",
    "P6/m2/c2/c",
    "P6_3/m2/c2/m",
    "P6_3/m2/m2/c",
    "P23",
    "F23",
    "I23",
    "P2_13",
    "I2_13",
    "P2/m-3",
    "P2/n-3",
    "F2/m-3",
    "F2/d-3",
    "I2/m-3",
    "P2_1/a-3",
    "I2_1/a-3",
    "P432",
    "P4_232",
    "F432",
    "F4_132",
    "I432",
    "P4_332",
    "P4_132",
    "I4_132",
    "P-43m",
    "F-43m",
    "I-43m",
    "P-43n",
    "F-43c",
    "I-43d",
    "P4/m-32/m",
    "P4/n-32/n",
    "P4_2/m-32/n",
    "P4_2/n-32/m",
    "F4/m-32/m",
    "F4/m-32/c",
    "F4_1/d-32/m",
    "F4_1/d-32/c",
    "I4/m-32/m",
    "I4_1/a-32/d",
]

POINT_GROUPS: list[str] = [
    "1",
    "-1",
    "2",
    "m",
    "2/m",
    "222",
    "mm2",
    "mmm",
    "4",
    "-4",
    "4/m",
    "422",
    "4mm",
    "-42m",
    "4/mmm",
    "3",
    "-3",
    "32",
    "3m",
    "-3m",
    "6",
    "-6",
    "6/m",
    "622",
    "6mm",
    "-62m",
    "6/mmm",
    "23",
    "m-3",
    "432",
    "-43m",
    "m-3m",
]

POINT_GROUPS_FULL_SYMBOL: list[str] = [
    "1",
    "-1",
    "2",
    "m",
    "2/m",
    "222",
    "mm2",
    "2/m2/m2/m",
    "4",
    "-4",
    "4/m",
    "422",
    "4mm",
    "-42m",
    "4/m2/m2/m",
    "3",
    "-3",
    "32",
    "3m",
    "-32/m",
    "6",
    "-6",
    "6/m",
    "622",
    "6mm",
    "-62m",
    "6/m2/m2/m",
    "23",
    "2/m-3",
    "432",
    "-43m",
    "4/m-32/m",
]

AUGMENTED_POINT_GROUPS: list[str] = [
    "1",
    "-1",
    "2",
    "m",
    "2/m",
    "222",
    "mm2",
    "mmm",
    "4",
    "-4",
    "4/m",
    "422",
    "4mm",
    "-42m",
    "-4m2", #
    "4/mmm",
    "3",
    "-3",
    "32",
    "312", #
    "321", #
    "3m",
    "3m1", #
    "31m", #
    "-3m",
    "-3m1", #
    "-31m", #
    "6",
    "-6",
    "6/m",
    "622",
    "6mm",
    "-62m",
    "-6m2", #
    "6/mmm",
    "23",
    "m-3",
    "432",
    "-43m",
    "m-3m",
]

ARITHMETIC_CLASSES: list[str] = [
    "1P",
    "-1P",
    "2P",
    "2C",
    "mP",
    "mC",
    "2/mP",
    "2/mC",
    "222P",
    "222C",
    "222F",
    "222I",
    "mm2P",
    "mm2C",
    "mm2A",
    "mm2F",
    "mm2I",
    "mmmP",
    "mmmC",
    "mmmF",
    "mmmI",
    "4P",
    "4I",
    "-4P",
    "-4I",
    "4/mP",
    "4/mI",
    "422P",
    "422I",
    "4mmP",
    "4mmI",
    "-42mP",
    "-4m2P",
    "-4m2I",
    "-42mI",
    "4/mmmP",
    "4/mmmI",
    "3P",
    "3R",
    "-3P",
    "-3R",
    "312P",
    "321P",
    "32R",
    "3m1P",
    "31mP",
    "3mR",
    "-31mP",
    "-3m1P",
    "-3mR",
    "6P",
    "-6P",
    "6/mP",
    "622P",
    "6mmP",
    "-6m2P",
    "-62mP",
    "6/mmmP",
    "23P",
    "23F",
    "23I",
    "m-3P",
    "m-3F",
    "m-3I",
    "432P",
    "432F",
    "432I",
    "-43mP",
    "-43mF",
    "-43mI",
    "m-3mP",
    "m-3mF",
    "m-3mI",
]

GEOMETIC_CLASSES: list[str] = [
    "1",
    "-1",
    "2",
    "m",
    "2/m",
    "222",
    "mm2",
    "mmm",
    "4",
    "-4",
    "4/m",
    "422",
    "4mm",
    "-42m",
    "4/mmm",
    "3",
    "-3",
    "32",
    "3m",
    "-3m",
    "6",
    "-6",
    "6/m",
    "622",
    "6mm",
    "-62m",
    "6/mmm",
    "23",
    "m-3",
    "432",
    "-43m",
    "m-3m",
]

LAUE_CLASSES: list[str] = [
    "-1",
    "2/m",
    "mmm",
    "4/m",
    "4/mmm",
    "-3",
    "-3m",
    "6/m",
    "6/mmm",
    "m3",
    "m3m",
]

HOLOHEDRY: list[str] = [
    "-1",
    "2/m",
    "mmm",
    "4/mmm",
    "-3m",
    "6/mmm",
    "m3m",
]

CRYSTAL_SYSTEMS: list[str] = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]

LATTICE_SYSTEMS: list[str] = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "rhombohedral",
    "hexagonal",
    "cubic",
]

CRYSTAL_FAMILIES: list[str] = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "hexagonal",
    "cubic",
]

BRAVAIS_LATTICE: list[str] = [
    "aP",
    "mP",
    "mS",
    "oP",
    "oS",
    "oI",
    "oF",
    "tP",
    "tI",
    "hR",
    "hP",
    "cP",
    "cI",
    "cF",
]


def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)

def rotx(theta: float) -> npt.NDArray:
    return np.array([[1, 0, 0],
                   [0, np.cos(theta), np.sin(theta)],
                   [0, -np.sin(theta), np.cos(theta)]])

def roty(theta: float) -> npt.NDArray:
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]])

def rotz(theta: float) -> npt.NDArray:
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])

def rotn(n: npt.NDArray, theta: float) -> npt.NDArray:
    c: float = np.cos(theta)
    s: float = np.sin(theta)
    n = n / np.linalg.norm(n)
    nx, ny, nz = n
    return np.array(
        [
            [nx**2*(1-c)+c, nx*ny*(1-c)-nz*s, nx*nz*(1-c)+ny*s],
            [nx*ny*(1-c)+nz*s, ny**2*(1-c)+c, ny*nz*(1-c)-nx*s],
            [nx*nz*(1-c)-ny*s, ny*nz*(1-c)+nx*s, nz**2*(1-c)+c],
        ]
    )
    return c * np.eye(3) + (1-c) * np.outer(n, n) + s * np.array(
        [   [0, -nz, ny],
            [nz, 0, -nx],
            [-ny, nx, 0],]
    )

def rodrigues(r: npt.NDArray, theta: float, n: npt.NDArray) -> npt.NDArray:
    """Rotation formula of Rodrigues.
        Args:
            r (npt.NDArray): Target vector.
            theta (float): Angle.
            n (npt.NDArray): Axis of rotation.
    """
    n = n / np.linalg.norm(n)
    return r*np.cos(theta) + (1-np.cos(theta))*np.dot(n,r)*n + np.cross(n,r)*np.sin(theta)

def mirror(n: npt.NDArray) -> npt.NDArray:
    n = n / np.linalg.norm(n)
    return np.eye(3) - 2 * np.outer(n, n)

def unit_vector(theta: float, phi: float) -> npt.NDArray:
    x: float = np.sin(theta) * np.cos(phi)
    y: float = np.sin(theta) * np.sin(phi)
    z: float = np.cos(theta)
    return np.array([x, y, z]).T

def circular_mean(theta_1: float, theta_2: float) -> float:
    """(theta_1 + theta_2) / 2

    Args:
        theta_1 (float): angle.
        theta_2 (float): angle.
    
    Note:
        -pi <= (theta_1 + theta_2) / 2 <= pi

    Returns:
        float: (theta_1 + theta_2) / 2
    """
    return np.arctan2((np.sin(theta_1)+np.sin(theta_2))/2, (np.cos(theta_1)+np.cos(theta_2))/2)

def circular_diff(theta_1: float, theta_2: float) -> float:
    """theta_1 - theta_2

    Args:
        theta_1 (float): angle.
        theta_2 (float): angle.
    
    Note:
        -pi < theta_1 - theta_2 <= pi

    Returns:
        float: theta_1 - theta_2
    """
    angle: float = (theta_1 - theta_2) % (2*np.pi)
    if angle > np.pi:
        angle -= 2*np.pi
    return angle

def circular_interpolate(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    return circular_diff(y1, y0) / (x1-x0) * (x-x0) + y0


def spacegroup_to_pointgroup(name: str) -> str:
    """Convert space group name to point group name.

    1. Capital alphabets and spaces are removed.
    2. Subscripts (represented by "_" with numbers) are removed.
    3. All "1" except a component of monoclinic symbols ("1" or "-1") are removed.
    4. "a","b","c","d","e", and "n" are converted to "m"
    5. Exception handling.

    Args:
        name (str): Space group name in International (Hermann-Mauguin) notation.
    
    Returns:
        (str): Point group name in International (Hermann-Mauguin) notation converted from the space group name.
    
    Examples:
        >>> point_group_name = spacegroup_to_pointgroup("Fd-3m") # Diamond
        >>> print(point_group_name)
            m-3m
    """
    name = re.sub(r"[A-Z]|\s", "", name)
    name = re.sub(r"_\d|_\{\d\}", "", name)
    if name != "1" and name != "-1":
        name = re.sub(r"1", "", name)
    name = re.sub(r"[a-en]", "m", name)
    return name

def pointgroup_to_laueclass(point_group: str) -> bool:
    laue: dict[str, str] = {
        "1":    "-1",
        "-1":   "-1",
        "2":    "2/m",
        "m":    "2/m",
        "2/m":  "2/m",
        "222":  "mmm",
        "mm2":  "mmm",
        "mmm":  "mmm",
        "4":    "4/m",
        "-4":   "4/m",
        "4/m":  "4/m",
        "422":  "4/mmm",
        "4mm":  "4/mmm",
        "-42m": "4/mmm",
        "-4m2": "4/mmm", # equivalent to -42m
        "4/mmm": "4/mmm",
        "3":    "-3",
        "-3":   "-3",
        "32":   "-3m",
        "3m":   "-3m",
        "-3m":  "-3m",
        "6":    "6/m",
        "-6":   "6/m",
        "6/m":  "6/m",
        "622":  "6/mmm",
        "6mm":  "6/mmm",
        "-62m": "6/mmm",
        "-6m2": "6/mmm", # equivalent to -62m
        "6/mmm": "6/mmm",
        "23":   "m3",
        "m-3":  "m3",
        "432":  "m3m",
        "-43m": "m3m",
        "m-3m": "m3m",
    }
    return laue[point_group]

def pointgroup_to_crystal_system(pointgroup_name: str, monoclinic_unique_axis: str | None = None, trigonal_rhombohedral: bool = False) -> str:
    table: dict[str, str] = {
        "1":    "triclinic",
        "-1":   "triclinic",
        "2":    "monoclinic",
        "m":    "monoclinic",
        "2/m":  "monoclinic",
        "222":  "orthorhombic",
        "mm2":  "orthorhombic",
        "mmm":  "orthorhombic",
        "4":    "tetragonal",
        "-4":   "tetragonal",
        "4/m":  "tetragonal",
        "422":  "tetragonal",
        "4mm":  "tetragonal",
        "-42m": "tetragonal",
        "-4m2": "tetragonal", # equivalent to -42m
        "4/mmm": "tetragonal",
        "3":    "trigonal",
        "-3":   "trigonal",
        "32":   "trigonal",
        "3m":   "trigonal",
        "-3m":  "trigonal",
        "6":    "hexagonal",
        "-6":   "hexagonal",
        "6/m":  "hexagonal",
        "622":  "hexagonal",
        "6mm":  "hexagonal",
        "-62m": "hexagonal",
        "-6m2": "hexagonal", # equivalent to -62m
        "6/mmm": "hexagonal",
        "23":   "cubic",
        "m-3":  "cubic",
        "432":  "cubic",
        "-43m": "cubic",
        "m-3m": "cubic",
    }
    crystal_system: str = table[pointgroup_name]
    if crystal_system == "monoclinic":
        if monoclinic_unique_axis == "b":
            crystal_system = crystal_system + "-b"
        elif monoclinic_unique_axis == "c":
            crystal_system = crystal_system + "-c"
        else:
            pass
    if crystal_system == "trigonal" and trigonal_rhombohedral:
        crystal_system = "rhombohedral"
    return crystal_system

def crystal_system_to_symbol(crystal_system: str) -> str:
    symbol: dict[str, str] = {
        "triclinic": "a",
        "monoclinic": "m",
        "orthorhombic": "o",
        "tetragonal": "t",
        "trigonal": "h",
        "hexagonal": "h",
        "cubic": "c",
    }
    return symbol[crystal_system]

def spacegroup_to_bravais_lattice(space_group: str) -> str:
    amothc: str = crystal_system_to_symbol(pointgroup_to_crystal_system(spacegroup_to_pointgroup(space_group)))
    centering: str = space_group[0]
    if centering == "A" or centering == "B" or centering == "C":
        centering = "S"
    return amothc + centering

def is_polar_point_group(point_group: str) -> bool:
    return point_group in ["1", "2", "3", "4", "6", "m", "mm2", "3m", "4mm", "6mm"]

def is_polar_space_group(space_group: str) -> bool:
    return is_polar_point_group(spacegroup_to_pointgroup(space_group))

def is_chiral_point_group(point_group: str) -> bool:
    return point_group in ["1", "2", "3", "4", "6", "222", "422", "32", "622", "23", "432"]

def is_chiral_space_group(space_group: str) -> bool:
    chiral: list[str] = [
        "P4_1", "P4_3",
        "P4_122", "P4_322",
        "P4_12_12", "P4_32_12",
        "P3_1", "P3_2",
        "P3_121", "P3_221",
        "P3_112", "P3_212",
        "P6_1", "P6_5",
        "P6_2", "P6_4",
        "P6_122", "P6_522",
        "P6_222", "P6_422",
        "P4_132", "P4_332",
    ]
    return space_group in chiral

def is_achiral_space_group(space_group: str) -> bool:
    return not is_chiral_space_group(space_group)

def is_centrosymmetric_point_group(point_group: str) -> bool:
    return point_group in ["-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m3", "m-3m"]

def is_centrosymmetric_space_group(space_group: str) -> bool:
    return is_centrosymmetric_point_group(spacegroup_to_pointgroup(space_group))

def is_enantiomorphic_space_group(space_group: str) -> bool:
    return is_chiral_space_group(space_group)

def is_bieberbach_space_group(space_group: str) -> bool:
    """Judge whether the space group is Bieberbach space group or not.

    Args:
        space_group (str): Name of the space group.

    Returns:
        bool: Whether the space group is Bieberbach space group or not.
    """
    bieberbach: list[str] = [
        "P1",
        "P2_1",
        "Pc",
        "Cc",
        "P2_12_12_1",
        "Pca2_1",
        "Pna2_1",
        "P4_1",
        "P4_3",
        "P3_1",
        "P3_2",
        "P6_1",
        "P6_5",
    ]
    return space_group in bieberbach

def is_sohncke_space_group(space_group: str) -> bool:
    return is_chiral_point_group(spacegroup_to_pointgroup(space_group))

def is_symmorphic_space_group(space_group: str) -> bool:
    return space_group[1:] in AUGMENTED_POINT_GROUPS

def generate_point_group(generators: list[npt.NDArray]) -> list[npt.NDArray]:
    """Generate a finite group.

    Args:
        generators (list[npt.NDArray]): Generators of the group. shape = (3, 3).

    Returns:
        list[npt.NDArray]: Group.
    """
    def simplify(state: list[npt.NDArray]) -> list[npt.NDArray]:
        uniques: list[npt.NDArray] = []
        visited: list[int] = [-1] * len(state)
        for i in range(len(state)-1):
            a: npt.NDArray = state[i]
            if visited[i] != -1:
                continue
            uniques.append(a)
            for j in range(i+1, len(state)):
                b: npt.NDArray = state[j]
                if np.sum(np.abs(a-b)) < 1e-5:
                    visited[j] = i
        return uniques
    
    def amplify(state: list[npt.NDArray]) -> list[npt.NDArray]:
        components: list[npt.NDArray] = []
        for i in range(len(state)):
            a: npt.NDArray = state[i]
            components.append(a)
            for j in range(i, len(state)):
                b: npt.NDArray = state[j]
                if i == j:
                    continue
                components.append(a @ b)
        res: list[npt.NDArray] = []
        for i in range(6):
            for a in components:
                res.append(np.linalg.matrix_power(a, i))
        return res

    res: list[npt.NDArray] = generators
    for _ in range(6):
        res = amplify(res)
        na: int = len(res)
        res = simplify(res)
        ns: int = len(res)
        if na == ns:
            break
    return res

def crystal_system_to_symmetry_directions(name: str) -> tuple[list[str]]:
    if name == "triclinic":
        return [], [], []
    elif name == "monoclinic-b":
        return ["010"], [], []
    elif name == "monoclinic-c":
        return ["001"], [], []
    elif name == "orthorhombic":
        return ["100"], ["010"], ["001"]
    elif name == "tetragonal":
        return ["001"], ["100", "010"], ["110", "1-10"]
    elif name == "hexagonal":
        return ["001"], ["100", "010", "-1-10"], ["1-10", "120", "-2-10"]
    elif name == "trigonal":
        return ["001"], ["100", "010", "-1-10"], ["1-10", "120", "-2-10"]
    elif name == "rhombohedral":
        return ["111"], ["1-10", "01-1", "-101"], []
    elif name == "cubic":
        return ["100", "010", "001"], ["111", "1-1-1", "-11-1", "-1-11"], ["110", "1-10", "101", "-101", "011", "01-1"]
    else:
        raise ValueError

def spacegroup_to_symmetry_directions(spacegroup_name: str, monoclinic_unique_axis: str | None = None, trigonal_rhombohedral: bool = False) -> tuple[list[str]]:
    pointgroup_name: str = spacegroup_to_pointgroup(spacegroup_name)
    if pointgroup_to_crystal_system(pointgroup_name) == "monoclinic":
        if monoclinic_unique_axis is None:
            divided_spacegroup_name: list[str] = [mat.group() for mat in re.finditer(r"[a-emn]|(\d(_\d)?/[a-emn])|(-\d)|\d(_\d)?|[A-Z]", spacegroup_name)]
            if divided_spacegroup_name[1] == divided_spacegroup_name[3] == "1":
                monoclinic_unique_axis = "b"
            elif divided_spacegroup_name[1] == divided_spacegroup_name[2] == "1":
                monoclinic_unique_axis = "c"
            else:
                raise ValueError
    
    symmetry_directions: tuple[list[str]] = crystal_system_to_symmetry_directions(
        pointgroup_to_crystal_system(
            pointgroup_name,
            monoclinic_unique_axis=monoclinic_unique_axis,
            trigonal_rhombohedral=trigonal_rhombohedral
        )
    )
    return symmetry_directions

def decompose_jahn_symbol(symbol: str) -> tuple[int, str, bool, bool]:
    """Extract information of a tensor represented by the given Jahn's symbol.

    Args:
        symbol (str): Jahn's symbol.\n
        [Rules of Jahn's symbol notation]:\n
        * 'e' represents axial tensor.
        * 'a' represents magnetic tensor.
        * Numbers after 'V' represent rank of partial tensor.
        * Symbols in '[]' are symmetric tensor.
        * Symbols in '{}' are asymmetric tensor.
        * Symbols in '[]*' are symmetric tensor by time reversal operation.
        * Symbols in '{}*' are asymmetric tensor by time reversal operation.

    Returns:
        (tuple[int, str, bool, bool]): 'rank', 'expression', 'axiality' and 'time_reversality' respectively.
            rank: Rank of the tensor.
            expression: Relationship among elements of the tensor represented by formula of indices.
            axiality: Axiality of the tensor. If it is polar tensor, this value will `False`.
            time_reversality: time reversality of the tensor. If it is magnetic tensor, this value will `True`.
    
    Examples:
        >>> decompose_jahn_symbol("ae[V^2]V")
        >>> (3, 'ijk=jik', True, True)
    """
    axiality: bool = "e" in symbol
    time_reversality: bool = "a" in symbol
    symbol = re.sub(r"a|e|\^", "", symbol) # aとeを消去
    symbol = re.sub(r"V([^\d]|$)", r"V1\1", symbol) # 'V' -> 'V1'
    symbol = re.sub(r"\[(.*)\]\*", r"(\1)", symbol) # []* -> ()
    symbol = re.sub(r"\{(.*)\}\*", r"<\1>", symbol) # {}* -> <>
    
    def symbol_to_expression(symbol: str, i: int, base: list[str], expr_list: list[str], n_char: int) -> tuple[str, int, list[str], list[str], int]:
        def regenerate_base_expr_list(base: list[str], expr_list: list[str], symmetry_type: int) -> tuple[list[str], list[str]]:
            if len(base) == 1:
                return base, expr_list
            new_base: list[str] = ["".join(base)]
            new_expr_list: list[str] = []
            divided_expr_list: list[str] = [[expr for expr in expr_list if b in expr] for b in base]
            # baseの統合
            for i in range(len(base)):
                left_bases: str = "".join(base[:i])
                right_bases: str = "".join(base[i+1:]) if i+1 < len(base) else ""
                for expr in divided_expr_list[i]:
                    # '=' の前半のindexをnew_baseに合わせて拡張
                    new_expr: str = re.sub(r"^(.*)=", left_bases+r"\1"+right_bases+"=", expr)
                    # '=' の後半のindexをnew_baseに合わせて拡張
                    new_expr = re.sub(r"(=-|=)(.*)$", r"\1"+left_bases+r"\2"+right_bases, new_expr)
                    # 時間反転操作がある場合は後ろにずらす
                    if "'" in new_expr:
                        new_expr = re.sub(r"'", "", new_expr) + "'"
                    new_expr_list.append(new_expr)
            for i, j in combinations(range(len(base)), 2): # 全てのbaseの置換に対して対称性ごとに処理
                ji: str = "".join([base[j] if n == i else (base[i] if n == j else b) for n,b in enumerate(base)])
                # 対称性に応じてbase間の関係を追加
                if symmetry_type == 0:
                    # 単純統合
                    pass
                elif symmetry_type == 1:
                    # 対称
                    new_expr_list.append(f"{new_base[0]}={ji}")
                elif symmetry_type == 2:
                    # 反対称
                    new_expr_list.append(f"{new_base[0]}=-{ji}")
                elif symmetry_type == 3:
                    # 時間反転で対称
                    new_expr_list.append(f"{new_base[0]}={ji}'")
                elif symmetry_type == 4:
                    # 時間反転で反対称
                    new_expr_list.append(f"{new_base[0]}=-{ji}'")
                else:
                    pass
            return new_base, new_expr_list

        characters: str = "ijklmnopqrstuvwxyzabcdefgh"
        left_parentheses: str = "[{(<"
        right_parentheses: str = "]})>"

        if i == -1: # 最終的な結果
            i += 1
            symbol, i, base, expr_list, n_char = symbol_to_expression(symbol, i, base, expr_list, n_char)
            base, expr_list = regenerate_base_expr_list(base, expr_list, symmetry_type=0)
            return symbol, i, base, expr_list, n_char
        while i < len(symbol):
            if symbol[i] in left_parentheses:
                symmetry_type: int = 1 + left_parentheses.index(symbol[i])
                i += 1
                symbol, i, new_base, new_expr_list, n_char = symbol_to_expression(symbol, i, [], [], n_char)
                new_base, new_expr_list = regenerate_base_expr_list(new_base, new_expr_list, symmetry_type=symmetry_type)
                base = base + new_base
                expr_list = expr_list + new_expr_list
            elif symbol[i] in right_parentheses:
                i += 1
                return symbol_to_expression(symbol, i, base, expr_list, n_char)
            elif symbol[i] == "V":
                base = base + [characters[n_char+b] for b in range(int(symbol[i+1]))]
                n_char += int(symbol[i+1])
                i += 2
                return symbol, i, base, expr_list, n_char
            else:
                raise ValueError
        return symbol, i, base, expr_list, n_char
    _, _, _, expr_list, rank = symbol_to_expression(symbol, -1, [], [], 0)

    return rank, ",".join(expr_list), axiality, time_reversality


class _UnionFind:
    def __init__(self, n: int): # O(n)
        self.parent: list[int] = [-1]*n
        self.n: int = n
    def root(self, x: int) -> int:
        """Return the number of the root of `x`.

        Note:
            Time complexity is O(α(n)), where n is the size of the entire set.

        Args:
            x (int): The number of the element.

        Returns:
            int: The number of the root of `x`.
        """
        if self.parent[x] < 0:
            return x
        else:
            self.parent[x] = self.root(self.parent[x])
            return self.parent[x]
        
    def size(self, x: int) -> int:
        """Return the size of the group to which `x` belongs.

        Note:
            Time complexity is O(n), where n is the size of the entire set.

        Args:
            x (int): The number of the element.

        Returns:
            int: The number of the size of the group to which `x` belongs.
        """
        return -self.parent[self.root(x)]
    
    def merge(self, x: int, y: int) -> bool: # xとyを結合する O(α(n))
        """Merge `x` and `y`.

        Note:
            Time complexity is O(α(n)), where n is the size of the entire set.

        Args:
            x (int): The number of the element.
            y (int): The number of the element.

        Returns:
            bool: Whether `x` and `y` belonged to the same group.
        """
        x = self.root(x)
        y = self.root(y)
        if x == y:
            return False
        if self.parent[x] > self.parent[y]: # for optimization
            x, y = y, x
        self.parent[x] += self.parent[y]
        self.parent[y] = x
        return True
    
    def issame(self, x: int, y: int) -> bool:
        """Judge whether `x` and `y` belong to the same group.

        Note:
            Time complexity is O(α(n)), where n is the size of the entire set.

        Args:
            x (int): The number of the element.
            y (int): The number of the element.

        Returns:
            bool: Whether `x` and `y` belong to the same group.
        """
        return self.root(x) == self.root(y)
    
    def family(self, x: int) -> list[int]:
        """Return the group of `x`.

        Note:
            Time complexity is O(n), where n is the size of the entire set.

        Args:
            x (int): The number of the element.

        Returns:
            list[int]: The group of `x`.
        """
        return [i for i in range(self.n) if self.issame(i, x)]
    
    def maximum(self) -> list[int]:
        """Return the group which has the maximum size among the groups.

        Note:
            Time complexity is O(n), where n is the size of the entire set.

        Returns:
            list[int]: The group which has the maximum size among the groups.
        """
        return self.family(self.parent.index(min(self.parent)))
    
    def all_root(self) -> list[int]:
        """Return the roots of the groups.

        Note:
            Time complexity is O(n), where n is the size of the entire set.

        Returns:
            list[int]: The roots of the groups.
        """
        return [i for i in range(self.n) if self.parent[i] < 0]
    
    def decompose(self) -> list[list[int]]:
        """Return the groups.

        Note:
            Time complexity is O(nα(n)), where n is the size of the entire set.

        Args:
            x (int): The number of the element.

        Returns:
            list[list[int]]: The groups.
        """
        return [self.family(i) for i in self.all_root()]

SymmetryOperationchild = TypeVar("SymmetryOperationchild", bound="SymmetryOperation")

class SymmetryOperation(MatrixREF):
    """Symmetry operation represented as matrix.

    This class is inherited from `MatrixREF`.

    Args:
        p (int): Square of the generator of simple extension Q(√p).
        mat (list[list[REF]]): 2-dimension matrix of `REF` instance.
        name (str): Name of the symmetry operation.
        time_reversal (bool): True if the symmetry operation contains time-reversal operation.

    Attributes:
        p (int): Square of the generator of simple extension Q(√p).
        mat (list[list[REF]]): 2-dimension matrix of `REF` instance.
        shape (tuple[int, int]): Shape of `MatrixREF.mat`. First element is the number of row, second is the number of column.
        name (str): Name of the symmetry operation.
        time_reversal (bool): True if the symmetry operation contains time-reversal operation.
        determinant (int): Determinant of the symmetry operation. This value should be +1 or -1 due to orthorhombic transformation.
    """
    def __init__(self, p: int, mat: list[list[REF]], name: str = "", time_reversal: bool = False):
        super().__init__(p, mat=mat)
        self.name: str = name
        self.time_reversal: bool = time_reversal
    
    @property
    def determinant(self) -> int:
        return int(np.sign(np.linalg.det(self.to_ndarray())))

    def __neg__(self: SymmetryOperationchild) -> SymmetryOperationchild:
        m, n = self.shape
        res: list[list[REF]] = [[REF(self.p)]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                res[i][j] = -self.mat[i][j]
        ret: SymmetryOperationchild = self.__class__(
            self.p, 
            mat=res, 
            name="-" + self.name,
            time_reversal=self.time_reversal
        )
        return ret
    
    def __matmul__(self: SymmetryOperationchild, other: SymmetryOperationchild) -> SymmetryOperationchild:
        res: MatrixREF = super().__matmul__(other)
        ret: SymmetryOperationchild = self.__class__(
            self.p, 
            mat=res.mat, 
            name=f"{self.name} {other.name}",
            time_reversal=self.time_reversal ^ other.time_reversal
        )
        return ret

Relation = List[Tuple[REF, int]]
Relations = List[List[Tuple[REF, int]]]
Relation_ternary = List[Tuple[REF, Tuple[int, ...]]]
Relations_ternary = List[List[Tuple[REF, Tuple[int, ...]]]]

class PhysicalPropertyTensorAnalyzer:
    """Analyze non-zero elements of physical property tensors based on the symmetry of crystallographic point group.

    There is software on the browser that performs similar or better than this class.
    'MTENSOR": https://www.cryst.ehu.es/cgi-bin/cryst/programs/mtensor.pl
    If you want to double-check the result of this class, use 'MTENSOR'.

    Note:
        All symmetry operations of crystallographic point groups can be represented as a 3×3 matrix on a simple rational extension field: 'M_{3×3}(Q(√3))' in an appropriate orthorhombic basis.
        Therefore, it is possible to determine which elements are equivalent or zero by straightforward exact calculation.

    Attributes:
        point_group_name (str): Target point group name written in International notation.
        unitary_matrices (list[MatrixREF]): list of the symmetry operations of the crystallographic point group represented as a matrix in an appropriate orthorhombic basis.

    """
    # 適切に定めた直交座標系を基底にとったときの対称操作の行列表示
    # 基本並進ベクトルたちが直交系をなすならそれを使う
    C1 = SymmetryOperation(3, [[1,0,0],
                                [0,1,0],
                                [0,0,1]], name="C1")

    C2_100 = SymmetryOperation(3, [[1,0,0],
                                    [0,-1,0],
                                    [0,0,-1]], name="C2_100")
    C2_010 = SymmetryOperation(3, [[-1,0,0],
                                    [0,1,0],
                                    [0,0,-1]], name="C2_010")
    C2_001 = SymmetryOperation(3, [[-1,0,0],
                                    [0,-1,0],
                                    [0,0,1]], name="C2_001")

    C2_n101 = SymmetryOperation(3, [[0,0,-1],
                                   [0,-1,0],
                                   [-1,0,0]], name="C2_n101")
    C2_1n10 = SymmetryOperation(3, [[0,-1,0],
                                    [-1,0,0],
                                    [0,0,-1]], name="C2_1n10")
    C2_01n1 = SymmetryOperation(3, [[-1,0,0],
                                    [0,0,-1],
                                    [0,-1,0]], name="C2_01n1")

    C2_110 = SymmetryOperation(3, [[0,1,0],
                                    [1,0,0],
                                    [0,0,-1]], name="C2_110")

    C3_111 = SymmetryOperation(3, [[0,0,1],
                                    [1,0,0],
                                    [0,1,0]], name="C3_111")
    C3_1n1n1 = SymmetryOperation(3, [[0,0,-1],
                                    [-1,0,0],
                                    [0,1,0]], name="C3_1n1n1")
    C3_n11n1 = SymmetryOperation(3, [[0,0,1],
                                    [-1,0,0],
                                    [0,-1,0]], name="C3_n11n1")
    C3_n1n11 = SymmetryOperation(3, [[0,1,0],
                                    [0,0,-1],
                                    [-1,0,0]], name="C3_n1n11")

    C3_001 = SymmetryOperation(3,  [[REF(3,Fraction(-1,2)),REF(3,b=Fraction(1,2)),0],
                                    [REF(3,b=Fraction(-1,2)),REF(3,Fraction(-1,2)),0],
                                    [0,0,1]], name="C3_001")

    C4_001 = SymmetryOperation(3, [[0,-1,0],
                                    [1,0,0],
                                    [0,0,1]], name="C4_001")
    C4_010 = SymmetryOperation(3, [[0,0,1],
                                    [0,1,0],
                                    [-1,0,0]], name="C4_010")
    C4_100 = SymmetryOperation(3, [[1,0,0],
                                    [0,0,-1],
                                    [0,1,0]], name="C4_100")


    m_100 = SymmetryOperation(3, [[-1,0,0],
                                    [0,1,0],
                                    [0,0,1]], name="m_100")
    m_010 = SymmetryOperation(3, [[1,0,0],
                                    [0,-1,0],
                                    [0,0,1]], name="m_010")
    m_001 = SymmetryOperation(3, [[1,0,0],
                                    [0,1,0],
                                    [0,0,-1]], name="m_001")

    m_110 = SymmetryOperation(3, [[0,-1,0],
                                    [-1,0,0],
                                    [0,0,1]], name="m_110")
    m_1n10 = SymmetryOperation(3, [[0,1,0],
                                    [1,0,0],
                                    [0,0,1]], name="m_1n10")
    m_n101 = SymmetryOperation(3, [[0,0,1],
                                    [0,1,0],
                                    [1,0,0]], name="m_n101")
    m_01n1 = SymmetryOperation(3, [[1,0,0],
                                    [0,0,1],
                                    [0,1,0]], name="m_01n1")

    inversion = SymmetryOperation(3, [[-1,0,0],
                                        [0,-1,0],
                                        [0,0,-1]], name="inversion")

    time_reversal = SymmetryOperation(3, [[1,0,0],
                                            [0,1,0],
                                            [0,0,1]], name="time-reversal", time_reversal=True)

    # 国際表記 -> Schönflies表記
    international_to_schoenflies_notation: dict[str, str] = {
        # 立方晶(cubic system)
        "m-3m": "Oh",
        "4/m-32/m": "Oh", # 表記揺れ
        "-43m": "Td",
        "432": "O",
        "m-3":"Th",
        "23": "T",
        # 正方晶(tetragonal system)
        "4/mmm": "D4h",
        "-42m": "D2d",
        "-4m2": "D2d", # 表記揺れ
        "4mm" :"C4v",
        "422" :"D4",
        "4/m": "C4h",
        "-4": "S4",
        "4": "C4",
        # 直方晶(orthorhombic system)
        "mmm": "D2h",
        "mm2": "C2v",
        "2mm": "C2v", # 表記揺れ
        "222": "D2",
        # 六方晶(hexagonal system)
        "6/mmm": "D6h",
        "-62m": "D3h",
        "-6m2": "D3h", # 表記揺れ
        "6mm": "C6v",
        "622": "D6",
        "6/m": "C6h",
        "-6": "C3h",
        "6": "C6",
        # 三方晶(trigonal system)
        "-3m": "D3d",
        "3m": "C3v",
        "32": "D3",
        "-3": "S6",
        "3": "C3",
        # 単斜晶(monoclinic system)
        "2/m": "C2h",
        "m": "S1",
        "2": "C2",
        # 三斜晶(triclinic system)
        "-1": "S2",
        "1": "C1"
    }

    point_groups: list[str] = [
        "1", "-1",
        "2", "m", "2/m",
        "3", "-3", "32", "3m", "-3m",
        "6", "-6", "6/m", "622", "6mm", "-62m", "6/mmm",
        "222", "mmm", "mm2",
        "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
        "23", "m-3", "432", "-43m", "m-3m",
    ]

    enantiomorphous_point_groups: list[str] = [
        "1", "2", "222", "4", "422", "3", "32", "6", "622", "23", "432"
    ]

    chiral_point_groups: list[str] = [
        "1", "2", "222", "4", "422", "3", "32", "6", "622", "23", "432"
    ]

    polar_point_groups: list[str] = [
        "1", "m", "2", "mm2", "4", "4mm", "3", "3m", "6", "6mm",
    ]

    centrosymmetric_point_groups: list[str] = [
        "-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m",
    ]

    noncentrosymmetric_point_groups: list[str] = [
        "1", "2", "m", "222", "mm2", "4", "-4", "422", "4mm", "-42m", "3", "3m", "32", "6", "-6", "622", "6mm", "-62m", "23", "432", "-43m",
    ]


    # 参照元 https://www.cryst.ehu.es/cryst/get_point_genpos.html (三方晶やC3_001の行列は改変)
    # 各結晶点群の生成元(適切に定めた直交座標系を基底にとっていることに注意)
    # 三方晶では
    # (default):   [001]方向をz軸，2回回転がある場合はa軸が回転軸，鏡映がある場合はa軸に垂直に鏡映面
    # _rombohedral: [111]方向をxyzでの(1,1,1)方向，c軸とz軸と[111]がx=y上になるようにとる
    pointgroup_generators: dict[str, list[SymmetryOperation]] = {
        # 立方晶
        "Oh": [C2_001, C2_010, C3_111, C2_110, inversion],
        "Td": [C2_001, C2_010, C3_111, m_1n10],
        "O":  [C2_001, C2_010, C3_111, C2_110],
        "Th": [C2_001, C2_010, C3_111, inversion],
        "T":  [C2_001, C2_010, C3_111],
        # 正方晶
        "D4h":[C2_001, C4_001, C2_010, inversion],
        "D2d":[C2_001, -C4_001, C2_010],
        "C4v":[C2_001, C4_001, m_010],
        "D4": [C2_001, C4_001, C2_010],
        "C4h":[C2_001, C4_001, inversion],
        "S4": [C2_001, -C4_001],
        "C4": [C2_001, C4_001],
        # 直方晶
        "D2h":[C2_001, C2_010, inversion],
        "C2v":[C2_001, m_010],
        "D2": [C2_001, C2_010],
        # 六方晶
        "D6h":[C3_001, C2_001, C2_110, inversion],
        "D3h":[C3_001, m_001, m_110],
        "C6v":[C3_001, C2_001, m_110],
        "D6": [C3_001, C2_001, C2_110],
        "C6h":[C3_001, C2_001, inversion],
        "C3h":[C3_001, m_001],
        "C6": [C3_001, C2_001],
        # 三方晶
        "D3d":[C3_001, C2_100, inversion],
        "C3v":[C3_001, m_100],
        "D3": [C3_001, C2_100],
        "S6": [C3_001, inversion],
        "C3": [C3_001],
        #"D3d_rhombohedral": [C3_111, C2_1n10, inversion],
        #"C3v_rhombohedral": [C3_111, m_1n10],
        #"D3_rhombohedral":  [C3_111, C2_1n10],
        #"S6_rhombohedral":  [C3_111, inversion],
        #"C3_rhombohedral":  [C3_111],
        # 単斜晶
        "C2h":[C2_001, inversion],
        "S1": [m_001],
        "C2": [C2_001],
        # 三斜晶
        "S2": [inversion],
        "C1": [C1],
    }

    def __init__(self, point_group_name: str) -> None:
        self.point_group_name: str = point_group_name
        pgname_in_schoenflies_notation: str = self.international_to_schoenflies_notation[self.point_group_name]
        self.unitary_matrices: list[SymmetryOperation] = self.pointgroup_generators[pgname_in_schoenflies_notation]

    @classmethod
    def _Gauss_Elimination_REF(cls, A: MatrixREF) -> MatrixREF: # 掃き出し法による行簡約化
        """Gaussian elimination (row reduction).

        Args:
            A (MatrixREF): Matrix of `REF`.

        Returns:
            MatrixREF: Matrix of `REF`.
        """
        B: MatrixREF = A.deepcopy()
        m: int
        n: int
        m, n = B.shape # m×n行列
        zero: REF = REF(B.p)
        for k in range(n):
            if k == m:
                break
            flag: bool = True
            i: int = m-1
            for idx in range(k,m):
                if B[idx][k] != zero:
                    flag = False
                    i = idx
                    break
            if flag: # k列は簡約済み
                continue
            if i != k: # i行とk行の入れ替え
                for j in range(k,n):
                    B[i][j], B[k][j] = B[k][j], B[i][j]
            w: REF = B[k][k].inv()
            for j in range(k,n): # k行全体をB[k][k]で割って先頭を1にする
                B[k][j] *= w
            for i in range(m):
                if i == k:
                    continue
                v: REF = B[i][k].deepcopy()
                for j in range(k,n): # i行からk行*B[i][k]を引く
                    B[i][j] -= v*B[k][j]
        return B

    @classmethod
    def _ternary(cls, n: int, fill: int, is_0indexed: bool = True) -> tuple[int, ...]: # fill: 桁数
        """Transform a given integer in ternary notation.

        Note:
            Note that the order of the numbers in the return value is reversed from the usual.
            For example, 11 = 102_(3), but this method returns (2, 0, 1).
            
        Args:
            n (int): Integer.
            fill (int): Number of digits.
            is_0indexed (bool, optional): If `False`, the result is expressed in 1-indexed. Defaults to True.
                examples:
                    >>> _ternary(11, 4, True) # (2,0,1,0)
                    >>> _ternary(5, 3, True) # (2,1,0)
                    >>> _ternary(5, 3, False) # (3,2,1)

        Returns:
            tuple[int, ...]: Integer in ternary notation.
        """
        res: list[int] = [] # res[i] := nの3進展開の"下から"i桁目
        r: int
        while n:
            n, r = divmod(n,3)
            res.append(r)
        if is_0indexed: # 0-indexed
            return tuple(res + [0]*max(0,fill-len(res)))
        else: # 1-indexed
            res = [i+1 for i in res]
            return tuple(res + [1]*max(0,fill-len(res)))
    
    @classmethod
    def _ternary2int(cls, ternary_ijk: tuple[int, ...]) -> int:
        """Reverse method of `_ternary`.

        Args:
            ternary_ijk (tuple[int, ...]): Integer expressed in ternary notation.

        Returns:
            int: Restored integer.
        """
        return sum([n * 3**i for i,n in enumerate(ternary_ijk)])
    
    @classmethod
    def _relations_from_symmetry_operation(cls, rank: int, R: SymmetryOperation, axiality: bool, time_reversality: bool = False) -> Relations:
        """Generate relations between each element of the tensor from symmetry operations.

        Note:
            The calculation is based on Neumann's principle.
            For example,
            rank=2, not axial: A_{ij} = R_{il} R{jm} A_{lm},
            rank=3, axial   : A_{ijk} = det(R) R_{il} R{jm} R_{kn} A_{lmn}.

        Args:
            rank (int): Rank of the tensor.
            R (SymmetryOperation): Symmetry operation.
            axiality (bool): Axiality of the tensor. True if the tensor is an axial tensor.
            time_reversality (bool, optional): Time-reversality of the tensor. \
                True if the sign is reversed by a time reversal operation. Defaults to False.

        Returns:
            Relations: Relations between elements of the tensor.
                i-th element of the return value `relations` represents the following formula:
                    \sum_{j} relations[i][j][0] * A_{relations[i][j][1]} = 0,
                where A is the physical property tensor and the indices of A shall be written in ternary notation.
        """
        n_elem: int = 3**rank
        zero: REF = REF(3) # 有理拡大体の零元
        relations: Relations = []
        for ijk in range(n_elem):
            now: Relation = []
            val_ijk: REF = REF(3, Fraction(1,1))
            for lmn in range(n_elem): # 添字について和をとる
                val: REF = REF(3, Fraction(1,1))
                for a, b in zip(cls._ternary(ijk,rank), cls._ternary(lmn,rank)):
                    val *= R[a][b]
                if val != zero: # 非ゼロの項のみ記録
                    sign: REF = REF(3, Fraction(-1,1)) # -1
                    # 軸性テンソルにimproper xor 磁気的テンソルに時間反転 -> 符号は-1倍
                    if (axiality and R.determinant == -1) ^ (time_reversality and R.time_reversal):
                        sign = -sign
                    if ijk == lmn:
                        val_ijk += sign * val
                    else:
                        now.append((sign * val, lmn))
            if val_ijk != zero:
                now.append((val_ijk, ijk))
            if now:
                relations.append(now)
        return relations

    @classmethod
    def _relations_from_expression(cls, rank: int, expr: str) -> Relations:
        """Generate relations between elements of the tensor from physical constraints.

        Note:
            You can use primed expressions (related time-reversal operation),
            but this method ignores those. For example, `"ij = -ji'"`.

        Args:
            rank (int): Rank of the tensor.
            expr (str): Expression of relations among indices based on physical constraints.
                If there is more than one expression, separate them with a comma.
                For example, `"ijk = -ikj"`, `"ijkl = ijlk, ijkl = jikl, ijkl = klij"`.

        Raises:
            ValueError: Lengths of components in the expression are different.
                For example, `"ijk = ikjl"`.

        Returns:
            Relations: Relations between elements of the tensor.
        """
        n_elem: int = 3**rank
        # 時間反転操作で等価になる場合"'"(prime)で表示できるが，この関数では無視する
        expressions: list[list[str]] = [s.split("=") for s in re.sub(r"[\u3000 \t]", "", expr).split(",") if not "'" in s]
        expressions_data: list[tuple[list[str], int, defaultdict[str,list[int]], int, defaultdict[str,list[int]]]] = []
        relations: Relations = []
        for expression in expressions:
            for i in range(1,len(expression)):
                expr0: str = expression[0]
                expr1: str = expression[i]
                sign0: int = 1
                sign1: int = 1
                if expr0[0] == "-":
                    sign0 = -1
                    expr0 = expr0[1:]
                if expr1[0] == "-":
                    sign1 = -1
                    expr1 = expr1[1:]
                if len(expr0) != rank or len(expr1) != rank:
                    raise ValueError(f"expressions must be all the same length: {rank}")
                characters: list[str] = list(set(list(expr0+expr1)))
                d0: defaultdict[str, list[int]] = defaultdict(list) # 添字表現=expressionにおいて，文字cが出現するindexのリスト
                d1: defaultdict[str, list[int]] = defaultdict(list)
                for j in range(rank):
                    d0[expr0[j]].append(j)
                    d1[expr1[j]].append(j)
                expressions_data.append((characters, sign0, d0, sign1, d1)) # この関係式で出現する文字全体の集合も持っておく
        for ijk in range(n_elem):
            for lmn in range(ijk,n_elem):
                ijk_ternary: tuple[int, ...] = cls._ternary(ijk, rank)
                lmn_ternary: tuple[int, ...] = cls._ternary(lmn, rank)
                for characters, s0, d0, s1, d1 in expressions_data:
                    flag: bool = True
                    for c in characters: # 表現の文字ごとに見る
                        numbers: set[int] = set()
                        for i in d0[c]:
                            numbers.add(ijk_ternary[i])
                        for l in d1[c]:
                            numbers.add(lmn_ternary[l])
                        if len(numbers) != 1: # 同じ文字に対応する添字の数字は1種類でないといけない
                            flag = False
                            break
                    if flag: # 全ての文字について条件を満たすならそれらのテンソル要素は表現に従う
                        # s0とs1の符号が一致するなら ijk-lmn=0, 違うなら ijk+lmn=0
                        relations.append([(REF(3, Fraction(1,1)), ijk), (REF(3, Fraction(-s0*s1,1)), lmn)])
                        break
        return relations

    @classmethod
    def _relation_to_ternary(cls, relation: Relation, rank: int) -> Relation_ternary:
        """Rewrite a `Relation`(list[tuple[REF, int]]) type variable in ternary notation.

        Args:
            relation (Relation): Relation between each element of the tensor.
            rank (int): Rank of the tensor.

        Returns:
            Relation_ternary: Relation in ternary notation.
        """
        # relationの添字を3進数に変換(printでの出力用)
        return [(val, cls._ternary(ijk,rank)) for val,ijk in relation]
    
    @classmethod
    def _relations_to_ternary(cls, relations: Relations, rank: int) -> list[Relation_ternary]:
        """Rewrite a `Relations`(list[list[tuple[REF, int]]]) type variable in ternary notation.

        Args:
            relations (Relations): Relations between elements of the tensor.
            rank (int): Rank of the tensor.

        Returns:
            list[Relation_ternary]: Relations in ternary notation.
        """
        # relationsの添字を3進数に変換(printでの出力用)
        return [cls._relation_to_ternary(relation, rank) for relation in relations]
    
    @classmethod
    def _extract_relation(cls, relations: Relations, rank: int, ternary_ijk: tuple[int, ...]) -> Relations:
        """Extract the relations which contain a particular index from a `Relations` type variable.

        Args:
            relations (Relations): Relations between elements of the tensor.
            rank (int): Rank of the tensor.
            ternary_ijk (tuple[int, ...]): Index in ternary notation.

        Returns:
            Relations: Relations which contain a particular index.
        """
        # ternary_ijkが入っている関係式を抽出
        res: Relations = []
        for relation in relations:
            if cls._ternary2int(ternary_ijk) in [cls._ternary(ijk,rank) for _,ijk in relation]:
                res.append(relation)
        return res
    
    @classmethod
    def _formulize_relations(cls, relations: Relations, rank: int) -> list[str]:
        """ToDo: is_0indexed に対応
        """
        # 関係式を数学的に表示
        res: list[str] = []
        for relation in relations:
            if len(relation) > 0:
                res.append(" + ".join([f"{val} ({''.join([str(i) for i in ijk])})" for val,ijk in cls._relation_to_ternary(relation,rank)]) + " = 0")
        return res

    @classmethod
    def _nonzero_matrix(cls, nonzero: set[int], direction: int, rank: int) -> str:
        """Display nonzero elements of a magnetic-dependent 2-rank tensor as matrix.

        Args:
            nonzero (set[int]): Set of indices of nonzero elements.
            direction (int): Direction of magnetic field.
            rank (int): Rank of the tensor. (2 + magnetic field order)

        Returns:
            str: 2D matrix of which nonzero elements are "o" and zero elements are empty.
        """
        res: list[list[str]] = [[" " if i>j else "." for j in range(3)] for i in range(3)]
        for ijk in nonzero:
            i, j, *klm = cls._ternary(ijk, rank)
            if i <= j and all([k == direction for k in klm]):
                # 3階: k=direction について ijk を行列表示
                # 4階: k=direction について ijkk を行列表示
                # n階: k=direction について ijk...k を行列表示
                res[i][j] = "o"
        return "\n".join([" ".join(r) for r in res])

    @classmethod
    def _summarize_same_term(cls, relations: Relations) -> Relations:
        """Summarize the same term in each `Relation`.

        Args:
            relations (Relations): Relations between elements of the tensor.

        Returns:
            Relations: Summarized relations.
        """
        # 同じ項をまとめる
        renewed_relations: Relations = []
        for relation in relations:
            dic: defaultdict[int, REF] = defaultdict(lambda: REF(3))
            for val, ijk in relation:
                dic[ijk] += val
            renewed_relations.append([(v,k) for k,v in dic.items() if v != REF(3)])
        return renewed_relations

    @classmethod
    def _delete_zero_term(cls, relations: Relations, nonzero: set[int]) -> tuple[Relations, set[int], bool]:
        """Delete zero terms in each `Relation`.

        Args:
            relations (Relations): Relations between elements of the tensor.
            nonzero (set[int]): Set of nonzero indices.

        Returns:
            tuple[Relations, set[int], bool]: Updated relations, set of nonzero indices and update-flag.
        """
        # 0の要素を削除
        flag: bool = False # 更新が起こったかどうか
        relations = cls._summarize_same_term(relations) # 同じ項をまとめる
        renewed: Relations = []
        for relation in relations:
            if len(relation) == 1: # 長さ1の式なのでこの項は0
                flag = True
                ijk: int = relation[0][1]
                if ijk in nonzero:
                    nonzero.discard(ijk) # 0の要素を除外
            else:
                now: Relation = []
                for val, ijk in relation:
                    if not ijk in nonzero: # 式中の0の要素を削除
                        flag = True
                        continue
                    now.append((val,ijk))
                if now:
                    renewed.append(now)
        return renewed, nonzero, flag

    @classmethod
    def _untangle_relations(cls, rank: int, relations: Relations, nonzero: set[int]) -> tuple[Relations, set[int], bool]:
        """Simplify the relations by regarding relations 
        as simultaneous equations of indices and performing row reduction.

        Args:
            rank (int): Rank of the tensor.
            relations (Relations): Relations between elements of the tensor.
            nonzero (set[int]): Set of nonzero indices.

        Returns:
            tuple[Relations, set[int], bool]: Updated relations, set of nonzero indices and update-flag.
        """
        # 係数行列簡約化で関係式を簡約化
        flag: bool = False
        zero: REF = REF(3)
        n_elem: int = 3**rank
        U: _UnionFind = _UnionFind(n_elem)
        renewed: Relations = []
        for relation in relations:
            for i in range(len(relation)):
                for j in range(i,len(relation)):
                    if i == j:
                        continue
                    if relation[i][1] in nonzero and relation[j][1] in nonzero:
                        U.merge(relation[i][1], relation[j][1])

        for family in U.decompose(): # 関係式で結びついているテンソル要素たちごとに処理
            if len(family) == 1:
                continue
            family_inv: dict[int, int] = {a:i for i, a in enumerate(family)}
            A: list[list[REF]] = [] # 係数行列
            for relation in relations:
                if U.issame(relation[0][1], family[0]):
                    a = [zero]*len(family)
                    for val, ijk in relation:
                        if ijk in nonzero:
                            a[family_inv[ijk]] = val
                    A.append(a)

            AA: MatrixREF = cls._Gauss_Elimination_REF(MatrixREF(3, A)) # Gaussの消去法(掃き出し法)で行簡約
            # 簡約化して行の非ゼロ要素が1つだけならそれに対応するテンソル要素の値は0
            # それ以外の行は前よりも簡単になった関係式になっている
            zero_elements: list[int] = []
            m, n = AA.shape
            for i in range(m):
                idxs: list[int] = []
                for j in range(n):
                    if AA[i][j] != zero:
                        idxs.append(j)
                if len(idxs) == 1: # 行の非ゼロ要素が1つだけなのでこれに対応するテンソル要素はゼロ
                    zero_elements.append(family[idxs[0]])
                else: # 新しい関係式
                    renewed.append([(AA[i][j], family[j]) for j in idxs])
            for z in zero_elements:
                if z in nonzero:
                    flag = True
                    nonzero.discard(z)
        return renewed, nonzero, flag # tuple[list[list[tuple[REF,int]]], set[int], bool]

    @classmethod
    def _remove_duplicate(cls, relations: Relations, nonzero: set[int]) -> tuple[Relations, set[int], bool]:
        """Remove duplicate relations.

        Args:
            relations (Relations): Relations between elements of the tensor.
            nonzero (set[int]): Set of nonzero indices.

        Returns:
            tuple[Relations, set[int], bool]: Updated relations, set of nonzero indices and update-flag.
        """
        # 複数の等価な式を一本化
        flag: bool = False
        renewed: Relations = []
        for idx1 in range(len(relations)):
            d1: defaultdict[int, REF] = defaultdict(lambda: REF(3))
            for val, ijk in relations[idx1]:
                d1[ijk] += val
            flag2: bool = True # relations[idx1]と等価なものが存在するならFalse
            for idx2 in range(idx1,len(relations)):
                if idx1 == idx2:
                    continue
                d2: defaultdict[int, REF] = defaultdict(lambda: REF(3)) # (関係式)=0において，関係式の各テンソル要素の係数を計算
                d2_inv: defaultdict[int, REF] = defaultdict(lambda: REF(3)) # 両辺に-1をかけた関係式
                for val, ijk in relations[idx2]:
                    d2[ijk] += val
                    d2_inv[ijk] -= val

                if d1 == d2 or d1 == d2_inv:
                    flag2 = False
                    break
            if flag2: # 他に等価な式がないならOK
                renewed.append(relations[idx1])
            else:
                flag = True
        return renewed, nonzero, flag # tuple[list[list[tuple[REF,int]]], set[int], bool]

    @classmethod
    def _simplify_coefficient(cls, ref_list: list[REF]) -> list[REF]: # O(len(A))
        """Simplify the coefficient of the rational and irrational parts of `REF` 
        to integers while preserving the ratios of the elements of `ref_list`.

        Args:
            ref_list (list[REF]): List of `REF`.

        Returns:
            list[REF]: Simplified list of `REF`.
        """
        # Rの要素たちの比を保ったままREFの有理部と無理部の係数を整数に簡約化
        def lcm(a: int, b: int) -> int:
            return a * b // gcd(a, b) # int 
        L: list[int] = []
        G: list[int] = []
        z: Fraction = Fraction()
        flag: bool = False # 有理部の係数が全て0ならFalse
        for r in ref_list:
            a: Fraction
            b: Fraction
            a, b = r.a, r.b
            if a != z: # 有理部が非ゼロなら分母と分子をそれぞれ追加
                flag = True
                L.append(a.denominator)
                G.append(a.numerator)
            if b != z: # 無理部が非ゼロなら分母と分子をそれぞれ追加
                L.append(b.denominator)
                G.append(b.numerator)
        l: int = reduce(lcm, L)
        g: int = reduce(gcd, G)
        f: Fraction = Fraction(l, g)
        res: list[REF]
        if flag:
            res = [r*f for r in ref_list]
        else: # 有理部と無理部の入れ替え
            res = [r.swap()*f for r in ref_list]
        return res # list[REF]

    @classmethod
    def _simplify_relations_value(cls, relations: Relations) -> Relations:
        """Simplify each relation in `relations` while preserving the ratios of the elements.

        Args:
            relations (Relations): Relations between elements of the tensor.

        Returns:
            Relations: Simplified relations.
        """
        # relation: list[tuple[REF,int]] (in relations)の比を保ったままREFの有理部と無理部の係数を整数に簡約化
        res: Relations = []
        for relation in relations:
            if len(relation) == 0:
                continue
            vals: list[REF] = []
            ijks: list[int] = []
            for val, ijk in relation:
                vals.append(val)
                ijks.append(ijk)
            res.append(list(zip(cls._simplify_coefficient(vals), ijks))) # REFの有理部と無理部の簡約化
        return res # list[list[tuple[REF, int]]]

    @classmethod
    def _extract_independent(cls, rank: int, relations: Relations, nonzero: set[int]) -> tuple[set[int], Relations]:
        """Extract independent elements.

        Args:
            rank (int): Rank of the tensor.
            relations (Relations): Relations between elements of the tensor.
            nonzero (set[int]): Set of nonzero indices.

        Returns:
            tuple[set[int], Relations]: Set of indices of independent elements 
                and relations where dependent elements are represented by independent elements.
        """
        def printer_indep_dep(dep):
            print([cls._ternary(dep, rank)])
        if len(relations) == 0:
            return nonzero, []
        # 独立な成分と従属成分に分離
        indep: set[int] = set()
        dep: set[int] = set()
        relations = [relation for relation in relations if len(relation) > 0]
        # 式に現れる項が少ない順に処理
        relations = sorted([sorted(relation, key=lambda x:cls._ternary(x[1], rank)) for relation in relations], key=lambda x:len(x))
        for relation in relations:
            family: list[int] = [ijk for _, ijk in relation]
            undetermined: int = 0
            for i, ijk in enumerate(family):
                if (not ijk in indep) and (not ijk in dep):
                    undetermined += 1
            if undetermined == 0:
                continue
            for i, ijk in enumerate(family):
                if ijk in indep or ijk in dep:
                    continue
                if undetermined == 1: # 関係式で最後に残ったものは従属成分
                    dep.add(ijk)
                    break
                indep.add(ijk) # 独立成分として扱う
                undetermined -= 1
        for ijk in nonzero: # 上の処理の後に従属成分にない非ゼロ項はすべて独立
            if not ijk in dep:
                indep.add(ijk)
        # 行列の列を入れ替えて，左側にdepを，右側にindepを寄せる
        # 掃き出せばdepをindepたちだけで表現できる
        active: list[int] = sorted(list(nonzero), key=lambda x:cls._ternary(x, rank))
        l: list[int] = []
        r: list[int] = []
        for ijk in active:
            if ijk in dep:
                l.append(ijk)
            elif ijk in indep:
                r.append(ijk)
            else:
                indep.add(ijk)
                r.append(ijk)
        active = l + r
        active_inv: dict[int, int] = {a:i for i, a in enumerate(active)}
        
        # printer_indep_dep(indep)
        # printer_indep_dep(dep)
        
        A: list[list[REF]] = [] # 係数行列
        for relation in relations:
            a = [REF(3)]*len(active)
            for val, ijk in relation:
                a[active_inv[ijk]] = val
            A.append(a)
        AA: MatrixREF = cls._Gauss_Elimination_REF(MatrixREF(3, A)) # 掃き出し法で行簡約
        
        dep_represented_by_indep: Relations = [] # 従属成分たちのみを対象とした関係式(基本的に引数のrelationsと等価)
        for i in range(len(AA)):
            idxs: list[int] = []
            for j in range(len(active)):
                if AA[i][j] != REF(3):
                    idxs.append(j)
            if len(idxs) > 0:
                dep_represented_by_indep.append([(AA[i][j], active[j]) for j in idxs])
        return indep, dep_represented_by_indep

    @classmethod
    def _represent_dep_by_indep(cls, rank: int, relation: Relation, is_0indexed: bool = True) -> str:
        """Represent dependent elements by independent elements.

        Args:
            rank (int): Rank of the tensor.
            relation (Relation): Relations between elements of the tensor.
            is_0indexed (bool, optional): If `False`, the result is expressed in 1-indexed.. Defaults to True.

        Returns:
            str: Formula where dependent elements are represented by independent elements.
        """
        # 独立成分のみで従属成分を表示
        r: REF
        ijk0: int
        r, ijk0 = relation[0]
        res: list[str] = []
        for val, ijk in relation[1:]:
            v: REF = -val/r
            if v.b == 0:
                if v.a == 1:
                    res.append(str(cls._ternary(ijk, rank, is_0indexed=is_0indexed)))
                elif v.a == -1:
                    res.append(f"-{cls._ternary(ijk, rank, is_0indexed=is_0indexed)}")
                else:
                    res.append(str(v.a)+str(cls._ternary(ijk, rank, is_0indexed=is_0indexed)))
            elif v.a == 0:
                if v.b.denominator == 1:
                    if v.b == 1:
                        res.append(f"√{v.p}"+str(cls._ternary(ijk, rank, is_0indexed=is_0indexed)))
                    else:
                        res.append(f"{v.b}√{v.p}"+str(cls._ternary(ijk, rank, is_0indexed=is_0indexed)))
                else:
                    if v.b.numerator == 1:
                        res.append(f"√{v.p}/{v.b.denominator}"+str(cls._ternary(ijk, rank, is_0indexed=is_0indexed)))
                    else:
                        res.append(f"{v.b.numerator}√{v.p}/{v.b.denominator}"+str(cls._ternary(ijk, rank, is_0indexed=is_0indexed)))
            else:    
                res.append(f"({v.a}+{v.b}√{v.p})"+str(cls._ternary(ijk, rank, is_0indexed=is_0indexed)))
        return f"{cls._ternary(ijk0, rank, is_0indexed=is_0indexed)} = " + " + ".join(res)


    @classmethod
    def _contract_relations_along_with_direction(cls, relations: Relations, rank: int, magnetic_field_direction: int) -> Relations:
        """Contract relations along with the direction of magnetic field.

        Note:
            It is assumed that the tensor A_{ijk...} is written as follows:
                A_{ijk...} = a_{ij} * H_{k} * H_{l} * H_{m} ...,
            where a_{ij} is an actual physical tensor and A_{ijk...} is a tensor introduced for computational convenience.

            This method contracts the indices of magnetic field along with one direction, like
                A_{ijkkk...} = a_{ij} * H_{k}^n.

        Args:
            relations (Relations): Relations between elements of the tensor.
            rank (int): Rank of the tensor.
            magnetic_field_direction (int): Direction of magnetic field.

        Returns:
            Relations: Contracted relations.
        """
        new_relations: Relations = []
        for relation in relations:
            flag: bool = True
            new_relation: Relation = []
            for val, ijk in relation:
                i, j, *klm = cls._ternary(ijk, rank)
                if len(klm) > 0 and not all([k == klm[0] for k in klm]): # 一軸磁場のみ考慮
                    flag = False
                    break
                else:
                    if len(klm) == 0: # 磁場依存しない(B^0に比例する)項を求めていることに注意
                        new_relation.append((val, cls._ternary2int((i,j,magnetic_field_direction))))
                    else: # 縮約してできる3階のテンソルの中で，別方向の一軸磁場での要素同士が関係を持つ可能性があるので，それは残す
                        new_relation.append((val, cls._ternary2int((i,j,klm[0]))))
            if flag:
                new_relations.append(new_relation)
        return new_relations

    @classmethod
    def _contract_nonzero_along_with_direction(cls, nonzero: set[int], rank: int, direction: int) -> set[int]:
        """Contract nonzero elements along with the direction of magnetic field.

        Note:
            It is assumed that the tensor A_{ijk...} is written as follows:
                A_{ijk...} = a_{ij} * H_{k} * H_{l} * H_{m} ...,
            where a_{ij} is an actual physical tensor and A_{ijk...} is a tensor introduced for computational convenience.

            This method contracts the indices of magnetic field along with one direction, like
                A_{ijkkk...} = a_{ij} * H_{k}^n.
        
        Args:
            nonzero (set[int]): Set of indices.
            rank (int): Rank of the tensor.
            direction (int): Direction of magnetic field.

        Returns:
            set[int]: Contracted set of indices.
        """
        # 指定した印加磁場方向の輸送テンソルの非ゼロ要素
        res: set[int] = set()
        for ijk in nonzero:
            i, j, *klm = cls._ternary(ijk, rank)
            if all([k == direction for k in klm]): # direction方向の一軸磁場に対する応答に対応する要素
                res.add(cls._ternary2int((i,j,direction)))
        return res

    @classmethod
    def _contract_along_with_direction(cls, relations: Relations, rank: int, nonzero: set[int], magnetic_field_direction: int) -> tuple[Relations, set[int], set[int], set[int]]:
        """Contract relations and nonzero elements along with the direction of magnetic field.

        Note:
            It is assumed that the tensor A_{ijk...} is written as follows:
                A_{ijk...} = a_{ij} * H_{k} * H_{l} * H_{m} ...,
            where a_{ij} is an actual physical tensor and A_{ijk...} is a tensor introduced for computational convenience.

            This method contracts the indices of magnetic field along with one direction, like
                A_{ijkkk...} = a_{ij} * H_{k}^n.
        
        Args:
            relations (Relations): Relations between elements of the tensor.
            rank (int): Rank of the tensor.
            nonzero (set[int]): Set of nonzero indices.
            magnetic_field_direction (int): Direction of magnetic field.

        Returns:
            tuple[Relations, set[int], set[int], set[int]]: Contracted relations, nonzero, 
                set of independent indices and set of dependent indices.
        """ 
        # rank 階のテンソルの磁場に対応する部分の足を縮約して，一軸磁場の2階のテンソルに直す
        n_nonzero: int = len(nonzero)
        vertex: list[int] = [ijk for ijk in sorted(nonzero)]
        ijk2idx: dict[int, int] = {ijk:i for i,ijk in enumerate(vertex)}
        G: list[dict[int, REF]] = [dict() for _ in range(n_nonzero)] # 絶対値が一致する項同士に辺を張ったグラフ
        pair_relations: Relations = sorted([relation for relation in relations if len(relation) == 2], key=lambda x:len(x))
        more_than_two_relations: Relations = sorted([relation for relation in relations if len(relation) > 2], key=lambda x:len(x))
        for relation in pair_relations:
            val0,ijk0 = relation[0]
            val1,ijk1 = relation[1]
            if ijk0 in ijk2idx and ijk1 in ijk2idx:
                idx0: int = ijk2idx[ijk0]
                idx1: int = ijk2idx[ijk1]
                G[idx0][idx1] = -val1/val0 # 辺の重みは符号が同じなら-1, 符号が逆なら+1 (ax+by=0 の x=-b/ay の-b/aの部分)
                G[idx1][idx0] = -val1/val0 # 辺の重みは符号が同じなら-1, 符号が逆なら+1 (ax+by=0 の x=-b/ay の-b/aの部分)
        
        def check_direction(ijk: int) -> bool:
            return all([i == magnetic_field_direction for i in cls._ternary(ijk, rank)[2:]])

        def check_uniaxiality(ijk: int) -> bool:
            ternary_ijk: tuple[int, ...] = cls._ternary(ijk, rank)
            if len(ternary_ijk) <= 2:
                return True
            else:
                return all([i == ternary_ijk[2] for i in ternary_ijk[2:]])

        def printer(relations: Relations) -> None:
            # デバッグ用
            print(*cls._formulize_relations(relations, rank), sep="\n")

        def printer_nonzero(nonzero: set[int], rank: int) -> None:
            # デバッグ用
            print(*sorted([cls._ternary(i,rank) for i in nonzero]), sep="\n")
        # printer(relations)
        # printer_nonzero(nonzero,rank)
        
        # 足の磁場部分が求めたい方向の一軸磁場になっている添字
        searched_vertex: list[int] = sorted([ijk for ijk in vertex if check_direction(ijk)], key=lambda x:cls._ternary(x,rank))
        searched_idx: list[int] = [ijk2idx[ijk] for ijk in searched_vertex] # 番号を振っておく
        zeros: set[int] = set() # 結果的に0になる項の添字(閉路上の辺重みの積が-1になる場合)
        connections: list[dict[int, REF]] = [dict() for _ in range(n_nonzero)] # UnionFindでいうところの，辺の根への付け替え
        used: set[int] = set()
        indep: set[int] = set()
        for start in searched_idx:
            if start in used: # もし既に訪れていたら，どれかの従属成分であることがわかる
                continue
            indep.add(vertex[start])
            visited: list[REF | None] = [None] * n_nonzero
            visited[start] = REF(3, Fraction(1,1))
            que = deque([start])
            all_zero_flag: bool = False
            while que: # ここからBFS
                idx: int = que.popleft()
                used.add(idx)
                for nex in G[idx]:
                    val_edge: REF = G[idx][nex]
                    val_nex: REF = val_edge * visited[idx]
                    if visited[nex] is None: # 未訪問の場合は更新
                        que.append(nex)
                        visited[nex] = val_nex
                    else:
                        # 符号が衝突したらある閉路上の辺重みの積が-1となり，連結成分は全てゼロだとわかる
                        if visited[nex] is not None and visited[nex] != val_nex:
                            # print("all_zero_flag")
                            # print(cls._ternary(vertex[idx],rank))
                            # print(cls._ternary(vertex[nex],rank))
                            # print(visited[nex], val_nex)
                            all_zero_flag = True
                    # 方向が別でも一軸性があればOK
                    if check_uniaxiality(vertex[nex]) and not nex in connections[start]: # 子孫(nex)を根(start)に繋ぐ
                        connections[start][nex] = val_nex
            if all_zero_flag:
                for idx in range(n_nonzero):
                    if visited[idx] is not None: # 訪問したidxは全て0になる
                        zeros.add(idx)
        
        new_relations: Relations = []
        dep: set[int] = set()
        for start in searched_idx:
            if len(connections[start]) > 0:
                for nex in connections[start]:
                    dep.add(vertex[nex])
                    new_relations.append(
                        [
                            (REF(3, Fraction(1,1)), vertex[start]),
                            (-connections[start][nex], vertex[nex]),
                        ]
                    )
        new_relations = new_relations + more_than_two_relations
        # print("indep")
        # printer_nonzero(indep,rank)
        # print("dep")
        # printer_nonzero(dep,rank)
        for ijk in nonzero:
            if not (ijk in zeros or ijk in dep):
                indep.add(ijk)
        for z in zeros:
            nonzero.discard(z)
        
        relations, nonzero, _ = cls._delete_zero_term(new_relations, nonzero)
        # print("nonzero",)
        # printer_nonzero(nonzero, 4)

        # 磁場の足1個は残してある
        contracted_relations: Relations = cls._contract_relations_along_with_direction(relations, rank, magnetic_field_direction)
        contracted_nonzero: set[int] = cls._contract_nonzero_along_with_direction(nonzero, rank, magnetic_field_direction)
        # print("c_nonzero")
        # printer_nonzero(contracted_nonzero,2)
        contracted_indep: set[int] = cls._contract_nonzero_along_with_direction(indep, rank, magnetic_field_direction)
        contracted_dep: set[int] = cls._contract_nonzero_along_with_direction(dep, rank, magnetic_field_direction)
        return contracted_relations, contracted_nonzero, contracted_indep, contracted_dep

    def get_elements_info(self, rank: int, axiality: bool, expr: str | None = None) -> None: # N階の極性テンソルがR∈self.unitary_matricesの対称操作で不変となるときの非ゼロになりうる要素の添字を計算
        """Determine which elements of a tensor are equivalent or zero by straightforward exact calculation based on Neumann's principle.

        Note:
            All analysis results are output to stdout.

        Args:
            rank (int): Rank of target physical property tensor.
            axiality (bool): True if the tensor is an axial tensor.
            expr (str | None): String representing a relation between elements that is already known. Some relations shall be separated by comma.

        Returns: 
            (list[tuple[int, ...]]): Indices(0-indexed) of non-zero elements of the tensor.

        Examples:
            >>> point_group_name = spacegroup_to_pointgroup("Fd-3m") # Pyrochlore
            >>> assert point_group_name == "m-3m"
            >>> PPTA = PhysicalPropertyTensorAnalyzer(point_group_name)
            >>> PPTA.get_elements_info(rank=4, axiality=False, expr="ijkl=ijlk=jikl=klij") # elastic modulus tensor (4-tensor): 4階の弾性率テンソル
            >>> PPTA.get_elements_info(rank=3, axiality=False, expr="ijk=ikj") # Optical Parametric Oscillator: 光パラメトリック発振
        """
        n_elem: int = 3**rank
        nonzero: set[int] = set(range(n_elem))
        relations: Relations = []
        if expr is not None: # 添字で直接表現したテンソル要素間の対称性を元に関係式を構築
            relations.extend(self._relations_from_expression(rank, expr))
        # print(self._relations_to_ternary(relations, rank))

        for R in self.unitary_matrices: # (結晶点群に属する)直交変換を元にテンソル要素間の関係式を構築
            relations.extend(self._relations_from_symmetry_operation(rank, R, axiality))

        # print("aaa",self._relations_to_ternary(relations, rank))

        def printer(relations: Relations) -> None:
            # デバッグ用
            print(*self._formulize_relations(relations, rank), sep="\n")
        
        flag: bool = True
        while flag:
            flag1: bool = True
            flag2: bool = True
            flag3: bool = True
            while flag1:
                relations, nonzero, flag1 = self._delete_zero_term(relations,nonzero) # 無効な式を削除
            relations, nonzero, flag2 = self._untangle_relations(rank,relations,nonzero) # 行簡約化で関係式を簡単化
            relations, nonzero, flag3 = self._remove_duplicate(relations,nonzero) # 重複した等価な式たちを一本化
            flag = flag1 or flag2 or flag3
            relations = self._simplify_relations_value(relations) # 関係式の係数を簡単な比に変換

        # 独立成分と従属成分を分離
        indep, dep_represented_by_indep = self._extract_independent(rank, relations, nonzero)
        res1: list[tuple[int, ...]] = sorted([self._ternary(ijk, rank) for ijk in indep])
        res2: list[str] = [self._represent_dep_by_indep(rank, relation) for relation in dep_represented_by_indep]
        
        print()
        print(f"number of nonzero elements: {len(nonzero)}")
        res0: list[tuple[int, ...]] = sorted([self._ternary(ijk, rank) for ijk in nonzero])
        print(f"nonzero elements: {res0}")
        print()
        # print(self._nonzero_matrix(nonzero,0,rank))
        # print(self._nonzero_matrix(nonzero,1,rank))
        # print(self._nonzero_matrix(nonzero,2,rank))
        print(f"number of independent elements: {len(res1)}")
        print(f"--independent elements--")
        print(*res1, sep="\n")
        print(f"------------------------")
        print()
        print(f"--dependent elements represented by indp.--")
        print(*res2, sep="\n")
        print(f"-------------------------------------------")
        return
    
    def get_elements_info_from_jahn_symbol(self, jahn_symbol: str) -> None:
        """Determine which elements of a tensor are equivalent or zero by straightforward exact calculation based on Neumann's principle.

        Note:
            All analysis results are output to stdout.

        Args:
            jahn_symbol (str): String designating the sort of tensor according to its transformation properties.

        Returns: 
            (list[tuple[int, ...]]): Indices(0-indexed) of non-zero elements of the tensor.

        Examples:
            >>> point_group_name = spacegroup_to_pointgroup("P-3c1") # Pyrochlore
            >>> assert point_group_name == "-3m"
            >>> PPTA = PhysicalPropertyTensorAnalyzer(point_group_name)
            >>> PPTA.get_elements_info_from_jahn_symbol("ae[V^2]V") # asymmetric part of thermal conductivity propotion to magnetic field linearly 
        """
        rank, expr, axiality, time_reversality = decompose_jahn_symbol(jahn_symbol)
        n_elem: int = 3**rank
        nonzero: set[int] = set(range(n_elem))
        relations: Relations = []
        # 添字で直接表現したテンソル要素間の対称性を元に関係式を構築
        relations.extend(self._relations_from_expression(rank, expr))
        # print(self._relations_to_ternary(relations, rank))

        for R in self.unitary_matrices: # (結晶点群に属する)直交変換を元にテンソル要素間の関係式を構築
            relations.extend(self._relations_from_symmetry_operation(rank, R, axiality))

        # print("aaa",self._relations_to_ternary(relations, rank))

        def printer(relations: Relations) -> None:
            # デバッグ用
            print(*self._formulize_relations(relations, rank), sep="\n")
        
        flag: bool = True
        while flag:
            flag1: bool = True
            flag2: bool = True
            flag3: bool = True
            while flag1:
                relations, nonzero, flag1 = self._delete_zero_term(relations,nonzero) # 無効な式を削除
            relations, nonzero, flag2 = self._untangle_relations(rank,relations,nonzero) # 行簡約化で関係式を簡単化
            relations, nonzero, flag3 = self._remove_duplicate(relations,nonzero) # 重複した等価な式たちを一本化
            flag = flag1 or flag2 or flag3
            relations = self._simplify_relations_value(relations) # 関係式の係数を簡単な比に変換

        # 独立成分と従属成分を分離
        indep, dep_represented_by_indep = self._extract_independent(rank, relations, nonzero)
        res1: list[tuple[int, ...]] = sorted([self._ternary(ijk, rank) for ijk in indep])
        res2: list[str] = [self._represent_dep_by_indep(rank, relation) for relation in dep_represented_by_indep]
        
        print()
        print(f"number of nonzero elements: {len(nonzero)}")
        res0: list[tuple[int, ...]] = sorted([self._ternary(ijk, rank) for ijk in nonzero])
        print(f"nonzero elements: {res0}")
        print()
        # print(self._nonzero_matrix(nonzero,0,rank))
        # print(self._nonzero_matrix(nonzero,1,rank))
        # print(self._nonzero_matrix(nonzero,2,rank))
        print(f"number of independent elements: {len(res1)}")
        print(f"--independent elements--")
        print(*res1, sep="\n")
        print(f"------------------------")
        print()
        print(f"--dependent elements represented by indp.--")
        print(*res2, sep="\n")
        print(f"-------------------------------------------")
        return


    def get_info_transport_tensor_under_magnetic_field(self, magnetic_field_dependence_dimension: int = 0, is_0indexed: bool = False) -> None: # 指定された点群上の2階の輸送テンソルの非ゼロになりうる要素の添字を計算
        """Determine which elements of a transport tensor are equivalent or zero by straightforward exact calculation based on Neumann's principle.

        Note:
            All analysis results are output to stdout.
        
        Args:
            magnetic_field_dependence_dimension (int): Nonnegative integers are allowed.
                0: constant components in magnetic field.
                1: linear components in magnetic field.
                2: quadratic components in magnetic field.
                3: cubic components in magnetic field.
                etc.
            is_0indexed (bool): True if the indices of tensor elements in the results are written in 0-indexed form. Defaults to False.

        Examples:
            >>> point_group_name = spacegroup_to_pointgroup("Fd-3m") # Diamond
            >>> assert point_group_name == "m-3m"
            >>> PPTA = PhysicalPropertyTensorAnalyzer(point_group_name)
            >>> PPTA.get_info_transport_tensor_under_magnetic_field(1) # odd terms of a transport tensor
        """
        characters: str = "ijklmnopqrstuvwxyzabcdefgh"
        rank: int
        expr: str
        axiality: bool
        if magnetic_field_dependence_dimension == 0:
            # 磁場に対してConstant
            rank = 2
            expr = ""
            axiality = False
        elif magnetic_field_dependence_dimension == 1:
            # 磁場に対してOdd
            rank = 3
            expr = "ijk=-jik"
            axiality = True
        elif magnetic_field_dependence_dimension == 2:
            # 磁場に対してEven
            rank = 4
            expr = "ijkl=jikl"
            axiality = False
        else:
            # 一般の場合
            rank = 2 + magnetic_field_dependence_dimension
            ij: str = characters[:2]
            ji: str = characters[1] + characters[0]
            klmn: str = characters[2:rank]
            sign: str = '-' if rank % 2 else ''
            expr = f"{ij+klmn}={sign}{ji+klmn}"
            axiality = (rank % 2 == 1)
            raise ValueError()
        
        def printer(relations: Relations) -> None:
            # デバッグ用
            print(*self._formulize_relations(relations, rank), sep="\n")
        
        def printer_nonzero(nonzero: set[int], rank: int) -> None:
            # デバッグ用
            print(*sorted([self._ternary(i,rank) for i in nonzero]), sep="\n")
        

        n_elem: int = 3**rank
        nonzero: set[int] = set(range(n_elem))
        relations: Relations = []
        if expr is not None: # 添字で直接表現したテンソル要素間の対称性を元に関係式を構築
            relations.extend(self._relations_from_expression(rank, expr))

        for R in self.unitary_matrices: # (結晶点群に属する)直交変換を元にテンソル要素間の関係式を構築
            # print(R.name)
            # print(self._untangle_relations(rank,self._relations_from_expression(rank, expr)+self._relations_from_symmetry_operation(rank, R, axiality),set(range(27))))
            # print(*self._formulize_relations(self._extract_relation(self._relations_from_symmetry_operation(rank, R, axiality), rank, (0,1,1)), rank), sep="\n")
            relations.extend(self._relations_from_symmetry_operation(rank, R, axiality))

        print("### initial process: done")

        flag: bool = True
        while flag:
            flag1: bool = True
            flag2: bool = True
            flag3: bool = True
            while flag1:
                relations, nonzero, flag1 = self._delete_zero_term(relations,nonzero) # 無効な式を削除
            relations, nonzero, flag2 = self._untangle_relations(rank,relations,nonzero) # 行簡約化で関係式を簡単化
            relations, nonzero, flag3 = self._remove_duplicate(relations,nonzero) # 重複した等価な式たちを一本化
            flag = flag1 or flag2 or flag3
            relations = self._simplify_relations_value(relations) # 関係式の係数を簡単な比に変換
        print("### simplification: done")
        printer(relations)
        # printer_nonzero(nonzero, rank)

        for direction0 in range(3):
            contracted_relations, contracted_nonzero, contracted_indep, contracted_dep = self._contract_along_with_direction(relations, rank, nonzero, direction0)
            contracted_relations, _, _ = self._remove_duplicate(contracted_relations, contracted_nonzero)

            result_rank: int = 3 # 足はijkで表示
            res0: list[tuple[int, ...]] = sorted([self._ternary(ijk, result_rank, is_0indexed=is_0indexed) for ijk in contracted_nonzero])
            res1: list[tuple[int, ...]] = sorted([self._ternary(ijk, result_rank, is_0indexed=is_0indexed) for ijk in contracted_indep])
            res2: list[str] = [self._represent_dep_by_indep(result_rank, relation, is_0indexed=is_0indexed) for relation in contracted_relations]
            
            direction1: int
            if not is_0indexed:
                direction1 = direction0 + 1 # 1-indexed
            else:
                direction1 = direction0 # 0-indexed
            print(f"###### magnetic field direction: B_{direction1} ######")
            print(fr"tensor element: $\rho_{{ij}} = \lambda_{{ij{direction1}}} B_{direction1}^{magnetic_field_dependence_dimension}$")
            print(f"number of nonzero elements: {len(res0)}")
            print(fr"nonzero elements of $\lambda$: {res0}")
            print(self._nonzero_matrix(nonzero,direction0,rank))
            print(f"number of independent elements: {len(res1)}")
            print(f"--independent elements--")
            print(*res1, sep="\n")
            print(f"--dependent elements represented by indp.--")
            print(*res2, sep="\n")
            print("###################################################")
            print()


        return 


class Atom():
    def __init__(
            self,
            name: str,
            site: str,
            pos: tuple[float, float, float],
            occupancy: float = 1.0,
            local_symmetry: str | None = None,
            spin: tuple[float, float, float] | None = None,
    ) -> None:
        """__init__

        Args:
            name (str): Unique name of the atom.
            site (str): Site name of the atom.
            pos (tuple[float, float, float]): Position in atomic fractional coordination.
            occupancy (float, optional): Occupancy. Defaults to 1.0.
            local_symmetry (str | None, optional): Local symmetry of the atom. The notation is described below. Defaults to None.
            spin (tuple[float, float, float] | None, optional): Spin moment in atomic fractional coordination. Defaults to None.
        
        Note:
            When `pos` = (h, k, l), the position in cartesian coordination is ha + kb + lc.
            Basic rules of the notation of `local_symmetry` follows International Crystallographic Tables vol.A.
            However, it will be ambiguous if more than one kind of symmetry operations belong to 
            the equivalent lattice symmetry directions (primary, secondary and tertiary),
            so additional terms are necessary in order to indicate the direction of symmetry operations.

            Lattice symmetry directions ([primary], [secondary], [tertiary]):
                "monoclinic-b":
                    ["010"], [], []
                "monoclinic-c":
                    ["001"], [], []
                "orthorhombic":
                    ["100"], ["010"], ["001"]
                "tetragonal":
                    ["001"], ["100", "010"], ["110", "1-10"]
                "hexagonal":
                    ["001"], ["100", "010", "-1-10"], ["1-10", "120", "-2-10"]
                "trigonal":
                    ["001"], ["100", "010", "-1-10"], ["1-10", "120", "-2-10"]
                "rhombohedral":
                    ["111"], ["1-10", "01-1", "-101"], []
                "cubic":
                    ["100", "010", "001"], ["111", "1-1-1", "-11-1", "-1-11"], ["110", "1-10", "101", "-101", "011", "01-1"]
        
            Example:
                local_symmetry = "-1" -> inversion (any)
                local_symmetry = "3/m" -> 3-fold:[001], mirror:[001] (trigonal)
                local_symmetry = ".|.|2" -> 2-fold:[001] (trigonal)
                local_symmetry = "-4|2|m" -> -4-fold:[001], 2-fold:[100,010], mirror:[110,1-10] (tetragonal)
                local_symmetry = "242|.|121111" -> 4-fold:[010], 2-fold:[100,001,1-10] (cubic)

                - Divide terms by "|" into primary, secondary and tertiary symmetry direction. \
                    If the term is simple enough to identify the operation such as "1", "-1", "3/m" and etc., \
                    you can abbreviate "|".
                - In the case of monoclinic, the lattice setting is absolutely important.
                - Use "." if no any symmetry operation in the symmetry direction.
                - If all directions in a symmetry direction have the same symmetry operation, write the symmetry operation.
                - If each direction in a symmetry direction has different symmetry operations, write the symmetry operation respectively.
                    - then, do not use "." but "1" for the symbol of no symmetry operation in each direction.
        """
        self.name: str = name
        self.site: str = site
        self.pos: npt.NDArray = np.array(pos)
        self.occupancy: float = occupancy
        self.local_symmetry: str | None = local_symmetry
        self.spin: npt.NDArray | None = None
        if spin is not None:
            self.spin = np.array(spin)
        
    def copy(self) -> Atom:
        return Atom(
            self.name,
            self.site,
            np.array(self.pos),
            self.occupancy,
            self.local_symmetry,
            np.array(self.spin) if self.spin is not None else None,
        )
    
    def properties(self) -> tuple:
        return (
            self.name,
            self.site,
            self.occupancy,
            self.local_symmetry
        )
    
    @staticmethod
    def unit_cell_norm(pos1: npt.NDArray, pos2: npt.NDArray) -> float:
        return np.linalg.norm(
            np.array(
                [min(abs(p1-p2), 1-abs(p1-p2)) for p1, p2 in zip(pos1, pos2)]
            )
        )
    
    @staticmethod
    def _arrange_array(arr: npt.NDArray, fmt: str = "#.6f") -> str:
        return '(' + ','.join([f'{v:{fmt}}' for v in arr]) + ')'

    def __eq__(self, _other: Any) -> bool:
        if isinstance(_other, Atom):
            if self.properties() == _other.properties() and self.unit_cell_norm(self.pos % 1, _other.pos % 1) < 1e-6:
                return True
            else:
                return False
        else:
            return False
    
    def __and__(self, _other: Any) -> bool:
        """exactly equal"""
        if isinstance(_other, Atom):
            if self.properties() == _other.properties() and np.linalg.norm(self.pos - _other.pos) < 1e-6:
                return True
            else:
                return False
        else:
            return False
        
    def __ne__(self, _other: Any) -> bool:
        return not self.__eq__(_other)
    
    def __hash__(self) -> int:
        return hash(self.properties())
    
    def __str__(self) -> str:
        if self.spin is None:
            return f"{self.site}({self._arrange_array(self.pos)})"
        else:
            return f"{self.site}({self._arrange_array(self.pos)}, {self._arrange_array(self.spin, fmt='#.2f')})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.name)},{repr(self.site)},"\
            f"{self._arrange_array(self.pos)},"\
            f"occupancy={repr(self.occupancy)},"\
            f"local_symmetry={repr(self.local_symmetry)},"\
            f"spin={self._arrange_array(self.spin) if self.spin is not None else None})"
    
    @classmethod
    def eval(cls, data: str) -> Atom:
        data = re.sub(r"Atom\((.+)\)", "\\1", data)
        data = re.sub(r"\((.+),(.+),(.+)\)", "(\\1$\\2$\\3)", data)
        args_kwargs: list[str] = re.split(r",\s*", data)
        args: list[str] = [s for s in args_kwargs if not "=" in s]
        args[2] = tuple(map(float, args[2].strip("()").split("$")))
        kwargs: dict[str, Any] = dict()
        for s in args_kwargs:
            if "=" in s:
                key, val = s.split("=")
                if key == "occupancy":
                    kwargs[key] = float(val)
                elif key == "local_symmetry":
                    kwargs[key] = val
                elif key == "spin":
                    kwargs[key] = tuple(map(float, s.strip("()").split("$")))
        return cls(*args, **kwargs)

    def fold(self) -> Atom:
        new_atom: Atom = self.copy()
        new_atom.pos %= 1.0
        return new_atom
    
class SpaceGroupSymmetryOperation():
    def __init__(
            self,
            crystal_structure: CrystalStructure,
            expression: str
    ) -> None:
        """__init__

        Args:
            crystal_structure (AutoCrystalStructure): crystal structure.
            expression (str): Expression of space group symmetry operation 
                written in International Crystallographic Tables vol.A.
            
            Examples:
                >>> expr = "-1;0,0,0"
                >>> expr = "3;0,0,z"
                >>> expr = "m;x,0,z"
                >>> expr = "g(2/3,1/6,2/3);2x-1/2,x,z"
                >>> expr = "-4^-;1/4,y,1/4;1/4,1/4,1/4"
                >>> expr = "3(-1/3,1/3,1/3);x+1/3,-x-1/6,-x"

            Note:
                self.matrices
                self.std_points
                self.translation
                these are matrices and vectors written in Cartesian coordination.

        """
        self.crystal_structure: CrystalStructure = crystal_structure
        self.expression: str = expression
        op, *positions = expression.strip(";").split(";")
        self.op: str = ""
        self.matrices: list[npt.NDArray] = [np.eye(3)]
        self.std_points: list[npt.NDArray] = [np.array([0.,0.,0.])]
        self.translation: npt.NDArray = np.array([0.,0.,0.])
        self.time_reversal: bool = ("'" in op)
        op = re.sub("'", "", op)

        def _check_xyz(term: str) -> bool:
            return "x" in term or "y" in term or "z" in term

        def _divide_into_term(expr: str) -> list[str]:
            return re.findall(r"[\+-]?[^\+-]+", expr)
        
        def _get_const(expr: str) -> float:
            """assume that constant is only one."""
            for term in _divide_into_term(expr):
                if not _check_xyz(term):
                    return float(Fraction(term))
            return 0.0
        
        def _get_coeff(expr: str, variable: str | None = None) -> float:
            """get a coefficient of x, y, z."""
            if not _check_xyz(expr):
                return 0.0
            if variable is None:
                variable = "x|y|z"
            for term in _divide_into_term(expr):
                if re.search(variable, term):
                    term = re.sub(variable, "", term)
                    if term == "+" or term == "":
                        return 1.0
                    elif term == "-":
                        return -1.0
                    else:
                        return float(Fraction(term))
            return 0.0

        def _get_plane_vectors(three_expr: str) -> tuple[npt.NDArray, npt.NDArray]:
            """get two vectors which generates mirror plane"""
            xyz: list[str] = three_expr.split(",")
            vecs: list[npt.NDArray] = []
            for p in ("x", "y", "z"):
                if not p in three_expr:
                    continue
                vecs.append(np.array([_get_coeff(i, variable=p) for i in xyz]))
            if not len(vecs) == 2:
                raise ValueError
            return vecs[0], vecs[1]

        def to_point(three_expr: str) -> npt.NDArray:
            """extract an immovable point whose position does not change 
                before or after the symmetry operation
            """
            point: npt.NDArray = np.array(list(map(lambda x:float(Fraction(x)), three_expr.split(","))))
            point = crystal_structure.to_xyz(point)
            return point
        
        def to_line(three_expr: str) -> tuple[npt.NDArray, npt.NDArray]:
            """extract an immovable line whose position does not change 
                before or after the symmetry operation
            """
            xyz: list[str] = three_expr.split(",")
            point: npt.NDArray = np.array([_get_const(i) for i in xyz])
            point = crystal_structure.to_xyz(point)
            direction: npt.NDArray = np.array([_get_coeff(i) for i in xyz])
            direction = crystal_structure.to_xyz(direction)
            return (point, direction)
        
        def to_plane(three_expr: str) -> tuple[npt.NDArray, npt.NDArray]:
            """extract an immovable plane whose position does not change 
                before or after the symmetry operation
            """
            xyz: list[str] = three_expr.split(",")
            point: npt.NDArray = np.array([_get_const(i) for i in xyz])
            point = crystal_structure.to_xyz(point)
            d1, d2 = _get_plane_vectors(three_expr)
            d1 = crystal_structure.to_xyz(d1)
            d2 = crystal_structure.to_xyz(d2)
            direction: npt.NDArray = np.cross(d1, d2)
            direction = crystal_structure.to_xyz(direction)
            return (point, direction)
        
        # simple translation or glide or screw
        if re.search("\(.+\)", op):
            self.translation = to_point(re.sub(r".+\((.+)\)", r"\1", op))
            op = re.sub("\(.+\)", "", op)

        if op == "1" or op == "t":
            self.matrices = [np.eye(3)]
        elif op == "-1":
            self.matrices = [-np.eye(3)]
            self.std_points = [to_point(positions[0])]
        elif op == "2" or op == "3" or op == "4" or op == "6":
            std_points, direction = to_line(positions[0])
            self.std_points = [std_points]
            self.matrices = [rotn(direction, 2*np.pi/int(op))]
        elif op == "3^-" or op == "4^-" or op == "6^-":
            std_points, direction = to_line(positions[0])
            self.std_points = [std_points]
            self.matrices = [rotn(direction, -2*np.pi/int(op.rstrip("^-")))]
        elif op == "-3" or op == "-4" or op == "-6":
            std_points, direction = to_line(positions[0])
            self.std_points = [std_points, to_point(positions[1])]
            self.matrices = [rotn(direction, 2*np.pi/(-int(op))), -np.eye(3)]
        elif op == "-3^-" or op == "-4^-" or op == "-6^-":
            std_points, direction = to_line(positions[0])
            self.std_points = [std_points, to_point(positions[1])]
            self.matrices = [rotn(direction, -2*np.pi/(-int(op.rstrip("^-")))), -np.eye(3)]
        elif op == "a" or op == "b" or op == "c":
            std_points, direction = to_plane(positions[0])
            self.std_points = [std_points]
            self.matrices = [mirror(direction)]
            self.translation += getattr(crystal_structure, op) * 1/2
        elif op == "m" or op == "n" or op == "g" or op == "d": # translation is already input
            std_points, direction = to_plane(positions[0])
            self.std_points = [std_points]
            self.matrices = [mirror(direction)]
        self.op = op
    
    def __str__(self) -> str:
        return str(self.matrices)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.crystal_structure)},{repr(self.expression)})"

    def operate(self, pos: npt.NDArray) -> npt.NDArray:
        pos = self.crystal_structure.to_xyz(pos)
        for matrix, std_point in zip(self.matrices, self.std_points):
            pos = matrix @ (pos-std_point) + std_point
        pos = pos + self.translation
        pos = self.crystal_structure.to_hkl(pos)
        return pos
    
    def operate_spin(self, spin: npt.NDArray) -> npt.NDArray:
        spin = self.crystal_structure.to_xyz(spin)
        for matrix in self.matrices:
            # the determinant of axial unitary matrix is -1.
            spin = matrix @ spin * int(np.sign(np.linalg.det(matrix)))
        if self.time_reversal:
            spin *= -1
        spin = self.crystal_structure.to_hkl(spin)
        return spin

    def generate_atom(self, atom: Atom) -> Atom:
        new_atom: Atom = atom.copy()
        new_atom.pos = self.operate(new_atom.pos)
        if new_atom.spin is not None:
            new_atom.spin = self.operate_spin(new_atom.spin)
        return new_atom
    
    @classmethod
    def point_group_operation(cls, 
            crystal_structure: CrystalStructure,
            expression: str
        ) -> SpaceGroupSymmetryOperation:
        """point_group_operation

        Args:
            crystal_structure (CrystalStructure): Instannce of `CrystalStructure`.
            expression (str): Point group symmetry operation written in the following notation.
                "{op};{crystal_direction}"
                - op: 1, 2, 3, 4, 6, -1, -3, -4, -6, m
                - crystal_direction: rotation axis or direction perpendicular to the mirror
                examples:
                    "-1": inversion. 'crystal_direction' is not necessary.
                    "3;001": 3-fold rotation along c-axis.
                    "m;100": mirror operation perpendicular to a-axis.

        Returns:
            SpaceGroupSymmetryOperation: Instannce of `SpaceGroupSymmetryOperation`.
        """
        new: SpaceGroupSymmetryOperation = cls(crystal_structure, "1;0,0,0")
        new.expression: str = expression
        op, crystal_direction = expression.strip(";").split(";")
        new.time_reversal: bool = ("'" in op)
        op = re.sub("'", "", op)
        direction: npt.NDArray = crystal_structure.direction_parser(crystal_direction)
        new.matrices: list[npt.NDArray] = []
        if op == "1":
            new.matrices.append(np.eye(3))
        elif op == "-1":
            new.matrices.append(-np.eye(3))
        elif op == "2" or op == "3" or op == "4" or op == "6":
            new.matrices.append(rotn(direction, 2*np.pi/int(op)))
        elif op == "-3" or op == "-4" or op == "-6":
            new.matrices.append(-rotn(direction, 2*np.pi/(-int(op))))
        elif op == "m":
            new.matrices.append(mirror(direction))
        new.std_points: list[npt.NDArray] = [np.array([0.,0.,0.]) for _ in range(len(new.matrices))]
        new.translation: npt.NDArray = np.array([0.,0.,0.])
        return new

class CrystalStructure():
    def __init__(self,
            name: str,
            La: float,
            Lb: float,
            Lc: float,
            alpha: float,
            beta: float,
            gamma: float,
            space_group: str,
        ) -> None:

        self.name: str = name
        self.space_group: str = space_group
        self.atoms: dict[str, Atom] = dict()
        self.polyhedra_atoms: dict[str, list[Atom]] = dict()

        self.La: float = La
        self.Lb: float = Lb
        self.Lc: float = Lc
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma
        ca: float = np.cos(np.deg2rad(alpha))
        cb: float = np.cos(np.deg2rad(beta))
        cc: float = np.cos(np.deg2rad(gamma))
        sc: float = np.sin(np.deg2rad(gamma))
        ndigits: int = 15
        self.ea: npt.NDArray = np.array([1., 0., 0.])
        self.eb: npt.NDArray = np.array([round(cc, ndigits=ndigits), round(sc, ndigits=ndigits), 0])
        self.ec: npt.NDArray = np.array(
            [
                round(cb, ndigits=ndigits),
                round((ca-cb*cc) / sc, ndigits=ndigits),
                round(np.sqrt(1 - ca**2-cb**2-cc**2 + 2*ca*cb*cc) / sc, ndigits=ndigits)
            ]
        )
        self.a: npt.NDArray = La * self.ea
        self.b: npt.NDArray = Lb * self.eb
        self.c: npt.NDArray = Lc * self.ec
        self.V: float = self.a @ np.cross(self.b, self.c)
        self.astar: npt.NDArray = np.cross(self.b, self.c) / self.V
        self.bstar: npt.NDArray = np.cross(self.c, self.a) / self.V
        self.cstar: npt.NDArray = np.cross(self.a, self.b) / self.V
        self.eastar: npt.NDArray = self.astar / np.linalg.norm(self.astar)
        self.ebstar: npt.NDArray = self.bstar / np.linalg.norm(self.bstar)
        self.ecstar: npt.NDArray = self.cstar / np.linalg.norm(self.cstar)
        self.abc: npt.NDArray = np.array([self.a, self.b, self.c]).transpose()
        self.abcstar: npt.NDArray = np.array([self.astar, self.bstar, self.cstar])
        self.eabc: npt.NDArray = np.array([self.ea, self.eb, self.ec]).transpose()

        self.trigonal_rhombohedral: bool = False
        self.monoclinic_unique_axis: str | None = None
    
    @staticmethod
    def _array_to_tsv(arr: npt.NDArray, fmt: str = " #.7g") -> str:
        return "\t".join([f"{v:{fmt}}" for v in arr])
    
    @staticmethod
    def _array_to_csv(arr: npt.NDArray, fmt: str = " #.7g") -> str:
        return ",".join([f"{v:{fmt}}" for v in arr])
    
    def set_monoclinic_unique_axis(self, axis_name: str) -> None:
        assert axis_name == "b" or axis_name == "c"
        self.monoclinic_unique_axis = axis_name

    def switch_using_rhombohedral_axes(self, flag: bool) -> None:
        self.trigonal_rhombohedral = flag

    def direction_parser(self, direction: str) -> npt.NDArray:
        """parse direction string.

        Args:
            direction (str): String represented a direction.

        Note:
            `direction` must follow some rules as below:
                1. "a", "b", "c", "astar", "bstar", "cstar" are allowed.
                2. Indices (h, k, l) and (h^*, k^*, l^*) are allowed.
        
        Examples:
            examples of `direction`
                `astar` // a*
                `"1-10"` // a - b
                `"-1*1*-2"` // -a* + b* - 2c

        Returns:
            npt.NDArray: Unit vector of the direction.
        """
        if direction in ["a", "b", "c", "astar", "bstar", "cstar"]:
            return getattr(self, "e"+direction)
        if direction in ["a*", "b*", "c*"]:
            direction = direction[0] + "star"
            return getattr(self, "e"+direction)
        mat = re.match("(-?\d\*?)(-?\d\*?)(-?\d\*?)", direction)
        if mat:
            sa, sb, sc = mat.groups()
            res: npt.NDArray = np.array([0.,0.,0.])
            if "*" in sa[-1]:
                res += self.eastar * float(sa[:-1])
            else:
                res += self.ea * float(sa)
            if "*" in sb[-1]:
                res += self.ebstar * float(sb[:-1])
            else:
                res += self.eb * float(sb)
            if "*" in sc[-1]:
                res += self.ecstar * float(sc[:-1])
            else:
                res += self.ec * float(sc)
            return res / np.linalg.norm(res)
        else:
            raise ValueError
    
    def crystal_direction(self, xyz: npt.NDArray) -> tuple[int, int, int]:
        max_denominator: int = 1000000
        aa, bb, cc = xyz@self.astar, xyz@self.bstar, xyz@self.cstar
        sa, sb, sc = int(np.sign(aa)), int(np.sign(bb)), int(np.sign(cc))
        hk: Fraction
        kl: Fraction
        lh: Fraction
        if aa == 0 and bb == 0 and cc == 0:
            return 0, 0, 0
        elif aa * bb == 0 and bb * cc == 0 and cc * aa == 0:
            if aa != 0: return sa * 1, 0, 0
            if bb != 0: return 0, sb * 1, 0
            if cc != 0: return 0, 0, sc * 1
        elif aa * bb * cc == 0:
            if aa == 0:
                kl = Fraction(bb / cc).limit_denominator(max_denominator=max_denominator)
                h, k, l = 0, sb * abs(kl.numerator), sc * abs(kl.denominator)
                return h, k, l
            if bb == 0:
                lh = Fraction(cc / aa).limit_denominator(max_denominator=max_denominator)
                k, l, h = 0, sc * abs(lh.numerator), sa * abs(lh.denominator)
                return h, k, l
            if cc == 0:
                hk = Fraction(aa / bb).limit_denominator(max_denominator=max_denominator)
                l, h, k = 0, sa * abs(hk.numerator), sb * (hk.denominator)
                return h, k, l
        else:
            hk = Fraction(aa / bb).limit_denominator(max_denominator=max_denominator)
            kl = Fraction(bb / cc).limit_denominator(max_denominator=max_denominator)
            k: int = sb * abs(lcm(hk.denominator, kl.numerator))
            h: int = sa * abs(hk.numerator * k // hk.denominator)
            l: int = sc * abs(kl.denominator * k // kl.numerator)
            return h, k, l
        
    def crystal_direction_str(self, xyz: npt.NDArray) -> str:
        h, k, l = self.crystal_direction(xyz)
        return f"{h}{k}{l}"

    @classmethod
    def from_cif(cls) -> CrystalStructure:
        raise NotImplementedError
        return

    @staticmethod
    def unit_vector(theta: float, phi: float) -> npt.NDArray:
        x: float = np.sin(theta) * np.cos(phi)
        y: float = np.sin(theta) * np.sin(phi)
        z: float = np.cos(theta)
        return np.array([x, y, z]).T
    
    @staticmethod
    def unit_vector_2d(phi: float) -> npt.NDArray:
        x: float = np.cos(phi)
        y: float = np.sin(phi)
        return np.array([x, y]).T
    
    def to_unit_xyz(self, fractional_pos: npt.NDArray) -> npt.NDArray:
        return self.eabc @ fractional_pos
    
    def to_xyz(self, fractional_pos: npt.NDArray) -> npt.NDArray:
        return self.abc @ fractional_pos
    
    def to_xyz_many(self, hkl_list: list[npt.NDArray]) -> list[npt.NDArray]:
        return [self.to_xyz(hkl) for hkl in hkl_list]
    
    def to_hkl(self, cartesian_pos: npt.NDArray) -> npt.NDArray:
        return self.abcstar @ cartesian_pos
    
    def to_hkl_many(self, xyz_list: list[npt.NDArray]) -> list[npt.NDArray]:
        return [self.to_hkl(xyz) for xyz in xyz_list]
    
    @staticmethod
    def xyz_to_thetaphi(xyz: npt.NDArray, is_degree: bool = False) -> npt.NDArray:
        """xyz coordinate to polar coordinate.

        Args:
            xyz (npt.NDArray): 3D vector.
            is_degree (bool, optional): True if the result should be in degree unit. Defaults to False.

        Returns:
            npt.NDArray: (theta, phi)
        """
        r: float = np.linalg.norm(xyz)
        if r == 0:
            return (0, 0)
        x, y, z = xyz / r
        theta: float = np.arccos(z)
        phi: float = np.arccos(x/np.sin(theta)) * (-1 if y < 0 else 1)
        if phi < 0:
            phi += 2 * np.pi
        if is_degree:
            theta = np.rad2deg(theta)
            phi = np.rad2deg(phi)
        return (theta, phi)

    def generate_atoms(
            self,
            generator_operations: list[str],
            atoms_data: list[tuple[Atom, int]],
            is_equate_equivalent_atoms: bool = True
    ) -> dict[str, set[Atom]]:
        """Generate atoms by symmetry operations from inequivalent atoms.

        Args:
            generator_operations (list[str]):
                Generator symmetry operations of the space group.
            atoms_data (list[tuple[Atom, int]]): 
                List of (Atom, multiplicity).
            is_equate_equivalent_atoms (bool):
                If True, equivalent atoms are equated even if the positions of the atoms are different by primitive translation vectors.

        Returns:
            dict[str, set[Atom]]: Atoms.
        """
        symmetry_operations: list[SpaceGroupSymmetryOperation] = [
            SpaceGroupSymmetryOperation(self, expression) for expression in generator_operations
        ]
        inequivalent_atoms: list[str] = [atom.site for atom, m in atoms_data]
        multiplicity: dict[str, int] = {}
        atoms: dict[str, list[Atom]] = {}
        for atom, m in atoms_data:
            if atom.site in multiplicity:
                multiplicity[atom.site] += m
                atoms[atom.site].append(atom)
            else:
                multiplicity[atom.site] = m
                atoms[atom.site] = [atom]

        def _check_terminate(atoms: dict[str, list[Atom]]):
            atoms_set: dict[str, set[Atom]] = {}
            for site in inequivalent_atoms:
                new_set: set[Atom] = set()
                for atom in atoms[site]:
                    if is_equate_equivalent_atoms:
                        new_set.add(atom.fold())
                    else:
                        new_set.add(atom.copy())
                atoms_set[site] = new_set
            for site in inequivalent_atoms:
                if multiplicity[site] != len(atoms_set[site]):
                    return True
            return False
        
        cnt: int = 0
        while _check_terminate(atoms):
            if cnt > 5:
                raise TimeoutError("infinite loop is detected. check the symmetry operations.")
            for site in inequivalent_atoms:
                new_list: list[Atom] = []
                for atom in atoms[site]:
                    new_list.append(atom)
                    for op in symmetry_operations:
                        new_list.append(op.generate_atom(atom))
                atoms[site] = new_list
            cnt += 1
        if is_equate_equivalent_atoms:
            atoms_set: dict[str, set[Atom]] = {
                site: set([atom.fold() for atom in atoms[site]]) for site in inequivalent_atoms
            }
        else:
            atoms_set: dict[str, set[Atom]] = {
                site: set([atom.copy() for atom in atoms[site]]) for site in inequivalent_atoms
            }

        self.unit_cell: dict[str, set[Atom]] = atoms_set
        return atoms_set
    
    def generate_atoms_list(
            self,
            generator_operations: list[str],
            atoms_data: list[tuple[Atom, int]],
            is_equate_equivalent_atoms: bool = True
        ) -> list[Atom]:
        atoms_set: dict[str, set[Atom]] = self.generate_atoms(
            generator_operations,
            atoms_data,
            is_equate_equivalent_atoms
        )
        atoms_list: list[Atom] = []
        for atoms in atoms_set.values():
            atoms_list = atoms_list + list(atoms)
        return atoms_list

    def set_atoms(self, atoms: dict[str, Atom]) -> None:
        """Set atoms.

        Args:
            atoms (dict[str, Atom]): Dictionary of (unique_name, Atom).
        """
        self.atoms.update(atoms)
    
    def set_polyhedra(self, name: str, atoms: list[Atom]) -> None:
        """Set polyhedra.

        Args:
            atoms (dict[str, list[Atom]]): Dictionary of (unique_name, atoms forming polyhedra).
        """
        self.polyhedra_atoms[name] = atoms

    def get_relative_vector_to_polyhedra(self, name: str) -> list[npt.NDArray]:
        center_atom: Atom = self.atoms[name]
        res: list[npt.NDArray] = [
            self.to_xyz(vertex_atom.pos - center_atom.pos) for vertex_atom in self.polyhedra_atoms[name]
        ]
        return res
    
    def get_relative_vector_to_polyhedra_dict(self) -> dict[str, npt.NDArray]:
        return {name: self.get_relative_vector_to_polyhedra(name) for name in self.polyhedra_atoms}
    

    def write_spins_to_vesta(self, filename_vesta: str, boundary: list[tuple[float, float]] | None = None) -> None:
        if boundary is None:
            boundary = [(0,1), (0,1), (0,1)]
        def bound_loop(pos: npt.NDArray, boundary: list[tuple[float, float]]) -> bool:
            (x0,x1), (y0,y1), (z0,z1) = boundary
            x,y,z = pos
            return product(
                range(int(np.ceil(x-x1)), int(np.floor(x-x0))+1),
                range(int(np.ceil(y-y1)), int(np.floor(y-y0))+1),
                range(int(np.ceil(z-z1)), int(np.floor(z-z0))+1),
            )

        with open(filename_vesta, mode="r") as f:
            full_contents: list[str] = f.readlines()
        
        symop: list[tuple[npt.NDArray, npt.NDArray]] = []
        structure: list[tuple[str, str, float, npt.NDArray, str, str, npt.NDArray, float]] = []
        site_texture: list[tuple[str, float, tuple[int, int, int], tuple[int, int, int, int]]] = []
        vectors: list[str] = []
        vectors_texture: list[str] = []
        VECTR_start_idx: int = -1
        VECTR_end_idx: int = -1
        VECTT_start_idx: int = -1
        VECTT_end_idx: int = -1
        idx: int = 0
        while idx < len(full_contents):
            line: str = full_contents[idx].rstrip()
            if line == "SYMOP":
                # da db dc w11 w12 w13 w21 w22 w23 w31 w32 w33 1
                # (x', y', z')^T = W @ (x, y, z)^T + dr
                idx += 1
                while full_contents[idx][0] == " ":
                    da, db, dc, *mat = full_contents[idx][1:].split()
                    dr = np.array([float(da), float(db), float(dc)])
                    mat = list(map(float, mat))
                    matrix = np.array([mat[:3],mat[3:6],mat[6:9]])
                    symop.append((dr, matrix))
                    idx += 1
                continue
            if line == "STRUC":
                idx += 1
                while full_contents[idx][0] == " ":
                    if all([s == "0" for s in full_contents[idx].split()]):
                        idx += 1
                        break
                    num, atom, atom2, occ, h, k, l, wyck, site_sym = full_contents[idx].split()
                    Ua, Ub, Uc, valence = full_contents[idx+1].split()
                    structure.append(
                        (atom, atom2, float(occ), np.array([float(h), float(k), float(l)]),
                        wyck, site_sym, np.array([float(Ua), float(Ub), float(Uc)]), float(valence))
                    )
                    idx += 2
                continue
            if line == "SITET":
                idx += 1
                while full_contents[idx][0] == " ":
                    if all([s == "0" for s in full_contents[idx].split()]):
                        idx += 1
                        break
                    num, atom2, radius, r, g, b, rp, gp, bp, ap, zero = full_contents[idx].split()
                    site_texture.append(
                        (atom2, float(radius), (int(r), int(g), int(b)), (int(rp), int(gp), int(bp), int(ap)))
                    )
                    idx += 1
                continue
            if line == "VECTR":
                VECTR_start_idx = idx
                idx += 1
                while True:
                    if full_contents[idx][0] == " ":
                        VECTR_end_idx = idx
                        idx += 1
                        break
                    vectors.append(
                        full_contents[idx]
                    )
                    idx += 1
                continue
            if line == "VECTT":
                VECTT_start_idx = idx
                idx += 1
                while full_contents[idx][0] == " ":
                    if all([s == "0" for s in full_contents[idx].split()]):
                        VECTT_end_idx = idx
                        idx += 1
                        break
                    vectors.append(
                        full_contents[idx]
                    )
                    idx += 1
                continue
            idx += 1
        
        # 非等価サイトから対称操作で生成されるすべての原子
        all_atoms_generated: list[tuple[int, int, npt.NDArray]] = []
        for i in range(len(structure)):
            for j, (dr, mat) in enumerate(symop):
                pos: npt.NDArray = mat @ structure[i][3] + dr
                all_atoms_generated.append((i, j, pos))
        
        vector_radius: float = 0.300 # [angstrom]
        for site in self.unit_cell:
            for atom in self.unit_cell[site]:
                if atom.spin is None:
                    continue
                # print(site, atom, atom.spin)
                r: npt.NDArray = atom.pos # スピンを持つ原子の位置
                for i, j, pos in all_atoms_generated:
                    if np.linalg.norm(pos % 1 - r % 1) < 1e-3: # 並進操作で重なるか
                        for nxyz in bound_loop(r, boundary): # 境界内の全等価原子を走査
                            dr = r-pos-np.array(nxyz) # 対称操作後に必要な並進操作
                            spin: npt.NDArray = np.array(atom.spin) / 2 # サイズ調整
                            k: int = len(vectors) + 1
                            vectors.append(
                                f"\t{k}  {spin[0]:.5f} {spin[1]:.5f} {spin[2]:.5f} 0\n\t\t{i+1} {j+1} {dr[0]:.0f} {dr[1]:.0f} {dr[2]:.0f}\n\t0 0 0 0 0\n"
                            )
                            rr, gg, bb = site_texture[i][2]
                            vectors_texture.append(
                                f"\t{k}  {vector_radius} {rr} {gg} {bb} 1\n"
                            )
        # 新しいvestaファイルに生成したスピンの書き込み
        full_contents[VECTT_start_idx+1:VECTT_end_idx] = vectors_texture
        full_contents[VECTR_start_idx+1:VECTR_end_idx] = vectors
        new_filename: str = re.sub(r"\.vesta", "", filename_vesta) + "_new" + ".vesta"
        with open(new_filename, mode="w") as f:
            f.write("".join(full_contents))
    
    def site_symmetry_parser(self, site_symmetry: str) -> list[npt.NDArray]:
        symmetry_directions: tuple[list[str]] = spacegroup_to_symmetry_directions(self.space_group, self.monoclinic_unique_axis, self.trigonal_rhombohedral)
        operations: list[SpaceGroupSymmetryOperation] = []
        if site_symmetry == "1":
            operations = [SpaceGroupSymmetryOperation(self, f"1;0,0,0")]
        elif site_symmetry == "-1":
            operations = [SpaceGroupSymmetryOperation(self, f"-1;0,0,0")]
        else:
            for directions, term in zip(symmetry_directions, site_symmetry.split(r"|")):
                if term == ".":
                    continue
                else:
                    ops: list[str] = [g.group() for g in re.finditer(r"m|(\d/m)|(-\d)|\d", term)]
                    if len(ops) == 1:
                        ops = [ops[0]] * len(directions)
                    for op, direction in zip(ops, directions):
                        for elem in op.split("/"):
                            if elem == "1":
                                continue
                            operations.append(SpaceGroupSymmetryOperation.point_group_operation(self, f"{elem};{direction}"))
        matrices: list[npt.NDArray] = [symop.matrices[0] for symop in operations]
        return generate_point_group(matrices)

class SpinStructure(CrystalStructure):
    def __init__(self,
            name: str,
            La: float,
            Lb: float,
            Lc: float,
            alpha: float,
            beta: float,
            gamma: float,
            space_group: str,
        ) -> None:
        super().__init__(name, La, Lb, Lc, alpha, beta, gamma, space_group)
        self.spin_lengths: dict[str, float] = dict()
        self.spin_angles: dict[str, tuple[float, float]] = dict()
        self.spin_interactions: list[tuple[str, Any, str, str | None]] = []
        self.spin_site_symop: dict[str, list[npt.NDArray]] = dict()
    
    def __len__(self) -> int:
        return len(self.spin_angles)
    
    def set_atoms(self, atoms: dict[str, Atom]) -> None:
        """Set atoms.

        Args:
            atoms (dict[str, Atom]): Dictionary of (unique_name, Atom).
        """
        self.atoms.update(atoms)
        for name, atom in atoms.items():
            self.spin_lengths[name] = np.linalg.norm(self.to_unit_xyz(atom.spin))
            if atom.local_symmetry is not None:
                self.set_spin_site_symop(name, self.site_symmetry_parser(atom.local_symmetry))
    
    def set_spin(self, name: str, spin_hkl: npt.NDArray) -> None:
        spin_xyz: npt.NDArray = self.to_unit_xyz(spin_hkl)
        self.atoms[name].spin = spin_xyz
        self.spin_lengths[name] = np.abs(spin_xyz)
        self.spin_angles[name] = self.xyz_to_thetaphi(spin_xyz)
    
    def set_spin_length(self, name: str, spin_length: float) -> None:
        self.spin_lengths[name] = spin_length
    
    def set_spin_length_dict(self, spin_length_dict: dict[str, float]) -> None:
        self.spin_lengths.update(spin_length_dict)
    
    def set_spin_angle(self, name: str, spin_angle: tuple[float, float]) -> None:
        self.spin_angles[name] = spin_angle
    
    def set_spin_angle_dict(self, spin_angle_dict: dict[str, tuple[float, float]]) -> None:
        self.spin_angles.update(spin_angle_dict)
    
    def set_spin_site_symop(self, name: str, symop_list: list[npt.NDArray]) -> None:
        self.spin_site_symop[name] = symop_list
    
    def set_spin_site_symop_dict(self, site_symop_dict: dict[str, list[npt.NDArray]]) -> None:
        self.spin_site_symop.update(site_symop_dict)
    
    def get_spin(self, name: str) -> npt.NDArray:
        return self.to_xyz(self.atoms[name].spin)
    
    def get_spins(self) -> list[npt.NDArray]:
        return [self.get_spin(name) for name in self.atoms]
    
    def get_magnetization(self) -> npt.NDArray:
        return sum(self.get_spins)
    
    def calc_magnetization(self, names: list[str], angles: list[tuple[float, float]]) -> npt.NDArray:
        return sum([self.spin_lengths[name] * self.unit_vector(theta, phi) for name,(theta,phi) in zip(names, angles)])

    def calc_spin(self, name: str, angle: tuple[float, float]) -> npt.NDArray:
        return self.spin_lengths[name] * self.unit_vector(*angle)

    def get_DM_vector_superexchange(self, name1: str, name2: str) -> npt.NDArray:
        r1: npt.NDArray = self.to_xyz(self.atoms[name1].pos)
        r2: npt.NDArray = self.to_xyz(self.atoms[name2].pos)
        ligands1: list[Atom] = self.polyhedra_atoms[name1]
        ligands2: list[Atom] = self.polyhedra_atoms[name2]
        for atom1 in ligands1:
            for atom2 in ligands2:
                if atom1 & atom2:
                    r0: npt.NDArray = self.to_xyz(atom1.pos)
                    ri: npt.NDArray = r0 - r1
                    rj: npt.NDArray = r0 - r2
                    return np.cross(ri, rj) / np.linalg.norm(np.cross(ri, rj))
        return np.array([0.,0.,0.])
    
    def set_interaction(self, kind: str, val: Any, name1: str, name2: str | None = None) -> None:
        if kind == "SIA" or kind == "DOM":
            val = (val[0], val[1] / np.linalg.norm(val[1]))
        self.spin_interactions.append((kind, val, name1, name2))
    
    def set_interaction_list(self, interaction_list: list[tuple[str, Any, str, str | None]]) -> None:
        for (kind, val, name1, name2) in interaction_list:
            self.set_interaction(kind, val, name1, name2)
    
    def calc_interaction(self, names: list[str], angles: list[tuple[float, float]], B: npt.NDArray) -> float:
        res: float = 0.0
        spins: dict[str, npt.NDArray] = {
            name: self.spin_lengths[name] * self.unit_vector(theta, phi) for name,(theta,phi) in zip(names, angles)
        }
        for kind, val, name1, name2 in self.spin_interactions:
            if kind == "EXI":
                res += val * (spins[name1] @ spins[name2])
            elif kind == "SIA":
                strength, direction = val
                res += strength * (spins[name1] @ direction) ** 2
            elif kind == "DMI":
                for symop in self.spin_site_symop[name1]:
                    res += val * (symop @ self.get_DM_vector_superexchange(name1, name2)) @ np.cross(spins[name1], spins[name2])
                # D = np.array([0.,0.,0.])
                # for symop in self.spin_site_symop[name1]:
                #     D += symop @ self.get_DM_vector_superexchange(name1, name2)
                # print(D)
                # res += val * self.get_DM_vector_superexchange(name1, name2) @ np.cross(spins[name1], spins[name2])
            elif kind == "DOM":
                strength, direction = val
                res += sum([strength * (spins[name] @ direction) ** 2 for name in names])
            elif kind == "ZMI":
                res += -val * (spins[name1] @ B)
            else:
                raise ValueError
        return res    
    
    @staticmethod
    def _array_to_tsv(arr: npt.NDArray, fmt: str = " #.7g") -> str:
        return "\t".join([f"{v:{fmt}}" for v in arr])
    
    def _to_str(self) -> str:
        fmt: str = " #.7g"
        contents: list[str] = []

        contents.append("NAME")
        contents.append(self.name)
        contents.append("NAME_END")

        contents.append("LATTICE")
        contents.append(f"\t{self.La:{fmt}}")
        contents.append(f"\t{self.Lb:{fmt}}")
        contents.append(f"\t{self.Lc:{fmt}}")
        contents.append(f"\t{self.alpha:{fmt}}")
        contents.append(f"\t{self.beta:{fmt}}")
        contents.append(f"\t{self.gamma:{fmt}}")
        contents.append("LATTICE_END")

        contents.append("SPACE_GROUP")
        contents.append(f"\t{self.space_group}")
        contents.append("SPACE_GROUP_END")

        contents.append("ATOMS")
        for name, atom in sorted(self.atoms.items()):
            contents.append(f"\t{name}\t{repr(atom)}")
        contents.append("ATOMS_END")

        contents.append("POLYH")
        for name, atoms_list in sorted(self.polyhedra_atoms.items()):
            contents.append(f"\t{name}")
            for atom in atoms_list:
                contents.append(f"\t\t{repr(atom)}")
        contents.append("POLYH_END")

        contents.append("SPIN_LENGTH")
        for name, length in sorted(self.spin_lengths.items()):
            contents.append(f"\t{name}\t{length:{fmt}}")
        contents.append("SPIN_LENGTH_END")
    
        contents.append("SPIN_ANGLE")
        for name, (theta, phi) in sorted(self.spin_angles.items()):
            contents.append(f"\t{name}\t{theta:{fmt}}\t{phi:{fmt}}")
        contents.append("SPIN_ANGLE_END")

        contents.append("INTERACTION")
        for kind, val, name1, name2 in sorted(self.spin_interactions, key=lambda x:x[0]):
            if kind == "EXI" or kind == "DMI" or kind == "ZMI":
                contents.append(f"\t{kind}\t{val:{fmt}}\t{name1}\t{name2}")
            elif kind == "SIA" or kind == "DOM":
                contents.append(f"\t{kind}\t{val[0]:{fmt}}\t{self._array_to_tsv(val[1], fmt)}\t{name1}\t{name2}")
            else:
                raise ValueError
        contents.append("INTERACTION_END")

        return "\n".join(contents)

    def save(self, filename: str | None = None) -> None:
        extension: str = ".tsv"
        if filename is None:
            filename = self.name + extension
        if os.path.splitext(filename)[1] != extension:
            filename = filename + extension

        with open(filename, mode="w") as f:
            f.write(self._to_str())

    @classmethod
    def _load(cls, lines: list[str]) -> SpinStructure:
        idx: int = 0
        while idx < len(lines):
            line: str = lines[idx].rstrip()
            line = re.sub("#.*", "", line)
            if line == "NAME":
                idx += 1
                name: str = lines[idx].rstrip()
                idx += 1
                continue
            if line == "LATTICE":
                idx += 1
                La: float = float(lines[idx].rstrip())
                Lb: float = float(lines[idx+1].rstrip())
                Lc: float = float(lines[idx+2].rstrip())
                alpha: float = float(lines[idx+3].rstrip())
                beta: float = float(lines[idx+4].rstrip())
                gamma: float = float(lines[idx+5].rstrip())
                idx += 6
                continue
            if line == "SPACE_GROUP":
                idx += 1
                space_group: str = lines[idx].rstrip()
                idx += 1
            idx += 1
        new: SpinStructure = cls(name, La, Lb, Lc, alpha, beta, gamma, space_group)
        
        idx: int = 0
        while idx < len(lines):
            line: str = lines[idx].rstrip()
            if line == "ATOMS":
                atoms: dict[str, Atom] = dict()
                idx += 1
                while lines[idx].rstrip() != "ATOMS_END":
                    name, atom_data = lines[idx].strip().split("\t")
                    atoms[name] = Atom.eval(atom_data)
                    idx += 1
                new.set_atoms(atoms)
                idx += 1
                continue
            if line == "POLYH":
                idx += 1
                while lines[idx].rstrip() != "POLYH_END":
                    name = lines[idx].strip()
                    idx += 1
                    atom_list = []
                    while True:
                        try:
                            atom_data = lines[idx].rstrip()
                            atom_list.append(Atom.eval(atom_data))
                            idx += 1
                        except:
                            break
                    new.set_polyhedra(name, atom_list)
                idx += 1
                continue
        
            if line == "SPIN_LENGTH":
                idx += 1
                while lines[idx].rstrip() != "SPIN_LENGTH_END":
                    name, length = lines[idx].rstrip().split()
                    new.set_spin_length(name, float(length))
                    idx += 1
                idx += 1
                continue
                
            if line == "SPIN_ANGLE":
                idx += 1
                while lines[idx].rstrip() != "SPIN_ANGLE_END":
                    name, theta, phi = lines[idx].rstrip().split()
                    new.set_spin_angle(name, (theta, phi))
                    idx += 1
                idx += 1
                continue
        
            if line == "INTERACTION":
                idx += 1
                while lines[idx].rstrip() != "INTERACTION_END":
                    kind, *values = lines[idx].rstrip().split()
                    if kind == "EXI" or kind == "DMI" or kind == "ZMI":
                        val, name1, name2 = values
                        val = float(val)
                    elif kind == "SIA" or kind == "DOM":
                        strength, x, y, z, name1, name2 = values
                        strength = float(strength)
                        direction = np.array([float(x), float(y), float(z)])
                        val = (strength, direction)
                    else:
                        raise ValueError
                    if name2 == "None":
                        name2 = None
                    new.set_interaction(kind, val, name1, name2)
                    idx += 1
                idx += 1
                continue
            idx += 1
        return new

    @classmethod
    def load(cls, filename: str) -> SpinStructure:
        with open(filename, mode="r") as f:
            full_contents: list[str] = f.readlines()
        return cls._load(full_contents)


def main() -> None:
    pass

if __name__ == "__main__":
    main()

