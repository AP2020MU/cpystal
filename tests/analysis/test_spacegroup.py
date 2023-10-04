from __future__ import annotations

import numpy as np

from cpystal.analysis.spacegroup import (
    SPACE_GROUPS,
    POINT_GROUPS,
    AUGMENTED_POINT_GROUPS,
    ARITHMETIC_CLASSES,
    GEOMETIC_CLASSES,
    LAUE_CLASSES,
    HOLOHEDRY,
    CRYSTAL_SYSTEMS,
    LATTICE_SYSTEMS,
    CRYSTAL_FAMILIES,
    BRAVAIS_LATTICE,
    spacegroup_to_pointgroup,
    pointgroup_to_laueclass,
    pointgroup_to_crystal_system,
    spacegroup_to_bravais_lattice,
    is_polar_point_group,
    is_polar_space_group,
    is_chiral_point_group,
    is_chiral_space_group,
    is_centrosymmetric_point_group,
    is_centrosymmetric_space_group,
    is_enantiomorphic_space_group,
    is_sohncke_space_group,
    is_symmorphic_space_group,
    is_bieberbach_space_group,
    generate_point_group,
    crystal_system_to_symmetry_directions,
    spacegroup_to_symmetry_directions,
    unit_vector,
    circular_mean,
    circular_diff,
    Atom,
    SpaceGroupSymmetryOperation,
    CrystalStructure,
    SpinStructure,
    decompose_jahn_symbol,
    PhysicalPropertyTensorAnalyzer,
)

def test_spacegroup_001():
    assert len(SPACE_GROUPS) == 230
    assert len(POINT_GROUPS) == 32
    assert len(AUGMENTED_POINT_GROUPS) == 40
    assert len(ARITHMETIC_CLASSES) == 73
    assert len(GEOMETIC_CLASSES) == 32
    assert len(LAUE_CLASSES) == 11
    assert len(HOLOHEDRY) == 7
    assert len(CRYSTAL_SYSTEMS) == 7
    assert len(LATTICE_SYSTEMS) == 7
    assert len(CRYSTAL_FAMILIES) == 6
    assert len(BRAVAIS_LATTICE) == 14
    assert POINT_GROUPS == GEOMETIC_CLASSES

def test_spacegroup_002():
    assert len([s for s in SPACE_GROUPS if is_centrosymmetric_space_group(s)]) == 85
    assert len([s for s in SPACE_GROUPS if is_symmorphic_space_group(s)]) == 73
    assert len([s for s in SPACE_GROUPS if is_polar_space_group(s)]) == 68
    assert len([s for s in SPACE_GROUPS if is_sohncke_space_group(s)]) == 65
    assert len([s for s in SPACE_GROUPS if is_enantiomorphic_space_group(s)]) == 22
    assert len([s for s in SPACE_GROUPS if is_chiral_space_group(s)]) == 22
    assert len([s for s in SPACE_GROUPS if is_bieberbach_space_group(s)]) == 13

def test_spacegroup_003():
    assert len([p for p in POINT_GROUPS if is_polar_point_group(p)]) == 10
    assert len([p for p in POINT_GROUPS if is_chiral_point_group(p)]) == 11
    assert len([p for p in POINT_GROUPS if is_centrosymmetric_point_group(p)]) == 10

def test_spacegroup_004():
    assert all([spacegroup_to_pointgroup(s) in AUGMENTED_POINT_GROUPS for s in SPACE_GROUPS])
    assert all([pointgroup_to_laueclass(p) in LAUE_CLASSES for p in POINT_GROUPS])
    assert all([pointgroup_to_crystal_system(p) in CRYSTAL_SYSTEMS for p in POINT_GROUPS])
    assert all([spacegroup_to_bravais_lattice(s) in BRAVAIS_LATTICE for s in SPACE_GROUPS])

def test_spacegroup_005():
    assert np.linalg.norm(unit_vector(np.pi/12, np.pi/3)) == 1.0
    assert circular_mean(np.pi, 0.0) == np.pi/2
    assert circular_mean(np.pi/3, 2*np.pi-np.pi/3) == 0.0
    assert circular_diff(np.pi, 0.0) == np.pi
    assert abs(circular_diff(np.pi/6, np.pi+np.pi/12) - -np.pi*11/12) < 1e-10
    assert abs(circular_diff(np.pi+np.pi/12, np.pi/6) - np.pi*11/12) < 1e-10
    assert abs(circular_diff(99*np.pi+np.pi/12, np.pi/6) - np.pi*11/12) < 1e-10

def test_spacegroup_006():
    La_CNO: float = 5.1610
    Lb_CNO: float = La_CNO
    Lc_CNO: float = 14.0984

    CNO: CrystalStructure = CrystalStructure("Co4Nb2O9", La_CNO, Lb_CNO, Lc_CNO, 90, 90, 120, "P-3c1")
    generator_operations: list[str] = [
        "3;0,0,z",
        "2';0,y,1/4",
        "-1';0,0,0",
    ]
    primitive_atoms: list[Atom] = [
        (Atom("Nb", "Nb1", (0, 0, 0.142203), local_symmetry="3|.|."), 4),
        (Atom("Co", "Co1", (2/3, 1/3, 0.013473), local_symmetry="3|.|.", spin=(0.73, 2.3, 0)), 4),
        (Atom("Co", "Co2", (1/3, 2/3, 0.192906), local_symmetry="3|.|.", spin=(-0.73, 2.3, 0)), 4),
        (Atom("O", "O1", (0.65787, 0.68039, 0.084418), local_symmetry="1"), 12),
        (Atom("O", "O2", (0, 0.71233, 1/4), local_symmetry=".|2|."), 6),
    ]
    CNO.generate_atoms(generator_operations, primitive_atoms)

def test_spacegroup_007():
    P: PhysicalPropertyTensorAnalyzer = PhysicalPropertyTensorAnalyzer("-3m")
    P.get_elements_info(2, False, expr="ij=ji") # rank-2, symmetric, polar
    P.get_elements_info(3, True, expr="ijk=-jik") # # rank-3, asymmetric for ij, axial
    P.get_info_transport_tensor_under_magnetic_field(magnetic_field_dependence_dimension=1)

    P.get_elements_info_from_jahn_symbol("aeV^2") # magnetoelectric tensor
    P.get_elements_info_from_jahn_symbol("e{V^2}*V") # magnetothermal tensor
    P.get_elements_info_from_jahn_symbol("ae[V^2]V") # piezomagnetic tensor

    assert (2, '', True, True) == decompose_jahn_symbol("aeV^2")
    assert (3, "ijk=-jik'", True, False) == decompose_jahn_symbol("e{V^2}*V")
    assert (3, 'ijk=jik', True, True) == decompose_jahn_symbol("ae[V^2]V")

