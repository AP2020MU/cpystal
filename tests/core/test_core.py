from __future__ import annotations

from cpystal.core.core import Crystal

def test_crystal_001():
    c = Crystal("H2O", date="20230523")
    assert c.name == "H2O"
    assert c.graphname == r"$\mathrm{H_{2}O}$"
    assert c.date == "20230523"
    assert c.formula_weight == 18.016

def test_crystal_002():
    c = Crystal("Co4Ta2O9")
    c.set_lattice_constant(5.1, 5.1, 14.7, 90, 90, 120)


