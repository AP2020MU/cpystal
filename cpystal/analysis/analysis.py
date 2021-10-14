"""`cpystal`: for dealing with crystals and experimental data of physical property.

`cpystal` is designed to handle experimental data on crystals.
It places particular emphasis on calculating and storing data on the properties of actual synthesized samples, 
and on graphing these data. In this respect, it is distinct from `pymatgen`, a similar crystal and material analysis module.
Of course, pymatgen is a very useful python module, so we use it as an adjunct in `cpystal`.
"""
from __future__ import annotations # class定義中に自己classを型ヒントとして使用するため
import re
from typing import List

from ..core import Crystal


def atoms_position_from_p1_file(p1_filename: str) -> List[List[str]]:
    """Get the position of atoms in unit cell from ".p1" file.

    Note:
        Atomic coordinates in the ".p1" file must be written as "Fractional coordinates".

    Args:
        p1_filename (str): Name of the p1 file.
    
    Returns:
        (List[List[str]]): list of [`atom_name`, `X`, `Y`, `Z`, `Occupation`]
    """
    with open(p1_filename) as f:
        lines: List[str] = f.readlines()
    idx: int = lines.index("Direct\n")
    res: List[List[str]] = []
    for i in range(idx+1,len(lines)):
        line: List[str] = lines[i].split()
        atom: str = re.sub(r"\d", "", line[3])
        XYZOccupation: List[str] = line[:3] + line[4:5]
        res.append([atom, *XYZOccupation])
    return res

def make_struct_file(cif_filename: str, p1_filename: str) -> str:
    """Make a ".struct" file from ".cif" file and ".p1" file.

    Note:
        Atomic coordinates in the ".p1" file must be written as "Fractional coordinates".
        The ".struct" file will be saved in the same directory as ".cif" file.

    Args:
        cif_filename (str): Name of the cif file.
        p1_filename (str): Name of the p1 file.
    
    Returns:
        (str): The content of saved ".struct" file.
    """
    material: Crystal = Crystal.from_cif(cif_filename)
    positions: List[str] = ["\t".join(line) for line in atoms_position_from_p1_file(p1_filename)]
    res: List[str] = ["! Text format structure file",
        "",
        f"{material.name}\t\t\t! Crystal Name",
        f"{material.a}\t{material.b}\t{material.c}\t! Lattice constants a, b, c (�)",
        f"{material.alpha}\t{material.beta}\t{material.gamma}\t! alpha, beta, gamma (degree)",
        "",
        f"{material.fu_per_unit_cell * int(sum(material.components.values()))}\t! Total number of atoms in the unit cell",
        "",
        "! Atom	X	Y	Z	Occupation",
        *positions,
        "",
        "0	! Debye characteristic temperature (K)",
        "0	! Thermal expansion coefficient (10^-6/K)"
    ]
    
    with open(f"{cif_filename.replace('.cif', '')}.struct", 'w') as f:
        f.write("\n".join(res))
    return "\n".join(res)

def main() -> None:
    pass
    return

if __name__ == "__main__":
    main()

