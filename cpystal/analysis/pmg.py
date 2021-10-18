from __future__ import annotations # class定義中に自己classを型ヒントとして使用するため

from collections import deque
from typing import Deque, List, Optional, Tuple

import matplotlib.pyplot as plt # type: ignore
import numpy as np

import pymatgen
from pymatgen.io.cif import CifParser
import pymatgen.analysis.diffraction.xrd

from ..core import Crystal


plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["legend.framealpha"] = 0

def compare_powder_Xray_experiment_with_calculation(experimental_data_filename: str, cif_filename: str, material: Optional[Crystal] = None) -> Tuple[plt.Figure, plt.Subplot]:
    """Compare experimental intensity data of powder X-ray diffraction with theoretical intensity distribution.

    Args:
        experimental_data_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".ras".
        cif_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".cif".
        material (Optional[Crystal]): `Crystal` instance of the measurement object.

    Returns:
        (Tuple[plt.Figure, plt.Subplot]): `plt.Figure` object and plotted `plt.Subplot` object.
    """
    # ここから実験データの読み込み
    data: List[List[float]] = []
    flag: bool = False
    with open(experimental_data_filename, encoding="shift_jis") as f:
        for line in f.readlines():
            if line.rstrip() == "*RAS_INT_START":
                flag = True
            elif line.rstrip() == "*RAS_INT_END":
                flag = False
            elif flag:
                data.append(list(map(float, line.strip().split())))

    N: int = len(data)
    two_theta: List[float] = [d[0] for d in data] # データは2θ
    theta: List[float] = [d[0]/2 for d in data] # データは2θ
    intensity: List[float] = [d[1] for d in data]
    normalized_intensity: List[float] = [d[1]/max(intensity)*100 for d in data]
    neg: List[float] = [i for i in intensity if i<=0]
    assert len(neg)==0 # 負のintensityをもつ壊れたデータがないことを確認
    
    neighbor_num: int = 20 # peak(極大値の中でも急激に増加するもの)判定で参照する近傍のデータ点数
    magnification: int = 4 # 周囲neighbor_num個の強度の最小値に比べて何倍大きければpeakと見なすかの閾値
    
    half: int = neighbor_num//2 # 中間点
    que: Deque[float] = deque([])
    peak: List[Tuple[float, int, float, float]] = []
    now: float = 0.0
    for i in range(N):
        que.append(intensity[i])
        now += intensity[i]
        if len(que) > neighbor_num:
            now -= que.popleft()
        else: # 最初の neighbor_num//2 個は判定しない
            continue
        if max(que) == intensity[i-half]: # 極大性判定
            # 近傍の(自分を除いた)平均値に対する比を元にピークを求める
            peak.append((intensity[i-half]/(now-intensity[i-half]),i-half,theta[i-half],intensity[i-half]))

    peak.sort(key=lambda x:x[0],reverse=True)

    Cu_K_alpha: float = 1.5418 # angstrom
    #Cu_K_alpha1 = 1.5405 # angstrom
    #Cu_K_alpha2 = 1.5443 # angstrom
    Cu_K_beta: float = 1.392 # angstrom
    display_num: int = 10
    for i, (_, p, theta_p, intensity_p) in enumerate(peak):
        if i == display_num:
            break
        d_hkl_over_n_alpha: float = Cu_K_alpha/np.sin(np.radians(theta_p))/2
        d_hkl_over_n_beta: float = Cu_K_beta/np.sin(np.radians(theta_p))/2
        print(f"θ = {theta_p}, 2θ = {2*theta_p}, intensity = {intensity_p}")
        print(f"    Kα: d_hkl/n = {d_hkl_over_n_alpha}")
        #print(f"    Kβ: d_hkl/n = {d_hkl_over_n_beta}")
        
    # ここから粉末X線回折の理論計算
    parser: pymatgen.io.cif.CifParser = CifParser(cif_filename)
    structure: pymatgen.core.structure.Structure = parser.get_structures()[0]
    analyzer: pymatgen.analysis.diffraction.xrd.XRDCalculator = pymatgen.analysis.diffraction.xrd.XRDCalculator(wavelength='CuKa')
    diffraction_pattern: pymatgen.analysis.diffraction.core.DiffractionPattern = analyzer.get_pattern(structure)
    for d_hkl, hkl, x, y in sorted(zip(diffraction_pattern.d_hkls, diffraction_pattern.hkls, diffraction_pattern.x, diffraction_pattern.y), key=lambda z:z[3], reverse=True)[:10]:
        print(x, hkl, d_hkl)

    fig: plt.Figure = plt.figure(figsize=(7,6))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    # 理論計算
    ax.bar(diffraction_pattern.x, diffraction_pattern.y, width=0.6, label="calculated", color="red")

    # 実験データ
    ax.plot(two_theta, normalized_intensity, label="experiment", color="blue")
    
    #ax.set_yscale('log')
    ax.set_xlabel(r"$2\theta\, [{}^{\circ}]$")
    ax.set_ylabel("intensity [a.u.]")
    if material is not None:
        ax.set_title(f"{material.graphname} powder X-ray diffraction")
    else:
        ax.set_title(f"powder X-ray diffraction")
    ax.legend()
    ax.set_xticks(range(0,100,10))
    plt.show()
    return fig, ax


def make_powder_Xray_diffraction_pattern_in_calculation(cif_filename: str, material: Optional[Crystal] = None) -> Tuple[plt.Figure, plt.Subplot]:
    """Calculate theoretical intensity distribution of powder X-ray diffraction.
    
    Args:
        cif_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".cif".
        material (Optional[Crystal]): `Crystal` instance of the measurement object.
    
    Returns:
        (Tuple[plt.Figure, plt.Subplot]): `plt.Figure` object and plotted `plt.Subplot` object.
    """
    try:
        parser: pymatgen.io.cif.CifParser = CifParser(cif_filename)
    except FileNotFoundError:
        raise FileNotFoundError("confirm current directory or use absolute path")
    structure: pymatgen.core.structure.Structure = parser.get_structures()[0]
    analyzer: pymatgen.analysis.diffraction.xrd.XRDCalculator = pymatgen.analysis.diffraction.xrd.XRDCalculator(wavelength='CuKa')
    diffraction_pattern: pymatgen.analysis.diffraction.core.DiffractionPattern = analyzer.get_pattern(structure)
    for d_hkl, hkl, x, y in sorted(zip(diffraction_pattern.d_hkls, diffraction_pattern.hkls, diffraction_pattern.x, diffraction_pattern.y), key=lambda z:z[3], reverse=True)[:10]:
        print(x, hkl, d_hkl)

    fig: plt.Figure = plt.figure(figsize=(7,6))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    # 理論計算
    ax.bar(diffraction_pattern.x, diffraction_pattern.y, width=0.6, label="calculated", color="red")

    #ax.set_yscale('log')
    ax.set_xlabel(r"$2\theta\, [{}^{\circ}]$")
    ax.set_ylabel("intensity [a.u.]")
    if material is not None:
        ax.set_title(f"{material.graphname} powder X-ray diffraction")
    else:
        ax.set_title(f"powder X-ray diffraction")
    ax.legend()
    ax.set_xticks(range(0,100,10))
    plt.show()
    return fig, ax


def Crystal_instance_from_cif_data(cif_filename: str) -> Crystal:
    """Generate a `Crystal` instance from a ".cif" file.

    Args:
        cif_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".cif".

    Returns:
        (Crystal): Generated `Crystal` instance.
    """
    parser: pymatgen.io.cif.CifParser = CifParser(cif_filename)
    structure: pymatgen.core.structure.Structure = parser.get_structures()[0]
    material: Crystal = Crystal("".join(structure.formula.split()))
    material.a = structure.lattice.a # [Å]
    material.b = structure.lattice.b # [Å]
    material.c = structure.lattice.c # [Å]
    material.alpha = structure.lattice.angles[0] # [°]
    material.beta  = structure.lattice.angles[1] # [°]
    material.gamma = structure.lattice.angles[2] # [°]
    material.V = structure.lattice.volume / 10**24 # [cm^3]
    return material

def main():
    pass
    return

if __name__ == "__main__":
    main()

