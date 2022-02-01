"""`cpystal.analysis` is a module for theoretical calculations and making some useful data files for numerical analysis.

Functions:
    `compare_powder_Xray_experiment_with_calculation`
        -Compare experimental intensity data of powder X-ray diffraction with theoretical intensity distribution.
    `make_powder_Xray_diffraction_pattern_in_calculation`
        -Calculate theoretical intensity distribution of powder X-ray diffraction.
    `Crystal_instance_from_cif_data`
        -Generate a `Crystal` instance from a ".cif" file.
    `atoms_position_from_p1_file`
        -Get the position of atoms in unit cell from ".p1" file.
    `make_struct_file`
        -Make a ".struct" file from ".cif" file and ".p1" file.

"""
from __future__ import annotations # class定義中に自己classを型ヒントとして使用するため

from bisect import bisect_left
from collections import deque
from typing import Deque, List, Optional, Tuple
import re

import matplotlib.pyplot as plt # type: ignore
import numpy as np
import pymatgen # type: ignore
from pymatgen.io.cif import CifParser # type: ignore
import pymatgen.analysis.diffraction.xrd # type: ignore
from scipy.stats import norm

from ..core import Crystal


plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["legend.framealpha"] = 0

def compare_powder_Xray_experiment_with_calculation(experimental_data_filename: str, cif_filename: str, material: Optional[Crystal] = None, unbackground: bool = False, issave: bool = False) -> Tuple[plt.Figure, plt.Subplot]:
    """Compare experimental intensity data of powder X-ray diffraction with theoretical intensity distribution.

    Notes:
        Removing background method should be improved.

    Args:
        experimental_data_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".ras".
        cif_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".cif".
        material (Optional[Crystal]): `Crystal` instance of the measurement object.
        unbackground (bool): If True, remove the background with piecewise linear interpolation.

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

    intensity: List[float] = [d[1] for d in data]
    
    neg: List[float] = [i for i in intensity if i<=0]
    assert len(neg)==0 # 負のintensityをもつ壊れたデータがないことを確認
    
    neighbor_num: int = 20 # peak(極大値の中でも急激に増加するもの)判定で参照する近傍のデータ点数
    magnification: int = 4 # 周囲neighbor_num個の強度の最小値に比べて何倍大きければpeakと見なすかの閾値
    half: int = neighbor_num//2 # 中間点
    que: Deque[float] = deque([])
    descending_intensity: List[Tuple[float, int, float, float]] = []
    now: float = 0.0
    for i in range(N):
        que.append(intensity[i])
        now += intensity[i]
        if len(que) > neighbor_num:
            now -= que.popleft()
        else: # 最初の neighbor_num//2 個は判定しない
            continue
        mid: int = i-half
        if max(que) == intensity[mid]: # 極大性判定
            # 近傍の(自分を除いた)平均値に対する比を元にピークを求める
            descending_intensity.append((intensity[mid]/(now-intensity[mid]), mid, two_theta[mid], intensity[mid]))

    descending_intensity.sort(key=lambda x:x[0],reverse=True)

    Cu_K_alpha: float = 1.5418 # angstrom
    #Cu_K_alpha1 = 1.5405 # angstrom
    #Cu_K_alpha2 = 1.5443 # angstrom
    Cu_K_beta: float = 1.392 # angstrom
    display_num: int = 10
    for i, (_, p, theta_p, intensity_p) in enumerate(descending_intensity):
        if i == display_num:
            break
        d_hkl_over_n_alpha: float = Cu_K_alpha/np.sin(np.radians(theta_p))/2
        d_hkl_over_n_beta: float = Cu_K_beta/np.sin(np.radians(theta_p))/2
        print(f"2θ = {2*theta_p:.3f}, intensity = {int(intensity_p)}")
        print(f"    Kα: d_hkl/n = {d_hkl_over_n_alpha:.2f}")
        #print(f"    Kβ: d_hkl/n = {d_hkl_over_n_beta}")

    # 変化が小さい部分からバックグラウンドを求める
    background_points: List[List[float]] = [[x,y] for _,_,x,y in sorted(descending_intensity[len(descending_intensity)//5*3:], key=lambda X:X[2])]
    def depeak(arr):
        score = [(arr[0][1]-arr[1][1])*2]
        for i in range(1, len(arr)-1):
            xim1,yim1 = arr[i-1]
            xi,yi = arr[i]
            xip1,yip1 = arr[i+1]
            score.append(2*yi-yim1-yip1)
        score.append((arr[-1][1]-arr[-2][1])*2)
        res = [arr[i] for i in sorted(sorted(range(len(arr)), key=lambda i:score[i])[:len(arr)//3*2])]
        return res
    background_points = depeak(background_points)
    background_x: List[List[float]] = [two_theta[0]] + [x for x,y in background_points] + [two_theta[-1]]
    background_y: List[List[float]] = [intensity[0]] + [y for x,y in background_points] + [intensity[-1]]
    
    # background_pointsから内挿
    def interpolate_bg(x: float):
        if x < background_points[0][0]:
            tht1,its1 = two_theta[0], intensity[0]
            tht2,its2 = background_points[0]
            return its1 + (its2-its1)/(tht2-tht1)*(x-tht1)
        if x > background_points[-1][0]:
            tht1,its1 = background_points[-1]
            tht2,its2 = two_theta[-1], intensity[-1]
            return its1 + (its2-its1)/(tht2-tht1)*(x-tht1)
        for i in range(len(background_points)-1):
            tht1,its1 = background_points[i]
            tht2,its2 = background_points[i+1]
            if tht1 <= x <= tht2:
                return its1 + (its2-its1)/(tht2-tht1)*(x-tht1)
    # 3次spline補間は相性が悪い
    #interpolate_bg = interp1d(background_x, background_y, kind="cubic")
    x = np.arange(10,90)
    y = [interpolate_bg(i) for i in x]
    plt.plot(x,y)
    plt.scatter(background_x,background_y)
    plt.show()
    if unbackground:
        intensity_unbackground: List[float] = [its-interpolate_bg(tht) for tht,its in zip(two_theta,intensity)]
    else:
        intensity_unbackground = intensity
    exp_tops: List[List[float]] = [[x,y] for _,_,x,y in sorted(descending_intensity[:display_num], key=lambda z:z[3], reverse=True)]
        
    # ここから粉末X線回折の理論計算
    parser: pymatgen.io.cif.CifParser = CifParser(cif_filename)
    structure: pymatgen.core.structure.Structure = parser.get_structures()[0]
    analyzer: pymatgen.analysis.diffraction.xrd.XRDCalculator = pymatgen.analysis.diffraction.xrd.XRDCalculator(wavelength='CuKa')
    diffraction_pattern: pymatgen.analysis.diffraction.core.DiffractionPattern = analyzer.get_pattern(structure)
    cal_tops: List[list[float]] = [[x,y] for _,_,x,y in sorted(zip(diffraction_pattern.d_hkls, diffraction_pattern.hkls, diffraction_pattern.x, diffraction_pattern.y), key=lambda z:z[3], reverse=True)[:display_num]]
    for d_hkl, hkl, x, y in sorted(zip(diffraction_pattern.d_hkls, diffraction_pattern.hkls, diffraction_pattern.x, diffraction_pattern.y), key=lambda z:z[3], reverse=True)[:display_num]:
        print(f"{x:.3f}, {hkl}, {d_hkl:.3f}")

    fig: plt.Figure = plt.figure(figsize=(12,6))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    def LSM(x, y, linear=False): # x: List, y: List
        x = np.array(x)
        y = np.array(y)
        if linear: # 線形関数近似
            a = x@y / (x ** 2).sum()
            return a*x, a, 0
        else: # 1次関数近似
            n = len(x)
            xs = np.sum(x)
            ys = np.sum(y)
            a = ((x@y - xs*ys/n) / (np.sum(x ** 2) - xs**2/n))
            b = (ys - a * xs)/n
            return a*x + b, a, b
    a: float = LSM([y for x,y in exp_tops],[y for x,y in cal_tops])[1]
    # 実験値を理論値に合わせて定数倍
    # 倍率は上位ピーク強度から最小二乗法
    normalized_intensity: List[float] = [a*x for x in intensity_unbackground]

    # 実験データ
    ax.plot(two_theta, normalized_intensity, label="obs.", color="blue", marker="o", markersize=1.5, linewidth=0.5, zorder=2)
    
    # 理論計算
    theor_x = np.arange(0,90,0.001)
    theor_y = np.zeros_like(theor_x)
    for tx, ty in zip(diffraction_pattern.x, diffraction_pattern.y):
        theor_y[bisect_left(theor_x,tx)] = ty
    Gaussian = norm.pdf(np.arange(-1,1,0.001),0,0.05)
    Gaussian /= Gaussian[len(Gaussian)//2]
    theor_y = np.convolve(theor_y, Gaussian, mode="same")
    ax.plot(theor_x, theor_y, linewidth=1.2, label="calc.", color="red", zorder=0)
    
    for x in diffraction_pattern.x:
        ax.plot([x,x], [-8,-5], color="green", linewidth=1, zorder=1)
    ax.plot([x,x], [-8,-5], color="green", linewidth=1, label="Bragg peak", zorder=1)

    ax.set_xlabel(r"$2\theta\, [{}^{\circ}]$")
    ax.set_ylabel("intensity [a.u.]")
    if material is not None:
        ax.set_title(f"{material.graphname} powder X-ray diffraction")
    else:
        ax.set_title(f"powder X-ray diffraction")
    ax.legend()
    ax.set_xticks(range(0,100,10))
    ax.set_xlim(0,90)
    ax.set_ylim(-10,max(max(normalized_intensity),max(theor_y))*1.1)
    ax.yaxis.set_ticklabels([]) # 目盛を削除
    plt.show()
    if issave:
        if unbackground:
            fig.savefig("./pXray_unbackground.png", transparent=True)
        else:
            fig.savefig("./pXray.png", transparent=True)
    return fig, ax

def _compare_powder_Xray_experiment_with_calculation(experimental_data_filename: str, cif_filename: str, material: Optional[Crystal] = None) -> Tuple[plt.Figure, plt.Subplot]:
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
    intensity: List[float] = [d[1] for d in data]
    normalized_intensity: List[float] = [d[1]/max(intensity)*100 for d in data]
    neg: List[float] = [i for i in intensity if i<=0]
    assert len(neg)==0 # 負のintensityをもつ壊れたデータがないことを確認
    
    neighbor_num: int = 20 # peak(極大値の中でも急激に増加するもの)判定で参照する近傍のデータ点数
    magnification: int = 4 # 周囲neighbor_num個の強度の最小値に比べて何倍大きければpeakと見なすかの閾値
    
    half: int = neighbor_num//2 # 中間点
    que: Deque[float] = deque([])
    descending_intensity: List[Tuple[float, int, float, float]] = []
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
            descending_intensity.append((intensity[i-half]/(now-intensity[i-half]),i-half,two_theta[i-half],intensity[i-half]))

    descending_intensity.sort(key=lambda x:x[0],reverse=True)

    Cu_K_alpha: float = 1.5418 # angstrom
    #Cu_K_alpha1 = 1.5405 # angstrom
    #Cu_K_alpha2 = 1.5443 # angstrom
    Cu_K_beta: float = 1.392 # angstrom
    display_num: int = 10
    for i, (_, p, theta_p, intensity_p) in enumerate(descending_intensity):
        if i == display_num:
            break
        d_hkl_over_n_alpha: float = Cu_K_alpha/np.sin(np.radians(theta_p))/2
        d_hkl_over_n_beta: float = Cu_K_beta/np.sin(np.radians(theta_p))/2
        print(f"2θ = {2*theta_p:.3f}, intensity = {int(intensity_p)}")
        print(f"    Kα: d_hkl/n = {d_hkl_over_n_alpha:.2f}")
        #print(f"    Kβ: d_hkl/n = {d_hkl_over_n_beta}")
        
    # ここから粉末X線回折の理論計算
    parser: pymatgen.io.cif.CifParser = CifParser(cif_filename)
    structure: pymatgen.core.structure.Structure = parser.get_structures()[0]
    analyzer: pymatgen.analysis.diffraction.xrd.XRDCalculator = pymatgen.analysis.diffraction.xrd.XRDCalculator(wavelength='CuKa')
    diffraction_pattern: pymatgen.analysis.diffraction.core.DiffractionPattern = analyzer.get_pattern(structure)
    for d_hkl, hkl, x, y in sorted(zip(diffraction_pattern.d_hkls, diffraction_pattern.hkls, diffraction_pattern.x, diffraction_pattern.y), key=lambda z:z[3], reverse=True)[:10]:
        print(f"{x:.3f}, {hkl}, {d_hkl:.3f}")

    fig: plt.Figure = plt.figure(figsize=(7,6))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    # 実験データ
    ax.plot(two_theta, normalized_intensity, label="experiment", color="blue", marker="o", markersize=2, linewidth=0.5, zorder=2)
    
    # 理論計算
    ax.bar(diffraction_pattern.x, diffraction_pattern.y, width=0.6, label="calculated", color="red", zorder=1)

    #ax.set_yscale('log')
    ax.set_xlabel(r"$2\theta\, [{}^{\circ}]$")
    ax.set_ylabel("intensity [a.u.]")
    if material is not None:
        ax.set_title(f"{material.graphname} powder X-ray diffraction")
    else:
        ax.set_title(f"powder X-ray diffraction")
    ax.legend()
    ax.set_xticks(range(10,100,10))
    ax.yaxis.set_ticklabels([]) # 目盛を削除
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

    Atomic coordinates in the ".p1" file must be written as "Fractional coordinates".
    The ".struct" file will be saved in the same directory as ".cif" file.

    Note:
        This function will be used for displaying theoretical Laue patterns in "Lauept.exe".

    Args:
        cif_filename (str): Name of the cif file.
        p1_filename (str): Name of the p1 file.
    
    Returns:
        (str): The content of saved ".struct" file.
    """
    material: Crystal = Crystal.from_cif(cif_filename)
    positions: List[str] = ["\t".join(line) for line in atoms_position_from_p1_file(p1_filename)]
    if material.fu_per_unit_cell is None:
            raise TypeError(f"unsupported operand type(s) for /: 'None' and 'int'\nset value 'fu_per_unit_cell'")
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

