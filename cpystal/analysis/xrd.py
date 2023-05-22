"""`cpystal.analysis.xrd` is a module for the analysis of x-ray diffraction by theoretical calculations and making some useful data files for numerical analysis.

Functions:
    `compare_powder_Xray_experiment_with_calculation`
        -Compare experimental intensity data of powder X-ray diffraction with theoretical intensity distribution.
    `compare_powder_Xray_experiment_with_calculation_of_some_materials`
        -Compare experimental intensity data of powder X-ray diffraction with theoretical intensity distribution of some materials.
    `make_powder_Xray_diffraction_pattern_in_calculation`
        -Calculate theoretical intensity distribution of powder X-ray diffraction.
    `atoms_position_from_p1_file`
        -Get the position of atoms in unit cell from ".p1" file.
    `make_struct_file`
        -Make a ".struct" file from ".cif" file and ".p1" file.

"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
import os
import re

import matplotlib.pyplot as plt # type: ignore
import numpy as np
import numpy.typing as npt
import pymatgen # type: ignore
from pymatgen.io.cif import CifParser # type: ignore
import pymatgen.analysis.diffraction.xrd # type: ignore
from scipy.stats import norm # type: ignore

from ..core import Crystal


plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["legend.framealpha"] = 0

def _cal_theoretical_XRD_pattern(cif_filename: str, primitive: bool, display_num: int, atomic_form_factor_based_on_ITC: bool = True) -> tuple[list[list[float]], npt.NDArray, npt.NDArray]:
    parser: pymatgen.io.cif.CifParser = CifParser(cif_filename)
    structure: pymatgen.core.structure.Structure = parser.get_structures(primitive=primitive)[0]
    analyzer: pymatgen.analysis.diffraction.xrd.XRDCalculator = pymatgen.analysis.diffraction.xrd.XRDCalculator(wavelength='CuKa')
    diffraction_pattern: pymatgen.analysis.diffraction.core.DiffractionPattern
    if atomic_form_factor_based_on_ITC:
        diffraction_pattern = analyzer.get_pattern_AP2020MU(structure)
    else:
        diffraction_pattern = analyzer.get_pattern(structure)
    tops: list[list[float]] = [[x,y] for _,_,x,y in sorted(zip(diffraction_pattern.d_hkls, diffraction_pattern.hkls, diffraction_pattern.x, diffraction_pattern.y), key=lambda z:z[3], reverse=True)[:display_num]]
    peak_info: list[float, tuple[int, int, int], float, float] = sorted(zip(diffraction_pattern.d_hkls, diffraction_pattern.hkls, diffraction_pattern.x, diffraction_pattern.y), key=lambda z:z[3], reverse=True)
    for i, (d_hkl, hkl, x, y) in enumerate(peak_info[:display_num]):
        hkl_explanations: str = ", ".join([f"{dic['hkl']}*{dic['multiplicity']}" for dic in hkl])
        print(f"{i+1}: {x:.3f}, {y:.1f}, {hkl_explanations}")

    theor_x: npt.NDArray = np.arange(0,90,0.001)
    theor_y: npt.NDArray = np.zeros_like(theor_x)
    for tx, ty in zip(diffraction_pattern.x, diffraction_pattern.y):
        theor_y[bisect_left(theor_x,tx)] = ty
    Gaussian: npt.NDArray = norm.pdf(np.arange(-1,1,0.001),0,0.05)
    Gaussian /= Gaussian[len(Gaussian)//2]
    theor_y = np.convolve(theor_y, Gaussian, mode="same")
    return theor_x, theor_y, peak_info

def _LSM_peak(X: list[float], Y: list[float], linear: bool = False) -> tuple[npt.NDArray[np.float64], float, float]:
    x: npt.NDArray[np.float64] = np.array(X)
    y: npt.NDArray[np.float64] = np.array(Y)
    x, y = x[:min(len(x),len(y))], y[:min(len(x),len(y))]
    a: float
    b: float
    if linear: # 線形関数近似
        a = x@y / (x ** 2).sum()
        return a*x, a, 0.0
    else: # 1次関数近似
        n: int = len(x)
        xs: float = np.sum(x)
        ys: float = np.sum(y)
        a = ((x@y - xs*ys/n) / (np.sum(x ** 2) - xs**2/n))
        b = (ys - a * xs)/n
        return a*x + b, a, b

def _peak_search(two_theta: list[float], intensity: list[float], neighbor_num: int) -> list[tuple[float, int, float, float]]:
    # neighbor_num: peak(極大値の中でも急激に増加するもの)判定で参照する近傍のデータ点数
    half: int = neighbor_num//2 # 中間点
    que: deque[float] = deque([])
    descending_intensity: list[tuple[float, int, float, float]] = []
    now: float = 0.0
    for i in range(len(intensity)):
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
    return descending_intensity


def _cal_multiplicity(hkl: tuple[int, int, int]) -> int:
    res: int = 1
    h, k, l = hkl
    if h != 0: res *= 2
    if k != 0: res *= 2
    if l != 0: res *= 2
    if h == k == l:
        pass
    elif h == k or k == l or l == h:
        res *= 3
    else:
        res *= 6
    return res

def _get_unique_hkl_in_same_dhkls_of_FCC(r2: int) -> tuple[int, int, int]:
    max_hkl_abs: int = int(r2**0.5)
    indice: list[tuple[int, int, int]] = []
    for h in range(max_hkl_abs+1):
        for k in range(max_hkl_abs+1):
            if h**2 + k**2 > r2:
                break
            for l in range(max_hkl_abs+1):
                if h**2 + k**2 + l**2 > r2:
                    break
                if h % 2 == k % 2 == l % 2 and h**2 + k**2 + l**2 == r2: # FCCの許容反射
                    indice.append((h,k,l))
    return sorted(indice, reverse=True)[0]

def _print_comparison_between_exp_and_cal(descending_intensity: list[tuple[float, int, float, float]], peak_info: list[float, tuple[int, int, int], float, float], display_num: int) -> None:
    Cu_K_alpha: float = 1.5418 # angstrom
    #Cu_K_alpha1 = 1.5405 # angstrom
    #Cu_K_alpha2 = 1.5443 # angstrom
    Cu_K_beta: float = 1.392 # angstrom
    for i, (_, p, two_theta_p, intensity_p) in enumerate(descending_intensity):
        if i == display_num:
            break
        d_hkl_over_n_alpha: float = Cu_K_alpha/np.sin(np.radians(two_theta_p/2))/2
        d_hkl_over_n_beta: float = Cu_K_beta/np.sin(np.radians(two_theta_p/2))/2
        print(f"{i+1}: 2θ = {two_theta_p:.3f}, intensity = {int(intensity_p)}")
    exp_tops: list[list[float]] = [[x,y] for _,_,x,y in sorted(descending_intensity[:display_num], key=lambda z:z[3], reverse=True)]
    cal_tops: list[list[float]] = [[x,y] for d_hkl, hkl, x, y in sorted(peak_info, key=lambda z:z[3], reverse=True)[:display_num]]

    less_tops: list[list[float]]
    more_tops: list[list[float]]
    lesser: str
    if len(cal_tops) < len(exp_tops):
        less_tops = cal_tops
        more_tops = exp_tops
        lesser = "cal"
    else:
        less_tops = exp_tops
        more_tops = cal_tops
        lesser = "exp"
    new_more_tops: list[list[float]] = []
    new_less_tops: list[list[float]] = []
    for lx, ly in less_tops:
        nearest_x: float = -1e10
        nearest_y: float = -1e10
        for mx, my in more_tops:
            if abs(lx-mx) < abs(lx-nearest_x):
                nearest_x = mx
                nearest_y = my
        if abs(lx-nearest_x) > 3:
            continue
        new_less_tops.append([lx, ly])
        new_more_tops.append([nearest_x, nearest_y])
    if lesser == "cal":
        cal_tops = new_less_tops
        exp_tops = new_more_tops
    elif lesser == "exp":
        exp_tops = new_less_tops
        cal_tops = new_more_tops
    else:
        raise RuntimeError

def _remove_background(two_theta: list[float], intensity: list[float], descending_intensity) -> list[float]:
    # 変化が小さい部分からバックグラウンドを求める
    background_points: list[list[float]] = [[x,y] for _,_,x,y in sorted(descending_intensity[len(descending_intensity)//5*3:], key=lambda X:X[2])]
    def depeak(arr: list[list[float]]) -> list[list[float]]:
        score: list[float] = [(arr[0][1]-arr[1][1])*2]
        for i in range(1, len(arr)-1):
            xim1,yim1 = arr[i-1]
            xi,yi = arr[i]
            xip1,yip1 = arr[i+1]
            score.append(2*yi-yim1-yip1)
        score.append((arr[-1][1]-arr[-2][1])*2)
        res: list[list[float]] = [arr[i] for i in sorted(sorted(range(len(arr)), key=lambda i:score[i])[:len(arr)//3*2])]
        return res
    background_points = depeak(background_points)
    background_x: list[float] = [two_theta[0]] + [x for x,y in background_points] + [two_theta[-1]]
    background_y: list[float] = [intensity[0]] + [y for x,y in background_points] + [intensity[-1]]
    
    # background_pointsから内挿
    def interpolate_bg(x: float) -> float:
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
        else:
            raise ValueError
    # 3次spline補間は相性が悪い
    #interpolate_bg = interp1d(background_x, background_y, kind="cubic")
    # x = np.arange(10,90)
    # y = [interpolate_bg(i) for i in x]
    # plt.plot(x,y)
    # plt.scatter(background_x,background_y)
    # plt.show()
    return [its-interpolate_bg(tht) for tht,its in zip(two_theta,intensity)]

def compare_powder_Xray_experiment_with_calculation(
        experimental_data_filename: str, 
        cif_filename: str, 
        material: Crystal | None = None, 
        unbackground: bool = False, 
        issave: bool = False, 
        primitive: bool = False,
        comparison: bool = True,
        atomic_form_factor_based_on_ITC: bool = True) -> tuple[plt.Figure, plt.Subplot]:
    """Compare experimental intensity data of powder X-ray diffraction with theoretical intensity distribution.

    Notes:
        Removing background method should be improved.
        The argument 'material' will be removed in future.

    Args:
        experimental_data_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".ras".
        cif_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".cif".
        material (Crystal | None): `Crystal` instance of the measurement object.
        unbackground (bool): If True, remove the background with piecewise linear interpolation. Defaults to False.
        issave (bool): If True, save the figure as png. Defaults to False.
        primitive (bool): If True, use the primitive unit cell insted of the conventional unit cell. Defaults to False.
        comparison (bool): If True, print camparison between experimental peaks and calculational peaks to standard output. Defaults to False.

    Returns:
        (tuple[plt.Figure, plt.Subplot]): `plt.Figure` object and plotted `plt.Subplot` object.
    """
    material = Crystal.from_cif(cif_filename)
    # ここから実験データの読み込み
    data: list[list[float]] = []
    flag: bool = False
    with open(experimental_data_filename, encoding="shift_jis") as f:
        for line in f.readlines():
            if line.rstrip() == "*RAS_INT_START":
                flag = True
            elif line.rstrip() == "*RAS_INT_END":
                break
            elif flag:
                data.append(list(map(float, line.strip().split())))

    two_theta: list[float] = [t for t,p,_ in data] # データは2θ
    intensity: list[float] = [p for t,p,_ in data]
    neighbor_num: int = 50 # peak(極大値の中でも急激に増加するもの)判定で参照する近傍のデータ点数
    descending_intensity: list[tuple[float, int, float, float]] = sorted(_peak_search(two_theta, intensity, neighbor_num), key=lambda x:x[0], reverse=True)
    display_num: int = 10
    
    if unbackground:
        intensity = _remove_background(two_theta, intensity, descending_intensity)

    theor_x, theor_y, peak_info = _cal_theoretical_XRD_pattern(cif_filename, primitive, display_num, atomic_form_factor_based_on_ITC=atomic_form_factor_based_on_ITC)

    if comparison:
        _print_comparison_between_exp_and_cal(descending_intensity, peak_info, display_num)
        
    fig: plt.Figure = plt.figure(figsize=(12,6))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.plot(two_theta, intensity, label="obs.", color="blue", marker="o", markersize=1.5, linewidth=0.5, zorder=2)
    ax.plot(theor_x, theor_y * max(intensity) / 100, linewidth=1.2, label="calc.", color="red", zorder=0)
    for _, _, x, _ in sorted(peak_info, key=lambda z:z[3], reverse=True)[:display_num]:
        ax.plot([x,x], [-8*max(intensity)/100, -5*max(intensity)/100], color="green", linewidth=1, zorder=1)
    ax.plot([x,x], [-8*max(intensity)/100, -5*max(intensity)/100], color="green", linewidth=1, label="Bragg peak", zorder=1)

    ax.set_xlabel(r"$2\theta\, [{}^{\circ}]$")
    # ax.set_ylabel("intensity [a.u.]")
    ax.set_ylabel("intensity [cps]")
    if material is not None:
        ax.set_title(f"powder XRD result compared with {material.graphname} calculation")
    else:
        ax.set_title(f"powder XRD result")
    ax.legend()
    ax.set_xticks(range(0,100,10))
    ax.set_xlim(0,90)
    ax.set_ylim(-10*max(intensity)/100,max(max(intensity),max(theor_y))*1.1)
    # ax.yaxis.set_ticklabels([]) # 目盛を削除
    plt.show()
    if issave:
        if unbackground:
            fig.savefig(f"./{os.path.splitext(os.path.basename(experimental_data_filename))[0]}_with_{material.name}_pXray_unbackground.png", transparent=True)
        else:
            fig.savefig(f"./{os.path.splitext(os.path.basename(experimental_data_filename))[0]}_with_{material.name}_pXray.png", transparent=True)
    return fig, ax

def compare_powder_Xray_experiment_with_calculation_of_some_materials(
        experimental_data_filename: str, 
        cif_filename_list: list[str], 
        unbackground: bool = False, 
        issave: bool = False,
        primitive: bool = False,
        comparison: bool = True,
        atomic_form_factor_based_on_ITC: bool = True) -> tuple[plt.Figure, plt.Subplot]:
    """Compare experimental intensity data of powder X-ray diffraction with theoretical intensity distribution of some materials.

    Notes:
        Removing background method should be improved.

    Args:
        experimental_data_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".ras".
        cif_filename_list (list[str]): list of input file name (if necessary, add file path to the head). The suffix of `filename` must be ".cif".
        unbackground (bool): If True, remove the background with piecewise linear interpolation.
        primitive (bool): If True, use the primitive unit cell insted of the conventional unit cell. Defaults to False.
        comparison (bool): If True, print camparison between experimental peaks and calculational peaks to standard output. Defaults to False.

    Returns:
        (tuple[plt.Figure, plt.Subplot]): `plt.Figure` object and plotted `plt.Subplot` object.
    """
    # ここから実験データの読み込み
    data: list[list[float]] = []
    flag: bool = False
    with open(experimental_data_filename, encoding="shift_jis") as f:
        for line in f.readlines():
            if line.rstrip() == "*RAS_INT_START":
                flag = True
            elif line.rstrip() == "*RAS_INT_END":
                break
            elif flag:
                data.append(list(map(float, line.strip().split())))

    two_theta: list[float] = [t for t,p,_ in data] # データは2θ
    intensity: list[float] = [p for t,p,_ in data]
    neighbor_num: int = 50 # peak(極大値の中でも急激に増加するもの)判定で参照する近傍のデータ点数
    descending_intensity: list[tuple[float, int, float, float]] = sorted(_peak_search(two_theta, intensity, neighbor_num), key=lambda x:x[0], reverse=True)
    display_num: int = 10
    
    if unbackground:
        intensity = _remove_background(two_theta, intensity, descending_intensity)
        
    fig: plt.Figure = plt.figure(figsize=(14,8))
    plt.subplots_adjust(wspace=0.4, hspace=0)
    axs: list[plt.Subplot] = []
    for i in range(len(cif_filename_list)):
        axs.append(fig.add_subplot(len(cif_filename_list),1,i+1))
        axs[i].xaxis.set_ticks_position('both')
        axs[i].yaxis.set_ticks_position('both')

    materials_list: list[Crystal] = []
    for num, cif_filename in enumerate(cif_filename_list):
        ax: plt.Subplot = axs[num]
        material: Crystal = Crystal.from_cif(cif_filename)
        materials_list.append(material)
        # ここから粉末X線回折の理論計算
        theor_x, theor_y, peak_info = _cal_theoretical_XRD_pattern(cif_filename, primitive, display_num, atomic_form_factor_based_on_ITC=atomic_form_factor_based_on_ITC)

        if comparison:
            print(f"####### {material.name} start #########")
            _print_comparison_between_exp_and_cal(descending_intensity, peak_info, display_num)
            print(f"####### {material.name} end #########")

        # for _, _, x, _ in sorted(peak_info, key=lambda z:z[3], reverse=True)[:display_num]:
        #     ax.plot([x,x], [-8*max(intensity)/100, -5*max(intensity)/100], color="green", linewidth=1, zorder=1)
        # ax.plot([x,x], [-8*max(intensity)/100, -5*max(intensity)/100], color="green", linewidth=1,  zorder=1)
        ax.plot(two_theta, intensity, color="blue", marker="o", markersize=1.5, linewidth=0.5, zorder=2)
        ax.plot(theor_x, theor_y * max(intensity) / 100, linewidth=1.2, label=rf"calc. {material.graphname}", color="red", zorder=3)

        ax.set_ylabel("intensity [a.u.]")
        ax.legend()
        ax.set_xticks(range(0,100,10))
        ax.set_xlim(0,90)
        ax.set_ylim(-max(intensity)*0.2, max(intensity)*1.1)
        ax.yaxis.set_ticklabels([]) # 目盛を削除
        if num != len(cif_filename_list)-1:
            ax.xaxis.set_ticklabels([]) # 目盛を削除
        else:
            ax.set_xlabel(r"$2\theta\, [{}^{\circ}]$")

    plt.show()
    if issave:
        if unbackground:
            fig.savefig(f"./pXray_unbackground_{os.path.splitext(os.path.basename(experimental_data_filename))[0]}_with_{'_'.join([material.name for material in materials_list])}.png", transparent=True)
        else:
            fig.savefig(f"./pXray_{os.path.splitext(os.path.basename(experimental_data_filename))[0]}_with_{'_'.join([material.name for material in materials_list])}.png", transparent=True)
    return fig, ax

def make_powder_Xray_diffraction_pattern_in_calculation(cif_filename: str, material: Crystal | None = None) -> tuple[plt.Figure, plt.Subplot]:
    """Calculate theoretical intensity distribution of powder X-ray diffraction.
    
    Args:
        cif_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".cif".
        material (Crystal | None): `Crystal` instance of the measurement object.
    
    Returns:
        (tuple[plt.Figure, plt.Subplot]): `plt.Figure` object and plotted `plt.Subplot` object.
    """
    try:
        parser: pymatgen.io.cif.CifParser = CifParser(cif_filename)
    except FileNotFoundError:
        raise FileNotFoundError("confirm current directory or use absolute path")
    structure: pymatgen.core.structure.Structure = parser.get_structures()[0]
    analyzer: pymatgen.analysis.diffraction.xrd.XRDCalculator = pymatgen.analysis.diffraction.xrd.XRDCalculator(wavelength='CuKa')
    diffraction_pattern: pymatgen.analysis.diffraction.core.DiffractionPattern = analyzer.get_pattern(structure)
    for d_hkl, hkl, x, y in sorted(zip(diffraction_pattern.d_hkls, diffraction_pattern.hkls, diffraction_pattern.x, diffraction_pattern.y), key=lambda z:z[3], reverse=True)[:10]: # type: ignore
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

def atoms_position_from_p1_file(p1_filename: str) -> list[list[str]]:
    """Get the position of atoms in unit cell from ".p1" file.

    Note:
        Atomic coordinates in the ".p1" file must be written as "Fractional coordinates".

    Args:
        p1_filename (str): Name of the p1 file.
    
    Returns:
        (list[list[str]]): list of [`atom_name`, `X`, `Y`, `Z`, `Occupation`]
    """
    with open(p1_filename) as f:
        lines: list[str] = f.readlines()
    idx: int = lines.index("Direct\n")
    res: list[list[str]] = []
    for i in range(idx+1,len(lines)):
        line: list[str] = lines[i].split()
        atom: str = re.sub(r"\d", "", line[3])
        XYZOccupation: list[str] = line[:3] + line[4:5]
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
    positions: list[str] = ["\t".join(line) for line in atoms_position_from_p1_file(p1_filename)]
    if material.fu_per_unit_cell is None:
            raise TypeError(f"unsupported operand type(s) for /: 'None' and 'int'\nset value 'fu_per_unit_cell'")
    res: list[str] = ["! Text format structure file",
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

