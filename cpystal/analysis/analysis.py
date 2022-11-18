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
from __future__ import annotations

from bisect import bisect_left
from collections import deque
import os
import re
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import numpy.typing as npt
import pymatgen # type: ignore
from pymatgen.io.cif import CifParser # type: ignore
import pymatgen.analysis.diffraction.xrd # type: ignore
from scipy.optimize import curve_fit # type: ignore
from scipy.stats import norm # type: ignore
from scipy import integrate # type: ignore
import tkinter as tk


from ..core import Crystal, PhysicalConstant


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

def cal_Debye_specific_heat(T: float, TD: float, num_atom_per_formula_unit: int) -> float:
    """Calculating Debye mol specific heat.

    Args:
        T (float): Temperature (K).
        TD (float): Debye temperature (K).
        num_atom_per_formula_unit (int): Number of atom in a formula unit.
    
    Returns:
        (float): Debye mol specific heat (JK^-1mol^-1).
    """
    def fD(t: float) -> float:
        def integrand(x: float) -> float:
            return 0. if x == 0. else x**4 / np.sinh(x/2.)**2
        return 0. if t == 0. else (3/4) * t**3 * integrate.quad(integrand, 0, 1./t)[0]
    R: float = PhysicalConstant().R # 気体定数 [JK^-1mol^-1]
    return 3 * R * num_atom_per_formula_unit * fD(T/TD)

def cal_thermal_conductivity(material: Crystal, T: float, TD: float, l: float) -> float:
    """Calculating thermal conductivity of phonon based on Debye model.

    Args:
        material (Crystal): Crystal instance.
        T (float): Temperature (K).
        TD (float): Debye temperature (K).
        l (float): Mean free path (cm).
    
    Returns:
        (float): Thermal conductivity of phonon based on Debye model (WK^-1m^-1).
    """
    n: int = material.num_atom_per_formula_unit
    v: float = material.cal_phonon_velocity(TD) # cm/s
    C_mol: float = cal_Debye_specific_heat(T, TD=TD, num_atom_per_formula_unit=n) # J/K/mol
    c: float = C_mol * n / (material.V * material.NA / material.fu_per_unit_cell) # J/K/cm^3
    return 1/3 * c * v * l * 100. # W/Km

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

# ### Nx+Ny+Nz=1を満たさない，スケーリングで値が異なるなどの問題点
# def demagnetizing_factor_rectangular_prism_Nf(a: float, b: float, c: float) -> float:
#     """Calculating demagnetizing factor of rectangular prism axbxc.

#     Thesis:
#         D.-X. Chen et al., IEEE Transactions on Magnetics 38, 4 (2002).

#     Args:
#         a (float): Length of an edge (arb. unit).
#         b (float): Length of an edge (arb. unit).
#         c (float): Length of an edge (arb. unit).
    
#     Returns:
#         (float): Demagnetizing factor.
#     """
#     def Ff(u: float, v: float) -> float:
#         return u * np.log(c**2*(8*u**2+4*v**2+c**2+4*u*np.sqrt(4*u**2+4*v**2+c**2)) / (4*v**2+c**2) / (8*u**2+c**2+4*u*np.sqrt(4*u**2+c**2)))

#     F1: float = np.sqrt(4*a**2+c**2) + np.sqrt(4*b**2+c**2) - np.sqrt(4*a**2+4*b**2+c**2) - c
#     Nf: float = 2/np.pi * np.arctan(4*a*b/(c*np.sqrt(4*a**2+4*b**2+c**2))) + (c/(2*np.pi*a*b))*(F1+Ff(a,b)+Ff(b,a))
#     return Nf

# def demagnetizing_factor_rectangular_prism_Nm(a: float, b: float, c: float) -> float:
#     """Calculating demagnetizing factor of rectangular prism axbxc.

#     Thesis:
#         D.-X. Chen et al., IEEE Transactions on Magnetics 38, 4 (2002).

#     Args:
#         a (float): Length of an edge (arb. unit).
#         b (float): Length of an edge (arb. unit).
#         c (float): Length of an edge (arb. unit).
    
#     Returns:
#         (float): Demagnetizing factor.
#     """
#     def Fm(u: float, v: float, w: float) -> float:
#         return u**2*v * np.log((u**2+w**2)*(u**2+2*v**2+2*v*np.sqrt(u**2+v**2)) / (u**2) / (u**2+2*v**2+w**2+2*v*np.sqrt(u**2+v**2+w**2)))

#     F2: float = a**3 + b**3 - 2*c**3 + (a**2+b**2-2*c**2)*np.sqrt(a**2+b**2+c**2)
#     F3: float = (2*c**2-a**2)*np.sqrt(a**2 + c**2) + (2*c**2-b**2)*np.sqrt(b**2 + c**2) - np.sqrt(b**2 + c**2)**3

#     Nm: float = 2/np.pi * np.arctan(a*b/(c*np.sqrt(a**2+b**2+c**2))) + 1/(3*np.pi*a*b*c)*(F2+F3) + 1/(2*np.pi*a*b*c) * (Fm(a,b,c)+Fm(b,a,c)-Fm(c,a,b)-Fm(c,b,a))
#     return Nm

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





class RawDataExpander:
    """This is a class for extending thermal properties data measured by Labview-controlled PPMS. 
    
    Attributes:
        filename (str): Input file name of data (if necessary, add file path to the head). The suffix of `filename` may be ".txt".
        full_contents (list[str]): Full contents of the loaded file splitted into lines.
        names (list[str]): All the name of experiments.
        Time (list[float]): Time when the measurement is conducted (sec).
        PPMSTemp (list[float]): PPMS temperature (K).
        Field (list[float]): Magnetic field (Oe).
        HeaterCurrent (list[float]): Heater current (mA).
        Q (list[float]): Heater power (mW).
        Angle (list[float]): Angle (degree).
        V1 (list[list[float]]): Ch1 complex voltage [Re, Im] (V), otherwise Cernox temperature and resistivity [T (K), R (Ohm)]
        V2 (list[list[float]]): Ch2 complex voltage [Re, Im] (V).
        V3 (list[list[float]]): Ch3 complex voltage [Re, Im] (V).
        V4 (list[list[float]]): Ch4 complex voltage [Re, Im] (V).
        V5 (list[list[float]]): Ch5 complex voltage [Re, Im] (V).
        V6 (list[list[float]]): Ch6 complex voltage [Re, Im] (V).
        CernoxTemp (list[float]): Cernox temperature (K). By default, calculated from V1.
        TC_TS (list[list[float]]): list of temperature dependence of thermocouple seebeck coefficient, [temperature (K), S (V/K)].
        S_TC (list[float]): Seebeck coefficient (V/K).
        dTx (list[float]): Temperature differnce of x-direction (K). By default, calculated from V2.
        dTy (list[float]): Temperature differnce of y-direction (K), By default, calculated from V3.
    """

    def __init__(self, filename: str, filename_Seebeck: str) -> None:
        """Initializer of this class.

        Args:
            filename (str): Input file name of data (if necessary, add file path to the head). The suffix of `filename` may be ".txt".
            filename_Seebeck (str): Input file name of thermocouple S vs T data (if necessary, add file path to the head). The suffix of `filename` may be ".txt".

        Note:
            Raw data format example:
            '''
            Start time: yyyy/mm/dd hh:mm:ss
            Sample Parameter:
            Length (um) of Tx:  2200.000000
            Length (um) of Ty:   701.000000
            Length (um) of Vx:   0.000000
            Length (um) of Vy:   0.000000
            Sample Width (um):   701.000000
            Sample Thickness (um):   82.500000
            Index	Time (s)	PPMS Temperature (K)	Field (Oe)	Angle (deg)	Heater Current (mA)	Ch1 Vx (V)[or Cernox Temperature (K)]	Ch1 Vy (V)[or Cernox Resistance (K)]	Ch2 Vx (V)	Ch2 Vy (V)	Ch3 Vx (V)	Ch3 Vx (V)	Ch4 Vx (V)	Ch4 Vx (V)	Ch5 Vx (V)	Ch5 Vy (V)	Ch6 Vx (V)	Ch6 Vy (V)	
            0.0000000000E+0	1.2200832367E-2	1.2499940000E+1	-6.6000000000E-2	0.0000000000E+0	0.0000000000E+0	1.2613500000E+1	-2.6852900000E+2	1.8073239800E-6	1.8073239800E-6	1.3243287300E-6	1.3243287300E-6	0.0000000000E+0	0.0000000000E+0	0.0000000000E+0	0.0000000000E+0	0.0000000000E+0	0.0000000000E+0	
            ...
            '''
        """
        self.filename: str = filename
        with open(file=filename, mode="r") as f:
            self.full_contents: list[str] = f.readlines()

        self.TC_TS: list[list[float]] = []
        with open(file=filename_Seebeck, mode="r") as f:
            for line in f.readlines()[1:]:
                s,t = map(float, line.split())
                self.TC_TS.append([t,s])
        self.TC_TS = sorted(self.TC_TS, key=lambda x:x[0]) # 温度順にソート
        
        
        self.StartTime: str = re.sub(r".+?:", r"", self.full_contents[0]).strip()
        self.LTx: float = float(re.sub(r".+:", r"", self.full_contents[2]))
        self.LTy: float = float(re.sub(r".+:", r"", self.full_contents[3]))
        self.LVx: float = float(re.sub(r".+:", r"", self.full_contents[4]))
        self.LVy: float = float(re.sub(r".+:", r"", self.full_contents[5]))
        self.Width: float = float(re.sub(r".+:", r"", self.full_contents[6]))
        self.Thickness: float = float(re.sub(r".+:", r"", self.full_contents[7]))

        self.Time: list[float] = []
        self.PPMSTemp: list[float] = []
        self.Field: list[float] = []
        self.Angle: list[float] = []
        self.HeaterCurrent: list[float] = []
        self.V1: list[list[float]] = []
        self.V2: list[list[float]] = []
        self.V3: list[list[float]] = []
        self.V4: list[list[float]] = []
        self.V5: list[list[float]] = []
        self.V6: list[list[float]] = []
        self.Q: list[float] = []

        idx: int = 14
        while True:
            try:
                row: str = self.full_contents[idx]
                ind, ti, temp, field, angle, cur, *values = map(float, row.split())
                v1re, v1im, v2re, v2im, v3re, v3im, v4re, v4im, v5re, v5im, v6re, v6im = values
            except:
                break
            idx += 1
            self.Time.append(ti)
            self.PPMSTemp.append(temp)
            self.Field.append(field)
            self.Angle.append(angle)
            self.HeaterCurrent.append(cur)
            self.Q.append(cur**2) # 1 kΩ
            self.V1.append([v1re,v1im])
            self.V2.append([v2re,v2im])
            self.V3.append([v3re,v3im])
            self.V4.append([v4re,v4im])
            self.V5.append([v5re,v5im])
            self.V6.append([v6re,v6im])        

        self.S_TC: list[float] = [self.Seebeck_at_T(t) for t in self.PPMSTemp]
        self.dTx: list[float]
        self.dTy: list[float]
        self.dVx: list[float]
        self.dVy: list[float]
    
    def _set_CernoxTemp(self, cernox_name: str, TR: list[list[float]]) -> None:
        X173409_logRlogT_table: list[tuple[float, float]] = [(3.29314 - 1.42456*log10t + 0.867728*log10t**2 - 0.324371*log10t**3 + 0.0380185*log10t**4, log10t) for log10t in np.linspace(np.log10(1.5),np.log10(320),100000)]
        X173079_logRlogT_table: list[tuple[float, float]] = [(3.29639174 - 1.34578352*log10t + 0.79354379*log10t**2 - 0.29893059*log10t**3 + 0.03486926*log10t**4, log10t) for log10t in np.linspace(np.log10(1.5),np.log10(320),100000)]

        def binary_search_value(logRlogT_table: list[tuple[float, float]], target: float) -> float:
            log_target: float = np.log10(target)
            ok: int = 0
            ng: int = len(logRlogT_table)
            while abs(ok-ng)>1:
                mid = (ng+ok)//2
                if logRlogT_table[mid][0] >= log_target:
                    ok = mid
                else:
                    ng = mid
            if ok >= len(logRlogT_table)-1:
                ok = len(logRlogT_table)-2
            logT: float = logRlogT_table[ok][1] + (logRlogT_table[ok+1][1]-logRlogT_table[ok][1]) / (logRlogT_table[ok+1][0]-logRlogT_table[ok][0]) * (log_target-logRlogT_table[ok][0])
            return 10**logT        

        self.CernoxTemp: list[float]
        if cernox_name == "X173409":
            self.CernoxTemp = [binary_search_value(X173409_logRlogT_table, abs(resis)) for temp, resis in TR]
        elif cernox_name == "X173079":
            self.CernoxTemp = [binary_search_value(X173079_logRlogT_table, abs(resis)) for temp, resis in TR]
        else:
            raise ValueError("cernox name is invalid.")

    def kxxkxy_mode(self, cernox_name: str = "X173409", attr_cor_to_V: list[int] | None = None) -> None:
        """
        
        Args:
            cernox_name (str): Cernox name. ("X173409" or "X173079")
            attr_cor_to_V (list[int] | None): Correspondence between attributes [Cernox_Temp, dTx, dTy] and 1-indexed voltage number. Defaults to [1,2,3].

        """

        if attr_cor_to_V is None:
            attr_cor_to_V = list(range(3))
        else:
            attr_cor_to_V = [i-1 for i in attr_cor_to_V]
        Voltages: list[list[list[float]]] = [self.V1, self.V2, self.V3, self.V4, self.V5, self.V6]
        self._set_CernoxTemp(cernox_name, Voltages[attr_cor_to_V[0]])
        self.dTx = [v/s for (v,_), s in zip(Voltages[attr_cor_to_V[1]], self.S_TC)]
        self.dTy = [v/s for (v,_), s in zip(Voltages[attr_cor_to_V[2]], self.S_TC)]

    def SxxSxy_mode(self, cernox_name: str = "X173409", attr_cor_to_V: list[int] | None = None) -> None:
        """
        
        Args:
            cernox_name (str): Cernox name. ("X173409" or "X173079")
            attr_cor_to_V (list[int] | None): Correspondence between attributes [Cernox_Temp, dTx, dVx, dVy] and 1-indexed voltage number. Defaults to [1,2,3].

        """
        if attr_cor_to_V is None:
            attr_cor_to_V = list(range(4))
        else:
            attr_cor_to_V = [i-1 for i in attr_cor_to_V]
        Voltages: list[list[list[float]]] = [self.V1, self.V2, self.V3, self.V4, self.V5, self.V6]
        self._set_CernoxTemp(cernox_name, Voltages[attr_cor_to_V[0]])
        self.dTx = [v/s for (v,_), s in zip(Voltages[attr_cor_to_V[1]], self.S_TC)]
        self.dVx = [v for (v,_) in Voltages[attr_cor_to_V[2]]]
        self.dVy = [v for (v,_) in Voltages[attr_cor_to_V[3]]]

    def Seebeck_at_T(self, T: float) -> float:
        if not (self.TC_TS[0][0] <= T <= self.TC_TS[-1][0]):
            return 0
        S_at_T: float = -1
        for i in range(len(self.TC_TS)-1):
            if self.TC_TS[i][0] <= T <= self.TC_TS[i+1][0]:
                S_at_T = self.TC_TS[i][1] + (self.TC_TS[i+1][1]-self.TC_TS[i][1]) / (self.TC_TS[i+1][0]-self.TC_TS[i][0]) * (T-self.TC_TS[i][0])
                break
        if S_at_T == -1:
            raise RuntimeError
        return S_at_T

    def graph_CenoxTemp_vs_Time(self, title: str = "") -> None:
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams["legend.framealpha"] = 0
        plt.rcParams['legend.fontsize'] = 8

        figsize: tuple[int, int] = (7,8)
        fig: plt.Figure = plt.figure(figsize=figsize)
        plt.subplots_adjust(wspace=0.2, hspace=0)

        ax: plt.Subplot = fig.add_subplot(111)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.plot(self.Time, self.CernoxTemp, marker='o', color="blue")
        ax.set_xlabel(xlabel=r"Time (sec)")
        ax.set_ylabel(ylabel=r"Cernox Temperature (K)")
        #ax.legend()
        #ax.set_xlim((0,max(self.PPMSTemp)))
        ax.set_title(title)
        plt.show()

    def graph_PPMSTemp_hist(self, time1: float, time2: float, bins: int) -> None:
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams["legend.framealpha"] = 0
        plt.rcParams['legend.fontsize'] = 8

        figsize: tuple[int, int] = (7,8)
        fig: plt.Figure = plt.figure(figsize=figsize)
        plt.subplots_adjust(wspace=0.2, hspace=0)

        ax: plt.Subplot = fig.add_subplot(111)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.get_xaxis().get_major_formatter().set_useOffset(True)

        idx1: int = bisect_left(self.Time, time1)
        idx2: int = bisect_left(self.Time, time2)
        offset: float = sum(self.CernoxTemp[idx1:idx2])/len(self.CernoxTemp[idx1:idx2])
        ax.hist([val-offset for val in self.CernoxTemp[idx1:idx2]], bins=bins, color="orange", ec='black', clip_on=False, rwidth=1.0)
        ax.set_xlabel(xlabel=r"Cernox Temperature (K)")
        #ax.legend()
        #ax.set_xlim((0,max(self.PPMSTemp)))
        ax.set_title(fr"t = {int(self.Time[idx1])}s ~ {int(self.Time[idx2])}s (count:{idx2-idx1+1})\
            $T_{{ave}}={offset:.6g}\pm{np.std(self.CernoxTemp[idx1:idx2]):.2g}$ K")
        ax.text(0.92, -0.07, r"+$T_{ave}$", fontsize=13, transform = ax.transAxes)
        plt.show()
    
    def graph_four_data_all(self) -> None:
        """
        0 < t < end での全データを一堂に会する
        """

        fig: plt.Figure = plt.figure(figsize=(5,10))
        plt.subplots_adjust(wspace=0.2, hspace=0)

        ax1: plt.Subplot = fig.add_subplot(411)
        ax1.xaxis.set_ticks_position('both')
        ax1.yaxis.set_ticks_position('both')
        ax2: plt.Subplot = fig.add_subplot(412)
        ax2.xaxis.set_ticks_position('both')
        ax2.yaxis.set_ticks_position('both')
        ax3: plt.Subplot = fig.add_subplot(413)
        ax3.xaxis.set_ticks_position('both')
        ax3.yaxis.set_ticks_position('both')
        ax4: plt.Subplot = fig.add_subplot(414)
        ax4.xaxis.set_ticks_position('both')
        ax4.yaxis.set_ticks_position('both')

        #ax1.plot(self.Time, self.Field, marker='o', color="blue", markersize=2)
        ax1.plot(self.Time, self.CernoxTemp, marker='o', color="orange", markersize=2)
        #ax1.set_xlabel(xlabel=r"Time (sec)")
        ax1.set_ylabel(ylabel=r"$T_{\mathrm{Cernox}}$ (K)")
        ax1.xaxis.set_ticklabels([]) # 目盛を削除
        
        ax2.plot(self.Time, self.dTx, marker='o', color="green", markersize=2)
        #ax2.set_xlabel(xlabel=r"Time (sec)")
        ax2.set_ylabel(ylabel=r"$\Delta T_{x}$ (K)")
        ax2.xaxis.set_ticklabels([]) # 目盛を削除

        ax3.plot(self.Time, self.dTy, marker='o', color="blue", markersize=2)
        #ax3.set_xlabel(xlabel=r"Time (sec)")
        ax3.set_ylabel(ylabel=r"$\Delta T_{y}$ (K)")
        ax3.xaxis.set_ticklabels([]) # 目盛を削除

        ax4.plot(self.Time, self.Q, marker='o', color="red", markersize=2)
        ax4.set_xlabel(xlabel=r"Time (sec)")
        ax4.set_ylabel(ylabel=r"$Q$ (mW)")

        plt.show()
        fig.savefig(f"./fig/four_data_all.png", bbox_inches="tight", transparent=True)     

    def graph_four_data(self, t1: float, t2: float) -> None:
        """
        t1 < t < t2 での全データを一堂に会する
        """
        idx1: int = bisect_left(self.Time, t1)
        idx2: int = bisect_left(self.Time, t2)
        fig: plt.Figure = plt.figure(figsize=(5,10))
        plt.subplots_adjust(wspace=0.2, hspace=0)

        ax1: plt.Subplot = fig.add_subplot(411)
        ax1.xaxis.set_ticks_position('both')
        ax1.yaxis.set_ticks_position('both')
        ax2: plt.Subplot = fig.add_subplot(412)
        ax2.xaxis.set_ticks_position('both')
        ax2.yaxis.set_ticks_position('both')
        ax3: plt.Subplot = fig.add_subplot(413)
        ax3.xaxis.set_ticks_position('both')
        ax3.yaxis.set_ticks_position('both')
        ax4: plt.Subplot = fig.add_subplot(414)
        ax4.xaxis.set_ticks_position('both')
        ax4.yaxis.set_ticks_position('both')

        #ax1.plot(self.Time, self.Field, marker='o', color="blue", markersize=2)
        ax1.plot(self.Time[idx1:idx2], self.CernoxTemp[idx1:idx2], marker='o', color="orange", markersize=2)
        #ax1.set_xlabel(xlabel=r"Time (sec)")
        ax1.set_ylabel(ylabel=r"$T_{\mathrm{Cernox}}$ (K)")
        ax1.xaxis.set_ticklabels([]) # 目盛を削除
        
        ax2.plot(self.Time[idx1:idx2], self.dTx[idx1:idx2], marker='o', color="green", markersize=2)
        #ax2.set_xlabel(xlabel=r"Time (sec)")
        ax2.set_ylabel(ylabel=r"$\Delta T_{x}$ (K)")
        ax2.xaxis.set_ticklabels([]) # 目盛を削除

        ax3.plot(self.Time[idx1:idx2], self.dTy[idx1:idx2], marker='o', color="blue", markersize=2)
        #ax3.set_xlabel(xlabel=r"Time (sec)")
        ax3.set_ylabel(ylabel=r"$\Delta T_{y}$ (K)")
        ax3.xaxis.set_ticklabels([]) # 目盛を削除

        ax4.plot(self.Time[idx1:idx2], self.Q[idx1:idx2], marker='o', color="red", markersize=2)
        ax4.set_xlabel(xlabel=r"Time (sec)")
        ax4.set_ylabel(ylabel=r"$Q$ (mW)")

        plt.show()
        fig.savefig(f"./fig/four_data_{int(t1)}to{int(t2)}.png", bbox_inches="tight", transparent=True)     


class ExpDataExpander:
    def __init__(self, filename: str, filename_Seebeck: str) -> None:
        self.filename: str = filename
        with open(file=filename, mode="r") as f:
            self.full_contents: list[str] = f.readlines()

        self.TC_TS: list[list[float]] = []
        with open(file=filename_Seebeck, mode="r") as f:
            for line in f.readlines()[1:]:
                s,t = map(float, line.split())
                self.TC_TS.append([t,s])
        self.TC_TS = sorted(self.TC_TS, key=lambda x:x[0]) # 温度順にソート

        self.StartTime: str = re.sub(r".+?:", r"", self.full_contents[0]).strip()
        self.LTx: float = float(re.sub(r".+:", r"", self.full_contents[2]))
        self.LTy: float = float(re.sub(r".+:", r"", self.full_contents[3]))
        self.LVx: float = float(re.sub(r".+:", r"", self.full_contents[4]))
        self.LVy: float = float(re.sub(r".+:", r"", self.full_contents[5]))
        self.Width: float = float(re.sub(r".+:", r"", self.full_contents[6]))
        self.Thickness: float = float(re.sub(r".+:", r"", self.full_contents[7]))
        
        self.Index: list[tuple[int, int, int, int]] = []
        self.T_PPMS: list[float] = []
        self.Field: list[float] = []
        self.Current: list[float] = []
        self.T_Cernox: list[float] = []
        self.dVx: list[float] = []
        self.dVy: list[float] = []
        self.dTx: list[float] = []
        self.dTy: list[float] = []

        self.dTx0: list[float] = []
        self.dTy0: list[float] = []
        self.dTx1: list[float] = []
        self.dTy1: list[float] = []

        self.errdTx: list[float] = []
        self.errdTy: list[float] = []

        self.errdTx0: list[float] = []
        self.errdTy0: list[float] = []
        self.errdTx1: list[float] = []
        self.errdTy1: list[float] = []

        idx: int = 14
        while idx < len(self.full_contents):
            row: str = self.full_contents[idx]
            if row.startswith("#"):
                idx += 1
                continue
            row = re.sub(r"#.*", "", row)

            sidx0, eidx0, aveT_PPMS0, errT_PPMS0, aveH0, errH0, aveCurrent0, errCurrent0, aveT_Cernox0, errT_Cernox0, aveVx0, errVx0, aveVy0, errVy0, \
            sidx1, eidx1, aveT_PPMS1, errT_PPMS1, aveH1, errH1, aveCurrent1, errCurrent1, aveT_Cernox1, errT_Cernox1, aveVx1, errVx1, aveVy1, errVy1, dTx, errdTx, kxx, *errkxx = map(float, row.split())
            self.Index.append((int(sidx0), int(eidx0), int(sidx1), int(eidx1)))
            self.T_PPMS.append(aveT_PPMS1)
            self.Field.append(aveH1)
            self.Current.append(aveCurrent1)
            self.T_Cernox.append(aveT_Cernox1)
            self.dVx.append(aveVx1-aveVx0)
            self.dVy.append(aveVy1-aveVy0)
            self.dTx.append((aveVx1-aveVx0) / self.Seebeck_at_T(aveT_PPMS1))
            self.dTy.append((aveVy1-aveVy0) / self.Seebeck_at_T(aveT_PPMS1))

            self.dTx0.append((aveVx0) / self.Seebeck_at_T(aveT_PPMS0))
            self.dTy0.append((aveVy0) / self.Seebeck_at_T(aveT_PPMS0))
            self.dTx1.append((aveVx1) / self.Seebeck_at_T(aveT_PPMS1))
            self.dTy1.append((aveVy1) / self.Seebeck_at_T(aveT_PPMS1))
            self.errdTx0.append((errVx0) / self.Seebeck_at_T(aveT_PPMS0))
            self.errdTy0.append((errVy0) / self.Seebeck_at_T(aveT_PPMS0))
            self.errdTx1.append((errVx1) / self.Seebeck_at_T(aveT_PPMS1))
            self.errdTy1.append((errVy1) / self.Seebeck_at_T(aveT_PPMS1))

            # print(errVx0/ self.Seebeck_at_T(aveT_PPMS1), errVx1/ self.Seebeck_at_T(aveT_PPMS1), errVy0/ self.Seebeck_at_T(aveT_PPMS1), errVy1/ self.Seebeck_at_T(aveT_PPMS1), (errVx0**2+errVx1**2)**0.5 / self.Seebeck_at_T(aveT_PPMS1), (errVy0**2+errVy1**2)**0.5 / self.Seebeck_at_T(aveT_PPMS1))
            self.errdTx.append((errVx0**2+errVx1**2)**0.5 / self.Seebeck_at_T(aveT_PPMS1))
            self.errdTy.append((errVy0**2+errVy1**2)**0.5 / self.Seebeck_at_T(aveT_PPMS1))
            idx += 1

        self.R: float = 1000.
        self.kxx: list[float] = [self.R*(i**2)/self.Width/self.Thickness / (dt/self.LTx) for i,dt in zip(self.Current,self.dTx)]
        self.errkxx: list[float] = [k * edtx / dtx for k,dtx,edtx in zip(self.kxx,self.dTx,self.errdTx)]

    def Seebeck_at_T(self, T: float) -> float:
        if not (self.TC_TS[0][0] <= T <= self.TC_TS[-1][0]):
            return 0
        S_at_T: float = -1
        for i in range(len(self.TC_TS)-1):
            if self.TC_TS[i][0] <= T <= self.TC_TS[i+1][0]:
                S_at_T = self.TC_TS[i][1] + (self.TC_TS[i+1][1]-self.TC_TS[i][1]) / (self.TC_TS[i+1][0]-self.TC_TS[i][0]) * (T-self.TC_TS[i][0])
                break
        if S_at_T == -1:
            raise RuntimeError
        return S_at_T

    def symmetrize(self) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        N: int = len(self.Field)
        lamxx: list[float] = [self.Width*self.Thickness/(self.R*(i**2)) * (dt/self.LTx) for i,dt in zip(self.Current,self.dTx)]
        lamyx: list[float] = [self.Width*self.Thickness/(self.R*(i**2)) * (dt/self.LTy) for i,dt in zip(self.Current,self.dTy)]
        lamxx_err: list[float] = [self.Width*self.Thickness/(self.R*(i**2)) * (edt/self.LTx) for i,edt in zip(self.Current,self.errdTx)]
        lamyx_err: list[float] = [self.Width*self.Thickness/(self.R*(i**2)) * (edt/self.LTy) for i,edt in zip(self.Current,self.errdTy)]

        # symmetrize
        lamxx_symm: list[float] = [(lamxx[i]+lamxx[N-1-i])/2 for i in range(N)]
        lamxx_symm_err: list[float] = [np.sqrt(lamxx_err[i]**2 + lamxx_err[N-1-i]**2)/2 for i in range(N)]

        H_: list[float]
        if N % 2 == 1:
            H_ = [-h for h in self.Field[N//2+1:][::-1]] + self.Field[N//2:]
        else:
            H_ = [-h for h in self.Field[N//2:][::-1]] + self.Field[N//2:]

        lamyx_symm: list[float] = [(lamyx[i]-lamyx[N-1-i])/2 for i in range(N)]
        lamyx_symm_err: list[float] = [np.sqrt(lamyx_err[i]**2 + lamyx_err[N-1-i]**2)/2 for i in range(N)]
        kxx: list[float] = [lx / (lx**2 + ly**2) for lx,ly in zip(lamxx_symm,lamyx_symm)]
        kxy: list[float] = [ly / (lx**2 + ly**2) for lx,ly in zip(lamxx_symm,lamyx_symm)]
        kxx_err: list[float] = [np.sqrt(((x**2-y**2)*x_err)**2 + (2*x*y*y_err)**2) / ((x**2 + y**2)**2) for x,y,x_err,y_err in zip(lamxx_symm,lamyx_symm,lamxx_symm_err,lamyx_symm_err)]
        kxy_err: list[float] = [np.sqrt((2*x*y*x_err)**2 + ((x**2-y**2)*y_err)**2) / ((x**2 + y**2)**2) for x,y,x_err,y_err in zip(lamxx_symm,lamyx_symm,lamxx_symm_err,lamyx_symm_err)]
        return H_, kxx, kxy, kxx_err, kxy_err
    
    def symmetrize_positive_half(self) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        H_, kxx, kxy, kxx_err, kxy_err = self.symmetrize()
        N: int = len(H_)
        if H_[0] < 0:
            return H_[N//2:], kxx[N//2:], kxy[N//2:], kxx_err[N//2:], kxy_err[N//2:]
        else:
            return H_[:(N+1)//2][::-1], kxx[:(N+1)//2][::-1], kxy[:(N+1)//2][::-1], kxx_err[:(N+1)//2][::-1], kxy_err[:(N+1)//2][::-1]
    
    def symmetrize_dT(self) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        N: int = len(self.Field)
        dTx_symm: list[float] = [(self.dTx[i]+self.dTx[N-1-i])/2 for i in range(N)]
        dTx_symm_err: list[float] = [np.sqrt(self.errdTx[i]**2 + self.errdTx[N-1-i]**2)/2 for i in range(N)]
        
        H_: list[float]
        if N % 2 == 1:
            H_ = [-h for h in self.Field[N//2+1:][::-1]] + self.Field[N//2:]
        else:
            H_ = [-h for h in self.Field[N//2:][::-1]] + self.Field[N//2:]

        dTy_symm: list[float] = [(self.dTy[i]-self.dTy[N-1-i])/2 for i in range(N)]
        dTy_symm_err: list[float] = [np.sqrt(self.errdTy[i]**2 + self.errdTy[N-1-i]**2)/2 for i in range(N)]
        return H_, dTx_symm, dTy_symm, dTx_symm_err, dTy_symm_err


class ExpDataExpanderSeebeck:
    def __init__(self, filename: str, filename_Seebeck: str) -> None:
        self.filename: str = filename
        with open(file=filename, mode="r") as f:
            self.full_contents: list[str] = f.readlines()

        self.TC_TS: list[list[float]] = []
        with open(file=filename_Seebeck, mode="r") as f:
            for line in f.readlines()[1:]:
                s,t = map(float, line.split())
                self.TC_TS.append([t,s])
        self.TC_TS = sorted(self.TC_TS, key=lambda x:x[0]) # 温度順にソート

        self.StartTime: str = re.sub(r".+?:", r"", self.full_contents[0]).strip()
        self.LTx: float = float(re.sub(r".+:", r"", self.full_contents[2]))
        self.LTy: float = float(re.sub(r".+:", r"", self.full_contents[3]))
        self.LVx: float = float(re.sub(r".+:", r"", self.full_contents[4]))
        self.LVy: float = float(re.sub(r".+:", r"", self.full_contents[5]))
        self.Width: float = float(re.sub(r".+:", r"", self.full_contents[6]))
        self.Thickness: float = float(re.sub(r".+:", r"", self.full_contents[7]))
        
        self.Index: list[tuple[int, int, int, int]] = []
        self.T_PPMS: list[float] = []
        self.Field: list[float] = []
        self.Current: list[float] = []
        self.T_Cernox: list[float] = []
        self.dVx: list[float] = []
        self.dVy: list[float] = []
        self.dTx: list[float] = []

        self.dTx0: list[float] = []
        self.dTx1: list[float] = []

        self.errdTx: list[float] = []

        self.errdTx0: list[float] = []
        self.errdTx1: list[float] = []

        self.Ex: list[float] = []
        self.Ey: list[float] = []

        self.Ex0: list[float] = []
        self.Ey0: list[float] = []
        self.Ex1: list[float] = []
        self.Ey1: list[float] = []

        self.errEx: list[float] = []
        self.errEy: list[float] = []

        self.errEx0: list[float] = []
        self.errEy0: list[float] = []
        self.errEx1: list[float] = []
        self.errEy1: list[float] = []

        self.Sxx: list[float] = []
        self.errSxx: list[float] = []


        idx: int = 14
        while idx < len(self.full_contents):
            row: str = self.full_contents[idx]
            if row.startswith("#"):
                idx += 1
                continue
            row = re.sub(r"#.*", "", row)

            sidx0, eidx0, aveT_PPMS0, errT_PPMS0, aveH0, errH0, aveCurrent0, errCurrent0, aveT_Cernox0, errT_Cernox0, aveVx0, errVx0, aveEx0, errEx0, aveEy0, errEy0, \
            sidx1, eidx1, aveT_PPMS1, errT_PPMS1, aveH1, errH1, aveCurrent1, errCurrent1, aveT_Cernox1, errT_Cernox1, aveVx1, errVx1, aveEx1, errEx1, aveEy1, errEy1, _dTx, _errdTx, _Sxx, _errSxx, _Sxy = map(float, row.split())
            self.Index.append((int(sidx0), int(eidx0), int(sidx1), int(eidx1)))
            self.T_PPMS.append(aveT_PPMS1)
            self.Field.append(aveH1)
            self.Current.append(aveCurrent1)
            self.T_Cernox.append(aveT_Cernox1)
            self.dVx.append(aveVx1-aveVx0)
            dTx: float = (aveVx1-aveVx0) / self.Seebeck_at_T(aveT_PPMS1)
            errdTx: float = (errVx0**2+errVx1**2)**0.5 / self.Seebeck_at_T(aveT_PPMS1)
            self.dTx.append(dTx)
            self.errdTx.append(errdTx)
            self.dTx0.append((aveVx0) / self.Seebeck_at_T(aveT_PPMS0))
            self.dTx1.append((aveVx1) / self.Seebeck_at_T(aveT_PPMS1))
            self.errdTx0.append((errVx0) / self.Seebeck_at_T(aveT_PPMS0))
            self.errdTx1.append((errVx1) / self.Seebeck_at_T(aveT_PPMS1))

            Ex: float = aveEx1-aveEx0
            Ey: float = aveEy1-aveEy0
            errEx: float = (errEx0**2+errEx1**2)**0.5
            errEy: float = (errEy0**2+errEy1**2)**0.5
            self.Ex.append(Ex)
            self.Ey.append(Ey)
            self.errEx.append(errEx)
            self.errEy.append(errEy)

            self.Ex0.append(aveEx0)
            self.Ex1.append(aveEx1)
            self.Ey0.append(aveEy0)
            self.Ey1.append(aveEy1)
            self.errEx0.append(errEx0)
            self.errEx1.append(errEx1)
            self.errEy0.append(errEy0)
            self.errEy1.append(errEy1)
            
            # self.Sxx.append(Sxx)
            # self.errSxx.append(errSxx)
            Sxx: float = Ex / dTx / self.LVx * self.LTx * 1e6 # (uV/K)
            self.Sxx.append(Sxx)
            self.errSxx.append(abs(Sxx) * ((errEx/Ex)**2 + (errdTx/dTx)**2)**0.5)
            idx += 1

        self.R: float = 1000.
        self.kxx: list[float] = [self.R*(i**2)/self.Width/self.Thickness / (dt/self.LTx) for i,dt in zip(self.Current,self.dTx)]
        self.errkxx: list[float] = [k * edtx / dtx for k,dtx,edtx in zip(self.kxx,self.dTx,self.errdTx)]

    def Seebeck_at_T(self, T: float) -> float:
        if not (self.TC_TS[0][0] <= T <= self.TC_TS[-1][0]):
            return 0
        S_at_T: float = -1
        for i in range(len(self.TC_TS)-1):
            if self.TC_TS[i][0] <= T <= self.TC_TS[i+1][0]:
                S_at_T = self.TC_TS[i][1] + (self.TC_TS[i+1][1]-self.TC_TS[i][1]) / (self.TC_TS[i+1][0]-self.TC_TS[i][0]) * (T-self.TC_TS[i][0])
                break
        if S_at_T == -1:
            raise RuntimeError
        return S_at_T

    def symmetrize(self) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
        N: int = len(self.Field)

        # symmetrize
        kxx_symm: list[float] = [(self.kxx[i]+self.kxx[N-1-i])/2 for i in range(N//2+1)][::-1]
        kxx_symm_err: list[float] = [np.sqrt(self.errkxx[i]**2 + self.errkxx[N-1-i]**2)/2 for i in range(N//2+1)][::-1]

        Sxx_symm: list[float] = [(self.Sxx[i]+self.Sxx[N-1-i])/2 for i in range(N//2+1)][::-1]
        Sxx_symm_err: list[float] = [np.sqrt(self.errSxx[i]**2 + self.errSxx[N-1-i]**2)/2 for i in range(N//2+1)][::-1]

        Sxy_no_symm: list[float] = [ey/dtx/self.LVy*self.LTx * 1e6 for ey,dtx in zip(self.Ey,self.dTx)] # (uV/K)
        Sxy_no_symm_err: list[float] = [abs(sxy) * ((eey/ey)**2 + (edtx/dtx)**2)**0.5 for sxy,ey,eey,dtx,edtx in zip(Sxy_no_symm, self.Ex, self.errEx, self.dTx, self.errdTx)]
        H_: list[float]
        if self.Field[0] < 0:
            H_ = self.Field[N//2:]
            Sxy_symm = [-(Sxy_no_symm[i]-Sxy_no_symm[N-1-i])/2 for i in range(N//2+1)][::-1]

        else:
            H_ = self.Field[:N//2+1][::-1]
            Sxy_symm = [(Sxy_no_symm[i]-Sxy_no_symm[N-1-i])/2 for i in range(N//2+1)][::-1]

        Sxy_symm_err: list[float] = [np.sqrt(Sxy_no_symm_err[i]**2 + Sxy_no_symm_err[N-1-i]**2)/2 for i in range(N//2+1)][::-1]

        return H_, kxx_symm, kxx_symm_err, Sxx_symm, Sxx_symm_err, Sxy_symm, Sxy_symm_err
    
    def symmetrize_dT(self) -> tuple[list[float], list[float], list[float]]:
        N: int = len(self.Field)
        dTx_symm: list[float] = [(self.dTx[i]+self.dTx[N-1-i])/2 for i in range(N//2+1)][::-1]
        dTx_symm_err: list[float] = [np.sqrt(self.errdTx[i]**2 + self.errdTx[N-1-i]**2)/2 for i in range(N//2+1)][::-1]

        H_: list[float]
        if self.Field[0] < 0:
            H_ = self.Field[N//2:]
        else:
            H_ = self.Field[:N//2+1][::-1]
        return H_, dTx_symm, dTx_symm_err


class ReMakeExpFromRaw:
    def __init__(self, filename: str, filename_Seebeck: str, cernox_name: str = "X173409", attr_cor_to_V: list[int] | None = None) -> None:
        self.filename: str = filename
        self.TC_TS: list[list[float]] = []
        with open(file=filename_Seebeck, mode="r") as f:
            for line in f.readlines()[1:]:
                s,t = map(float, line.split())
                self.TC_TS.append([t,s])
        self.TC_TS = sorted(self.TC_TS, key=lambda x:x[0]) # 温度順にソート

        self.RawData: RawDataExpander = RawDataExpander(filename, filename_Seebeck)
        self.RawData.kxxkxy_mode(cernox_name, attr_cor_to_V)
        filename_exp: str = re.sub(r"Raw", r"Exp", self.filename)
        self.ExpData: ExpDataExpander = ExpDataExpander(filename_exp, filename_Seebeck)

        self.StartTime: str = self.RawData.StartTime
        self.LTx: float = self.RawData.LTx
        self.LTy: float = self.RawData.LTy
        self.LVx: float = self.RawData.LVx
        self.LVy: float = self.RawData.LVy
        self.Width: float = self.RawData.Width
        self.Thickness: float = self.RawData.Thickness

        self.Index: list[tuple[int, int, int, int]] = self.ExpData.Index

        self.T_PPMS: list[float] = []
        self.Field: list[float] = []
        self.Current: list[float] = []
        self.T_Cernox: list[float] = []
        self.dVx: list[float] = []
        self.dVy: list[float] = []
        self.dTx: list[float] = []
        self.dTy: list[float] = []
        self.dTx0: list[float] = []
        self.dTy0: list[float] = []
        self.dTx1: list[float] = []
        self.dTy1: list[float] = []
        self.errdTx: list[float] = []
        self.errdTy: list[float] = []
        self.R: float = 1000.0
        self.channel: tuple[int, int] = (1, 2)
        for sidx0, eidx0, sidx1, eidx1 in self.Index:
            tp0, etp0, h0, eh0, cur0, ecur0, tc0, etc0, vx0, evx0, vy0, evy0 = self.ave_std(eidx0)
            tp1, etp1, h1, eh1, cur1, ecur1, tc1, etc1, vx1, evx1, vy1, evy1 = self.ave_std(eidx1)
            self.T_PPMS.append(tp1)
            self.Field.append(h1)
            self.Current.append(cur1)
            self.T_Cernox.append(tc1)
            self.dVx.append(vx1-vx0)
            self.dVy.append(vy1-vy0)
            self.dTx.append((vx1-vx0) / self.Seebeck_at_T(tp1))
            self.dTy.append((vy1-vy0) / self.Seebeck_at_T(tp1))

            self.dTx0.append((vx0) / self.Seebeck_at_T(tp0))
            self.dTy0.append((vy0) / self.Seebeck_at_T(tp0))
            self.dTx1.append((vx1) / self.Seebeck_at_T(tp1))
            self.dTy1.append((vy1) / self.Seebeck_at_T(tp1))

            self.errdTx.append((evx0**2+evx1**2)**0.5 / self.Seebeck_at_T(tp1))
            self.errdTy.append((evy0**2+evy1**2)**0.5 / self.Seebeck_at_T(tp1))
        
        self.kxx: list[float] = [self.R*(i**2)/self.Width/self.Thickness / (dt/self.LTx) for i,dt in zip(self.Current,self.dTx)]
        self.errkxx: list[float] = [k * edtx / dtx for k,dtx,edtx in zip(self.kxx,self.dTx,self.errdTx)]


    def ave_std(self, eidx: int) -> tuple[float, ...]:
        slc: slice = slice(eidx-59,eidx+1)
        channel: tuple[int, int] = self.channel
        aveT_PPMS: float = np.average(self.RawData.PPMSTemp[slc])
        errT_PPMS: float = np.std(self.RawData.PPMSTemp[slc])
        aveH: float = np.average(self.RawData.Field[slc])
        errH: float = np.std(self.RawData.Field[slc])
        aveCurrent: float = np.average(self.RawData.HeaterCurrent[slc])
        errCurrent: float = np.std(self.RawData.HeaterCurrent[slc])
        aveT_Cernox: float = np.average(self.RawData.CernoxTemp[slc])
        errT_Cernox: float = np.std(self.RawData.CernoxTemp[slc])
        Vs = [self.RawData.V1, self.RawData.V2, self.RawData.V3, self.RawData.V4, self.RawData.V5, self.RawData.V6]
        aveVx: float = np.average(Vs[channel[0]][slc])
        errVx: float = np.std(Vs[channel[0]][slc])
        aveVy: float = np.average(Vs[channel[1]][slc])
        errVy: float = np.std(Vs[channel[1]][slc])
        # print("######")
        # print(aveT_PPMS, errT_PPMS, aveH, errH, aveCurrent, errCurrent, aveT_Cernox, errT_Cernox, aveVx, errVx, aveVy, errVy)
        # print("+++++++")
        return aveT_PPMS, errT_PPMS, aveH, errH, aveCurrent, errCurrent, aveT_Cernox, errT_Cernox, aveVx, errVx, aveVy, errVy

    def Seebeck_at_T(self, T: float) -> float:
        if not (self.TC_TS[0][0] <= T <= self.TC_TS[-1][0]):
            return 0
        S_at_T: float = -1
        for i in range(len(self.TC_TS)-1):
            if self.TC_TS[i][0] <= T <= self.TC_TS[i+1][0]:
                S_at_T = self.TC_TS[i][1] + (self.TC_TS[i+1][1]-self.TC_TS[i][1]) / (self.TC_TS[i+1][0]-self.TC_TS[i][0]) * (T-self.TC_TS[i][0])
                break
        if S_at_T == -1:
            raise RuntimeError
        return S_at_T

    def symmetrize(self) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        N: int = len(self.Field)
        lamxx: list[float] = [self.Width*self.Thickness/(self.R*(i**2)) * (dt/self.LTx) for i,dt in zip(self.Current,self.dTx)]
        lamyx: list[float] = [self.Width*self.Thickness/(self.R*(i**2)) * (dt/self.LTy) for i,dt in zip(self.Current,self.dTy)]
        lamxx_err: list[float] = [self.Width*self.Thickness/(self.R*(i**2)) * (edt/self.LTx) for i,edt in zip(self.Current,self.errdTx)]
        lamyx_err: list[float] = [self.Width*self.Thickness/(self.R*(i**2)) * (edt/self.LTy) for i,edt in zip(self.Current,self.errdTy)]

        # symmetrize
        lamxx_symm: list[float] = [(lamxx[i]+lamxx[N-1-i])/2 for i in range(N//2+1)][::-1]
        lamxx_symm_err: list[float] = [np.sqrt(lamxx_err[i]**2 + lamxx_err[N-1-i]**2)/2 for i in range(N//2+1)][::-1]

        lamyx_symm: list[float]
        H_: list[float]
        if self.Field[0] < 0:
            H_ = self.Field[N//2:]
            lamyx_symm = [-(lamyx[i]-lamyx[N-1-i])/2 for i in range(N//2+1)][::-1]

        else:
            H_ = self.Field[:N//2+1][::-1]
            lamyx_symm = [(lamyx[i]-lamyx[N-1-i])/2 for i in range(N//2+1)][::-1]

        lamyx_symm_err: list[float] = [np.sqrt(lamyx_err[i]**2 + lamyx_err[N-1-i]**2)/2 for i in range(N//2+1)][::-1]
        kxx: list[float] = [lx / (lx**2 + ly**2) for lx,ly in zip(lamxx_symm,lamyx_symm)]
        kxy: list[float] = [ly / (lx**2 + ly**2) for lx,ly in zip(lamxx_symm,lamyx_symm)]
        kxx_err: list[float] = [np.sqrt(((x**2-y**2)*x_err)**2 + (2*x*y*y_err)**2) / ((x**2 + y**2)**2) for x,y,x_err,y_err in zip(lamxx_symm,lamyx_symm,lamxx_symm_err,lamyx_symm_err)]
        kxy_err: list[float] = [np.sqrt((2*x*y*x_err)**2 + ((x**2-y**2)*y_err)**2) / ((x**2 + y**2)**2) for x,y,x_err,y_err in zip(lamxx_symm,lamyx_symm,lamxx_symm_err,lamyx_symm_err)]
        return H_, kxx, kxy, kxx_err, kxy_err
    
    def symmetrize_dT(self) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        N: int = len(self.Field)
        dTx_symm: list[float] = [(self.dTx[i]+self.dTx[N-1-i])/2 for i in range(N//2+1)][::-1]
        dTx_symm_err: list[float] = [np.sqrt(self.errdTx[i]**2 + self.errdTx[N-1-i]**2)/2 for i in range(N//2+1)][::-1]
        
        dTy_symm: list[float]
        H_: list[float]
        if self.Field[0] < 0:
            H_ = self.Field[N//2:]
            dTy_symm = [-(self.dTy[i]-self.dTy[N-1-i])/2 for i in range(N//2+1)][::-1]

        else:
            H_ = self.Field[:N//2+1][::-1]
            dTy_symm = [(self.dTy[i]-self.dTy[N-1-i])/2 for i in range(N//2+1)][::-1]
        dTy_symm_err: list[float] = [np.sqrt(self.errdTy[i]**2 + self.errdTy[N-1-i]**2)/2 for i in range(N//2+1)][::-1]
        return H_, dTx_symm, dTy_symm, dTx_symm_err, dTy_symm_err

    def make_new_exp_data_file(self) -> None:
        filename_new: str = re.sub(r"Raw", r"NewExp", self.filename)
        with open(filename_new, mode="w") as f:
            f.write("".join(self.ExpData.full_contents[0:14]))
            for i, (sidx0, eidx0, sidx1, eidx1) in enumerate(self.Index):
                line = (sidx0, eidx0) + self.ave_std(eidx0) + (sidx1, eidx1) + self.ave_std(eidx1) + (self.dTx[i], self.errdTx[i], self.kxx[i], self.errkxx[i])
                f.write("\t".join([f"{v:.9e}" for v in line]) + "\n")


class AATTPMD(RawDataExpander, tk.Frame):
    """App. for Analysis of Thermal Transport Property Measurement Data
    """
    def __init__(self, filename: str, filename_Seebeck: str, cernox_name: str, attr_cor_to_V: list[int] | None = None) -> None:
        RawDataExpander.__init__(self, filename, filename_Seebeck)
        self.kxxkxy_mode(cernox_name, attr_cor_to_V)
        filename_exp: str = re.sub(r"Raw", r"Exp", self.filename)
        self.ExpData: ExpDataExpander = ExpDataExpander(filename_exp, filename_Seebeck)

        root: tk.Tk = tk.Tk()
        tk.Frame.__init__(self, root)

        self.master: tk.Tk = root
        self.master.title(f"{self.filename}")
        self.master.geometry('1500x900')

        #-----------------------------------------------

        # matplotlib配置用フレーム
        mtpltlb_frame: tk.Frame = tk.Frame(self.master)

        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams["legend.framealpha"] = 0
        plt.rcParams['legend.fontsize'] = 8


        figsize: tuple[int, int] = (11,8)
        fig: plt.Figure = plt.figure(figsize=figsize)
        plt.subplots_adjust(wspace=0.2, hspace=0, bottom=0.20)
        
        self.ax1: plt.Subplot = fig.add_subplot(221)
        self.ax1.xaxis.set_ticks_position('both')
        self.ax1.yaxis.set_ticks_position('both')
        self.ax1.plot(self.Time, self.Field, marker='o', color="blue", markersize=2)
        self.ax1.set_xlabel(xlabel=r"Time (sec)")
        self.ax1.set_ylabel(ylabel=r"Magnetic Field (Oe)")
        self.ax1.xaxis.set_ticklabels([]) # 目盛を削除
        self.ax1.set_title(self.filename)

        self.ax2: plt.Subplot = fig.add_subplot(223)
        self.ax2.xaxis.set_ticks_position('both')
        self.ax2.yaxis.set_ticks_position('both')
        self.ax2.plot(self.Time, self.CernoxTemp, marker='o', color="blue", markersize=2)
        self.ax2.plot(self.Time, self.PPMSTemp, marker='o', color="red", markersize=2)
        self.ax2.set_xlabel(xlabel=r"Time (sec)")
        self.ax2.set_ylabel(ylabel="Temperature (K)\n(Cernox:blue, PPMS:red)")

        self.ax3: plt.Subplot = fig.add_subplot(222)
        self.ax3.xaxis.set_ticks_position('both')
        self.ax3.yaxis.set_ticks_position('both')
        self.ax3.plot(self.Time, self.dTx, marker='o', color="blue", markersize=2)
        self.ax3.set_xlabel(xlabel=r"Time (sec)")
        self.ax3.set_ylabel(ylabel=r"$\Delta T_{x}$ (K)")
        self.ax3.xaxis.set_ticklabels([]) # 目盛を削除

        self.ax4: plt.Subplot = fig.add_subplot(224)
        self.ax4.xaxis.set_ticks_position('both')
        self.ax4.yaxis.set_ticks_position('both')
        self.ax4.plot(self.Time, self.dTy, marker='o', color="blue", markersize=2)
        self.ax4.set_xlabel(xlabel=r"Time (sec)")
        self.ax4.set_ylabel(ylabel=r"$\Delta T_{y}$ (K)")

        # マウスのhoverで描画する縦線
        self.ln1_hover, = self.ax1.plot([],[], color="black", linewidth=1)
        self.ln2_hover, = self.ax2.plot([],[], color="black", linewidth=1)
        self.ln3_hover, = self.ax3.plot([],[], color="black", linewidth=1)
        self.ln4_hover, = self.ax4.plot([],[], color="black", linewidth=1)
        
        # マウスのclickで描画する縦線
        self.ln1_start, = self.ax1.plot([],[], color="green", linewidth=1)
        self.ln2_start, = self.ax2.plot([],[], color="green", linewidth=1)
        self.ln3_start, = self.ax3.plot([],[], color="green", linewidth=1)
        self.ln4_start, = self.ax4.plot([],[], color="green", linewidth=1)

        # マウスのclickで描画する縦線
        self.ln1_end, = self.ax1.plot([],[], color="red", linewidth=1)
        self.ln2_end, = self.ax2.plot([],[], color="red", linewidth=1)
        self.ln3_end, = self.ax3.plot([],[], color="red", linewidth=1)
        self.ln4_end, = self.ax4.plot([],[], color="red", linewidth=1)

        # 指定されたsidx0, eidx0, sidx1, eidx1に対応する縦線
        self.ln1_s0, = self.ax1.plot([],[], color="blue", linewidth=1)
        self.ln2_s0, = self.ax2.plot([],[], color="blue", linewidth=1)
        self.ln3_s0, = self.ax3.plot([],[], color="blue", linewidth=1)
        self.ln4_s0, = self.ax4.plot([],[], color="blue", linewidth=1)

        self.ln1_e0, = self.ax1.plot([],[], color="orange", linewidth=1)
        self.ln2_e0, = self.ax2.plot([],[], color="orange", linewidth=1)
        self.ln3_e0, = self.ax3.plot([],[], color="orange", linewidth=1)
        self.ln4_e0, = self.ax4.plot([],[], color="orange", linewidth=1)

        self.ln1_s1, = self.ax1.plot([],[], color="blue", linewidth=1)
        self.ln2_s1, = self.ax2.plot([],[], color="blue", linewidth=1)
        self.ln3_s1, = self.ax3.plot([],[], color="blue", linewidth=1)
        self.ln4_s1, = self.ax4.plot([],[], color="blue", linewidth=1)

        self.ln1_e1, = self.ax1.plot([],[], color="orange", linewidth=1)
        self.ln2_e1, = self.ax2.plot([],[], color="orange", linewidth=1)
        self.ln3_e1, = self.ax3.plot([],[], color="orange", linewidth=1)
        self.ln4_e1, = self.ax4.plot([],[], color="orange", linewidth=1)

        # figとFrameの対応付け
        self.fig_canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(fig, mtpltlb_frame)
        self.toolbar: NavigationToolbar2Tk = NavigationToolbar2Tk(self.fig_canvas, mtpltlb_frame)
        self.cid1: Any = fig.canvas.mpl_connect('button_press_event', self._fig_click)
        self.cid2: Any = fig.canvas.mpl_connect('motion_notify_event', self._fig_hover)
        self.fig_canvas.get_tk_widget().pack(expand=False, side=tk.LEFT)

        mtpltlb_frame.pack(side=tk.LEFT)

        #-----------------------------------------------
        ### sliderを生成
        self.sliders: list[tk.DoubleVar] = [tk.DoubleVar(), tk.DoubleVar()]
        slider0: tk.Scale = tk.Scale(self.master,
                    variable = self.sliders[0],
                    command = self._slider_scroll,
                    orient = tk.HORIZONTAL,
                    length = 300,
                    width = 20,
                    sliderlength = 10,
                    from_ = 0,
                    to = max(self.Time)+1,
                    resolution = 1,
                    tickinterval = 0
                    )
        slider0.pack()

        slider1: tk.Scale = tk.Scale(self.master,
                    variable = self.sliders[1],
                    command = self._slider_scroll,
                    orient = tk.HORIZONTAL,
                    length = 300,
                    width = 20,
                    sliderlength = 10,
                    from_ = 0,
                    to = max(self.Time)+1,
                    resolution = 1,
                    tickinterval = 0
                    )
        slider1.pack()
        self.sliders[0].set(0)
        self.sliders[1].set(max(self.Time)+1)

        #-----------------------------------------------
        ### slider制御のFrame
        slider_control_frame: tk.Frame = tk.Frame(self.master, borderwidth=3, relief="ridge")
        # resetボタン
        reset_button: tk.Button = tk.Button(slider_control_frame, text="Reset", command=self._reset_click)
        reset_button.grid(row=0, column=0)

        # 相対位置固定チェックボタン
        self.t_lock: float = 0.0
        self.is_locked: tk.BooleanVar = tk.BooleanVar()
        self.is_locked.set(False)
        lock_cbutton: tk.Checkbutton = tk.Checkbutton(slider_control_frame, variable=self.is_locked, text="Lock", command=self._lock_click)
        lock_cbutton.grid(row=1, column=0)

        # increment関係のFrame
        increment_frame: tk.Frame = tk.Frame(slider_control_frame, borderwidth=3, relief="ridge")
        increment_button: tk.Button = tk.Button(increment_frame, text="Increment", command=self._increment_click)
        increment_button.grid(row=0, column=0)
        self.increment: tk.Entry = tk.Entry(increment_frame, width=8)
        self.increment.grid(row=1, column=0)
        increment_frame.grid(rowspan=2, column=1,row=0,sticky=tk.N+tk.S)

        slider_control_frame.pack()

        #-----------------------------------------------
        ### データ解析のFrame
        analysis_frame: tk.Frame = tk.Frame(self.master, borderwidth=3, relief="ridge")

        # データsave先のfilename
        lbl_filename: tk.Label = tk.Label(analysis_frame, text="save filename")
        lbl_filename.pack()
        self.filename_to_save: tk.Entry = tk.Entry(analysis_frame, width=20)
        self.filename_to_save.pack()

        # 測定番号を元に時間範囲を指定
        exp_idx_frame: tk.Frame = tk.Frame(analysis_frame, borderwidth=3, relief="ridge")
        lbl_exp_idx: tk.Label = tk.Label(exp_idx_frame, text="Exp index", foreground="black")
        lbl_exp_idx.grid(row=0, column=0)
        self.exp_idx: tk.Entry = tk.Entry(exp_idx_frame, width=8)
        self.exp_idx.grid(row=1, column=0)
        lbl_now_idx: tk.Label = tk.Label(exp_idx_frame, text="now index", foreground="black")
        lbl_now_idx.grid(row=0, column=1)
        self.now_idx_value: tk.StringVar = tk.StringVar()
        self.now_idx_value.set(f"0")
        self.lbl_now_idx_value = tk.Label(exp_idx_frame, textvariable=self.now_idx_value, relief="sunken", width=15)
        self.lbl_now_idx_value.grid(row=1, column=1)
        update_button: tk.Button = tk.Button(exp_idx_frame, text="Update", command=self._update_click)
        update_button.grid(row=0, column=0)
        prev_button: tk.Button = tk.Button(exp_idx_frame, text="Prev", command=self._prev_click)
        prev_button.grid(row=2, column=0)
        next_button: tk.Button = tk.Button(exp_idx_frame, text="Next", command=self._next_click)
        next_button.grid(row=2, column=1)
        lbl_kxx: tk.Label = tk.Label(exp_idx_frame, text="kxx (W/Km)")
        lbl_kxx.grid(row=3, column=0)
        self.kxx_value: tk.StringVar = tk.StringVar()
        self.lbl_kxx_value: tk.Label = tk.Label(exp_idx_frame, textvariable=self.kxx_value, relief="sunken", width=15)
        self.lbl_kxx_value.grid(row=4, column=0)
        exp_idx_frame.pack(pady=10)

        # データとして使う時間範囲の設定をするためのFrame
        range_frame = tk.Frame(analysis_frame, borderwidth=3, relief="ridge")
        lbl_start = tk.Label(range_frame, text="start (s)", foreground="green")
        lbl_start.grid(row=0, column=0)
        self.time_start: tk.Entry = tk.Entry(range_frame, width=8)
        self.time_start.grid(row=1, column=0)
        lbl_end = tk.Label(range_frame, text="end (s)", foreground="red")
        lbl_end.grid(row=0, column=1)
        self.time_end: tk.Entry = tk.Entry(range_frame, width=8)
        self.time_end.grid(row=1, column=1)
        self.start_or_end: int = 0
        self.is_select_range_by_click: tk.BooleanVar = tk.BooleanVar()
        self.is_select_range_by_click.set(True)
        select_range_by_click_cbutton = tk.Checkbutton(range_frame, variable=self.is_select_range_by_click, text="select range by click", command=self._select_range_by_click_click)
        select_range_by_click_cbutton.grid(row=2, columnspan=2)
        range_frame.pack(pady=10)

        # データを表示するFrame
        value_frame: tk.Frame = tk.Frame(analysis_frame, borderwidth=3, relief="ridge")
        lbl_field: tk.Label = tk.Label(value_frame, text="H (Oe)")
        lbl_field.grid(row=0, column=0)
        self.field_value: tk.StringVar = tk.StringVar()
        self.lbl_field_value: tk.Label = tk.Label(value_frame, textvariable=self.field_value, relief="sunken", width=15)
        self.lbl_field_value.grid(row=1, column=0)
        lbl_Cernox_temp = tk.Label(value_frame, text="T_Cernox (K)")
        lbl_Cernox_temp.grid(row=2, column=0)
        self.Cernox_temp_value: tk.StringVar = tk.StringVar()
        self.lbl_Cernox_temp_value: tk.Label = tk.Label(value_frame, textvariable=self.Cernox_temp_value, relief="sunken", width=15)
        self.lbl_Cernox_temp_value.grid(row=3, column=0)
        lbl_dTx: tk.Label = tk.Label(value_frame, text="ΔTx (K)")
        lbl_dTx.grid(row=0, column=1)
        self.dTx_value: tk.StringVar = tk.StringVar()
        self.lbl_dTx_value: tk.Label = tk.Label(value_frame, textvariable=self.dTx_value, relief="sunken", width=15)
        self.lbl_dTx_value.grid(row=1, column=1)
        lbl_dTy: tk.Label = tk.Label(value_frame, text="ΔTy (K)")
        lbl_dTy.grid(row=2, column=1)
        self.dTy_value: tk.StringVar = tk.StringVar()
        self.lbl_dTy_value: tk.Label = tk.Label(value_frame, textvariable=self.dTy_value, relief="sunken", width=15)
        self.lbl_dTy_value.grid(row=3, column=1)
        value_frame.pack(pady=10)

        # 指定した時間範囲のデータを計算させるボタン
        calc_button = tk.Button(analysis_frame, text="Calc", command=self._calc_click)
        calc_button.pack()

        # 計算したデータをsaveさせるボタン
        save_button = tk.Button(analysis_frame, text="Save", command=self._save_click)
        save_button.pack()

        analysis_frame.pack(pady=20)

        #-----------------------------------------------
    
    def _update_xlim(self, t1: float, t2: float) -> None:
        self.ax1.set_xlim(t1,t2)
        self.ax2.set_xlim(t1,t2)
        self.ax3.set_xlim(t1,t2)
        self.ax4.set_xlim(t1,t2)

    def _update_ylim(self, t1: float, t2: float) -> None:
        lineax1 = self.ax1.lines[0]
        yax1 = lineax1._yorig[bisect_left(lineax1._xorig,t1):bisect_left(lineax1._xorig,t2)]
        yax1m,yax1M = min(yax1), max(yax1)
        self.ax1.set_ylim(yax1m-(yax1M-yax1m)*0.05,yax1M+(yax1M-yax1m)*0.05)
        lineax2 = self.ax2.lines[0]
        yax2 = lineax2._yorig[bisect_left(lineax2._xorig,t1):bisect_left(lineax2._xorig,t2)]
        lineax2_2 = self.ax2.lines[1]
        yax2_2 = lineax2_2._yorig[bisect_left(lineax2_2._xorig,t1):bisect_left(lineax2_2._xorig,t2)]
        yax2m,yax2M = min(min(yax2),min(yax2_2)), max(max(yax2),max(yax2_2))
        self.ax2.set_ylim(yax2m-(yax2M-yax2m)*0.05,yax2M+(yax2M-yax2m)*0.05)
        lineax3 = self.ax3.lines[0]
        yax3 = lineax3._yorig[bisect_left(lineax3._xorig,t1):bisect_left(lineax3._xorig,t2)]
        yax3m,yax3M = min(yax3), max(yax3)
        self.ax3.set_ylim(yax3m-(yax3M-yax3m)*0.05,yax3M+(yax3M-yax3m)*0.05)
        lineax4 = self.ax4.lines[0]
        yax4 = lineax4._yorig[bisect_left(lineax4._xorig,t1):bisect_left(lineax4._xorig,t2)]
        yax4m,yax4M = min(yax4), max(yax4)
        self.ax4.set_ylim(yax4m-(yax4M-yax4m)*0.05,yax4M+(yax4M-yax4m)*0.05)

    def _update_start_time(self, x_start: float) -> None:
        self.time_start.delete(0, tk.END)
        self.time_start.insert(tk.END, f"{x_start:.1f}")
        self.start_or_end = 1
        self.ln1_start.set_data([x_start,x_start], self.ax1.get_ylim())
        self.ln2_start.set_data([x_start,x_start], self.ax2.get_ylim())
        self.ln3_start.set_data([x_start,x_start], self.ax3.get_ylim())
        self.ln4_start.set_data([x_start,x_start], self.ax4.get_ylim())

    def _update_end_time(self, x_end: float) -> None:
        self.time_end.delete(0, tk.END)
        self.time_end.insert(tk.END, f"{x_end:.1f}")
        self.start_or_end = 0
        self.ln1_end.set_data([x_end,x_end], self.ax1.get_ylim())
        self.ln2_end.set_data([x_end,x_end], self.ax2.get_ylim())
        self.ln3_end.set_data([x_end,x_end], self.ax3.get_ylim())
        self.ln4_end.set_data([x_end,x_end], self.ax4.get_ylim())

    def _update_exp_line(self, t1: float, t2: float, t3: float, t4: float) -> None:
        """バックグラウンド測定開始・終了時間とQ>0での測定開始・終了時間の描画
        """
        self.ln1_s0.set_data([t1,t1], self.ax1.get_ylim())
        self.ln2_s0.set_data([t1,t1], self.ax2.get_ylim())
        self.ln3_s0.set_data([t1,t1], self.ax3.get_ylim())
        self.ln4_s0.set_data([t1,t1], self.ax4.get_ylim())

        self.ln1_e0.set_data([t2,t2], self.ax1.get_ylim())
        self.ln2_e0.set_data([t2,t2], self.ax2.get_ylim())
        self.ln3_e0.set_data([t2,t2], self.ax3.get_ylim())
        self.ln4_e0.set_data([t2,t2], self.ax4.get_ylim())

        self.ln1_s1.set_data([t3,t3], self.ax1.get_ylim())
        self.ln2_s1.set_data([t3,t3], self.ax2.get_ylim())
        self.ln3_s1.set_data([t3,t3], self.ax3.get_ylim())
        self.ln4_s1.set_data([t3,t3], self.ax4.get_ylim())

        self.ln1_e1.set_data([t4,t4], self.ax1.get_ylim())
        self.ln2_e1.set_data([t4,t4], self.ax2.get_ylim())
        self.ln3_e1.set_data([t4,t4], self.ax3.get_ylim())
        self.ln4_e1.set_data([t4,t4], self.ax4.get_ylim())

    def _reset_click(self, event: Any | None = None) -> None:
        """'Reset'を押したときにsliderの値をを初期値にリセット
        """
        t1: float = 0
        t2: float = max(self.Time)+1
        self.sliders[0].set(t1)
        self.sliders[1].set(t2)
        self._update_xlim(t1,t2)
        self._update_ylim(t1,t2)
        self.fig_canvas.draw()

    def _lock_click(self) -> None:
        """'Lock'を押したときにsliderの相対位置を固定
        """
        t1: float = self.sliders[0].get()
        t2: float = self.sliders[1].get()
        self.t_lock = t2-t1

    def _increment_click(self) -> None:
        """'Increment'を押したときに描画範囲や'start time'や'end time'を自動的に変更
        """
        increment: float = float(self.increment.get())
        t1: float = self.sliders[0].get() + increment
        t2: float = self.sliders[1].get() + increment
        self.sliders[0].set(t1)
        self.sliders[1].set(t2)
        self._update_xlim(t1,t2)
        self._update_ylim(t1,t2)
        if self.time_start.get():
            x_start: float = float(self.time_start.get()) + increment
            self._update_start_time(x_start)
        if self.time_end.get():
            x_end: float = float(self.time_end.get()) + increment
            self._update_end_time(x_end)
        self.fig_canvas.draw()

    def _reflection_exp_idx(self, idx: int) -> None:
        """描画にexp_idxを反映させる
        """
        sidx0, eidx0, sidx1, eidx1 = self.ExpData.Index[idx]
        t1: float = self.Time[sidx0]
        t2: float = self.Time[eidx0]
        t3: float = self.Time[sidx1]
        t4: float = self.Time[eidx1]
        self.sliders[0].set(t1-30)
        self.sliders[1].set(t4+30)
        self._update_xlim(t1-30,t4+30)
        self._update_ylim(t1-30,t4+30)
        self._update_exp_line(t1,t2,t3,t4)
        self.now_idx_value.set(f"{idx}")
        kxx: float = self.ExpData.kxx[idx]
        self.kxx_value.set(f"{kxx:.4f}")

    def _update_click(self) -> None:
        """'Update'を押したときに描画範囲を自動的に変更
        """
        now: int
        if self.exp_idx.get() == '' or self.exp_idx.get() is None:
            now = 0
        else:
            now = int(self.exp_idx.get())
        self._reflection_exp_idx(now)
        self.fig_canvas.draw()

    def _prev_click(self) -> None:
        """'Prev'を押したときに描画範囲を自動的に変更
        """
        value: int = max(0, int(self.now_idx_value.get())-1)
        self._reflection_exp_idx(value)
        self.fig_canvas.draw()

    def _next_click(self) -> None:
        """'Next'を押したときに描画範囲を自動的に変更
        """
        value: int = min(len(self.ExpData.Index)-1, int(self.now_idx_value.get())+1)
        self._reflection_exp_idx(value)
        self.fig_canvas.draw()

    def _select_range_by_click_click(self) -> None:
        """'select range by click'をclickしたときに'is_select_range_by_click'を変更
        """
        self.is_select_range_by_click.set(self.is_select_range_by_click.get()^False)

    def _calc_click(self, event: Any | None = None) -> None:
        """'Calc'ボタンをクリックしたときに'start time'から'end time'の各物理量の平均値と標準偏差を計算
        """
        t_start: float = float(self.time_start.get())
        t_end: float = float(self.time_end.get())
        idx: list[int] = [i for i,t in enumerate(self.Time) if t_start <= t <= t_end]
        now_H: list[float] = [self.Field[i] for i in idx]
        ave_H: float = np.average(now_H)
        std_H: float = np.std(now_H)

        now_Cernox_temp: list[float] = [self.CernoxTemp[i] for i in idx]
        ave_Cernox_temp: float = np.average(now_Cernox_temp)
        std_Cernox_temp: float = np.std(now_Cernox_temp)
        
        now_dTx: list[float] = [self.dTx[i] for i in idx]
        ave_dTx: float = np.average(now_dTx)
        std_dTx: float = np.std(now_dTx)

        now_dTy: list[float] = [self.dTy[i] for i in idx]
        ave_dTy: float = np.average(now_dTy)
        std_dTy: float = np.std(now_dTy)
        self.field_value.set(f"{ave_H:.1f} ± {std_H:.2g}")
        self.Cernox_temp_value.set(f"{ave_Cernox_temp:.4f} ± {std_Cernox_temp:.2g}")
        self.dTx_value.set(f"{ave_dTx:.4f} ± {std_dTx:.2g}")
        self.dTy_value.set(f"{ave_dTy:.4f} ± {std_dTy:.2g}")
        
    def _save_click(self) -> None:
        """'Save'ボタンをクリックしたときに'start time'から'end time'の各物理量の平均値と
            標準偏差を指定したファイルに保存
        """
        filename = self.filename_to_save.get()
        try:
            t_start: float = float(self.time_start.get())
            t_end: float = float(self.time_end.get())
            idx: list[int] = [i for i,t in enumerate(self.Time) if t_start <= t <= t_end]

            now_H: list[float] = [self.Field[i] for i in idx]
            ave_H: float = np.average(now_H)
            std_H: float = np.std(now_H)

            now_Cernox_temp: list[float] = [self.CernoxTemp[i] for i in idx]
            ave_Cernox_temp: float = np.average(now_Cernox_temp)
            std_Cernox_temp: float = np.std(now_Cernox_temp)
            
            now_dTx: list[float] = [self.dTx[i] for i in idx]
            ave_dTx: float = np.average(now_dTx)
            std_dTx: float = np.std(now_dTx)

            now_dTy: list[float] = [self.dTy[i] for i in idx]
            ave_dTy: float = np.average(now_dTy)
            std_dTy: float = np.std(now_dTy)

            res: str = ", ".join(map(str, [t_start, t_end, 
                                ave_H, std_H, 
                                ave_Cernox_temp, std_Cernox_temp,
                                ave_dTx, std_dTx,
                                ave_dTy, std_dTy]))
            with open(file=filename, mode="w") as f:
                f.write(res)
        except:
            print("failed: save data")

    def _slider_scroll(self, event: Any | None = None) -> None:
        """sliderを変化させたときにx座標の描画範囲を変更
        """
        t1: float = self.sliders[0].get()
        t2: float = self.sliders[1].get()
        if self.is_locked.get():
            if t2 != t1+self.t_lock:
                t2 = t1+self.t_lock
                self.sliders[1].set(t2)
        # 描画範囲を更新
        self._update_xlim(t1,t2)
        self._update_ylim(t1,t2)
        self.fig_canvas.draw()

    def _fig_hover(self, event: Any) -> None:
        """figure領域をマウスオーバーしたときにそのx座標に対応する縦線を描画
        """
        x: float = event.xdata
        self.ln1_hover.set_data([x,x], self.ax1.get_ylim())
        self.ln2_hover.set_data([x,x], self.ax2.get_ylim())
        self.ln3_hover.set_data([x,x], self.ax3.get_ylim())
        self.ln4_hover.set_data([x,x], self.ax4.get_ylim())
        self.fig_canvas.draw()

    def _fig_click(self, event: Any) -> None:
        """figure領域をclickしたときにそのx座標に対応する縦線を描画
        """
        x: float = event.xdata
        if self.is_select_range_by_click.get():
            if self.start_or_end == 0:
                self._update_start_time(x)
                self.fig_canvas.draw()
            else:
                self._update_end_time(x)
                self.fig_canvas.draw()
        
    def excute(self, save_filename: str | None = None, Tx_gain: float = 1) -> None:
        """アプリを実行
        """
        if save_filename is not None:
            self.filename_to_save.insert(tk.END, save_filename)

        self.mainloop()
    
    def delete(self) -> None:
        self.master.destroy()


def main() -> None:
    pass
    return

if __name__ == "__main__":
    main()

