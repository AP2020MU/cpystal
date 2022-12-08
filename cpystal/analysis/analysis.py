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
import datetime
import glob
import os
import re
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # type: ignore
import matplotlib.animation
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import numpy.typing as npt
import pymatgen # type: ignore
from pymatgen.io.cif import CifParser # type: ignore
import pymatgen.analysis.diffraction.xrd # type: ignore
from scipy.optimize import curve_fit # type: ignore
from scipy.stats import norm # type: ignore
from scipy import integrate, optimize # type: ignore
import tkinter as tk
import tqdm


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


def cal_dTx(
        Q: float,
        lx: float,
        w: float,
        t: float,
        kxx: float,
    ) -> float:
    """Calculating temperature difference due to heat current.

    Note:
        dTx = (Q/wt) * (lx/kxx)

    Args:
        Q (float): Heater power (mW).
        lx (float): Length between thermocouples (um).
        w (float): Width of the sample (um).
        t (float): Thickness of the sample (um).
        kxx (float): Thermal conductivity (W/Km).

    Returns:
        (float): dTx (K).
    """
    return Q / (w*t) * lx / kxx * 1000.0


def cal_kxx(
        Q: float,
        lx: float,
        w: float,
        t: float,
        dTx: float,
    ) -> float:
    """Calculating temperature difference due to heat current.

    Note:
        dTx = (Q/wt) * (lx/kxx)

    Args:
        Q (float): Heater power (mW).
        lx (float): Length between thermocouples (um).
        w (float): Width of the sample (um).
        t (float): Thickness of the sample (um).
        dTx (float): dTx (K).

    Returns:
        (float): Thermal conductivity (W/Km).
    """
    return Q / (w*t) * lx / dTx * 1000.0


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
        self._operated_filename: str = filename
        self.filename_Seebeck: str = filename_Seebeck
        with open(file=filename, mode="r") as f:
            self.full_contents: list[str] = f.readlines()

        self.TC_TS: list[list[float]] = []
        with open(file=filename_Seebeck, mode="r") as f:
            for line in f.readlines()[1:]:
                s,t = map(float, line.split())
                self.TC_TS.append([t,s])
        self.TC_TS = sorted(self.TC_TS, key=lambda x:x[0]) # 温度順にソート
        
        self.header_length: int = 14

        self.StartTime: datetime.datetime = datetime.datetime.strptime(re.sub(r"[^\d]+?:", r"", self.full_contents[0]).strip(), '%Y/%m/%d %H:%M:%S')
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
        self.Q: list[list[float]] = []
        self.R: float = 1000.

        idx: int = self.header_length
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
            self.Q.append(self.R*cur**2/1000.)
            self.V1.append([v1re,v1im])
            self.V2.append([v2re,v2im])
            self.V3.append([v3re,v3im])
            self.V4.append([v4re,v4im])
            self.V5.append([v5re,v5im])
            self.V6.append([v6re,v6im])        

        self.S_TC: list[float] = []
        for i,t in enumerate(self.PPMSTemp):
            if self.Seebeck_at_T(t) == 0:
                self.S_TC.append(self.Seebeck_at_T((self.PPMSTemp[i-1]+self.PPMSTemp[i+1])/2))
            else:
                self.S_TC.append(self.Seebeck_at_T(t))
    
    def check_same_condition(self, other: RawDataExpander) -> bool:
        return self.LTx == other.LTx and self.LTy == other.LTy and self.LVx == other.LVx and \
            self.LVy == other.LVy and self.Width == other.Width and self.Thickness == other.Thickness and \
                self.cernox_name == other.cernox_name and self.attr_cor_to_V == other.attr_cor_to_V
    
    def __add__(self, other: RawDataExpander) -> RawDataExpander:
        new: RawDataExpander = self.__class__(self.filename, self.filename_Seebeck)
        new.kxxkxy_mode(self.cernox_name, self.attr_cor_to_V)
        if self.check_same_condition(other):
            if self.StartTime < other.StartTime:
                new._operated_filename = self.filename + other.filename
                new.Time = self.Time + [t + (other.StartTime-self.StartTime).total_seconds() for t in other.Time]
                new.PPMSTemp = self.PPMSTemp + other.PPMSTemp
                new.Field = self.Field + other.Field
                new.Angle = self.Angle + other.Angle
                new.HeaterCurrent = self.HeaterCurrent + other.HeaterCurrent
                new.CernoxTemp = self.CernoxTemp + other.CernoxTemp
                new.V1 = self.V1 + other.V1
                new.V2 = self.V2 + other.V2
                new.V3 = self.V3 + other.V3
                new.V4 = self.V4 + other.V4
                new.V5 = self.V5 + other.V5
                new.V6 = self.V6 + other.V6
                new.Q = self.Q + other.Q
                new.S_TC = self.S_TC + other.S_TC
                new.dTx = self.dTx + other.dTx
                new.dTy = self.dTy + other.dTy
            else:
                new._operated_filename = other.filename + self.filename
                new.Time = other.Time + [t + (self.StartTime-other.StartTime).total_seconds() for t in self.Time]
                new.PPMSTemp = other.PPMSTemp + self.PPMSTemp
                new.Field = other.Field + self.Field
                new.Angle = other.Angle + self.Angle
                new.HeaterCurrent = other.HeaterCurrent + self.HeaterCurrent
                new.CernoxTemp = other.CernoxTemp + self.CernoxTemp
                new.V1 = other.V1 + self.V1
                new.V2 = other.V2 + self.V2
                new.V3 = other.V3 + self.V3
                new.V4 = other.V4 + self.V4
                new.V5 = other.V5 + self.V5
                new.V6 = other.V6 + self.V6
                new.Q = other.Q + self.Q
                new.S_TC = other.S_TC + self.S_TC
                new.dTx = other.dTx + self.dTx
                new.dTy = other.dTy + self.dTy
        else:
            print(self.LTx, other.LTx, self.LTy, other.LTy, self.LVx, other.LVx, \
            self.LVy, other.LVy, self.Width, other.Width, self.Thickness, other.Thickness, \
                self.cernox_name, other.cernox_name, self.attr_cor_to_V, other.attr_cor_to_V)
            raise ValueError()
        return new
    
    def _set_CernoxTemp(self, cernox_name: str, TR: list[float]) -> None:
        X173409_logRlogT_table: list[tuple[float, float]] = [(3.29314 - 1.42456*log10t + 0.867728*log10t**2 - 0.324371*log10t**3 + 0.0380185*log10t**4, log10t) for log10t in np.linspace(np.log10(1.5),np.log10(320),100000)]
        X173079_logRlogT_table: list[tuple[float, float]] = [(3.29639174 - 1.34578352*log10t + 0.79354379*log10t**2 - 0.29893059*log10t**3 + 0.03486926*log10t**4, log10t) for log10t in np.linspace(np.log10(1.5),np.log10(320),100000)]

        def binary_search_X173409(target: float) -> float:
            log_target = np.log10(target)
            ok: int = 0
            ng: int = len(X173409_logRlogT_table)
            while abs(ok-ng)>1:
                mid = (ng+ok)//2
                if X173409_logRlogT_table[mid][0] >= log_target:
                    ok = mid
                else:
                    ng = mid
            if ok >= len(X173409_logRlogT_table)-1:
                ok = len(X173409_logRlogT_table)-2
            logT = X173409_logRlogT_table[ok][1] + (X173409_logRlogT_table[ok+1][1]-X173409_logRlogT_table[ok][1]) / (X173409_logRlogT_table[ok+1][0]-X173409_logRlogT_table[ok][0]) * (log_target-X173409_logRlogT_table[ok][0])
            return 10**logT
        def binary_search_X173079(target: float) -> float:
            log_target = np.log10(target)
            ok: int = 0
            ng: int = len(X173079_logRlogT_table)
            while abs(ok-ng)>1:
                mid = (ng+ok)//2
                if X173079_logRlogT_table[mid][0] >= log_target:
                    ok = mid
                else:
                    ng = mid
            if ok >= len(X173079_logRlogT_table)-1:
                ok = len(X173079_logRlogT_table)-2
            logT = X173079_logRlogT_table[ok][1] + (X173079_logRlogT_table[ok+1][1]-X173079_logRlogT_table[ok][1]) / (X173079_logRlogT_table[ok+1][0]-X173079_logRlogT_table[ok][0]) * (log_target-X173079_logRlogT_table[ok][0])
            return 10**logT

        self.CernoxTemp: list[float]
        if cernox_name == "X173409":
            self.CernoxTemp = [binary_search_X173409(abs(resis)) for temp, resis in TR]
        elif cernox_name == "X173079":
            self.CernoxTemp = [binary_search_X173079(abs(resis)) for temp, resis in TR]
        else:
            raise ValueError("cernox name is invalid.")

    def kxxkxy_mode(self, cernox_name: str = "X173409", attr_cor_to_V: list[int] | None = None) -> None:
        """
        
        Args:
            cernox_name (str): Cernox name. ("X173409" or "X173079")
            attr_cor_to_V (list[int] | None): Correspondence between attributes [Cernox_Temp, dTx, dTy] and 0-indexed voltage number. Defaults to [0,1,2].

        """

        if attr_cor_to_V is None:
            attr_cor_to_V = list(range(3))
        Voltages: list[list[list[float]]] = [self.V1, self.V2, self.V3, self.V4, self.V5, self.V6]
        self.cernox_name: str = cernox_name
        self.attr_cor_to_V: list[int] = attr_cor_to_V
        self._set_CernoxTemp(cernox_name, Voltages[attr_cor_to_V[0]])
        self.dTx: list[float] = [v/s for (v,_), s in zip(Voltages[attr_cor_to_V[1]], self.S_TC)]
        self.dTy: list[float] = [v/s for (v,_), s in zip(Voltages[attr_cor_to_V[2]], self.S_TC)]

    def SxxSxy_mode(self, cernox_name: str = "X173409", attr_cor_to_V: list[int] | None = None) -> None:
        """
        
        Args:
            cernox_name (str): Cernox name. ("X173409" or "X173079")
            attr_cor_to_V (list[int] | None): Correspondence between attributes [Cernox_Temp, dTx, dVx, dVy] and 1-indexed voltage number. Defaults to [1,2,3,4].

        """
        if attr_cor_to_V is None:
            attr_cor_to_V = list(range(4))
        else:
            attr_cor_to_V = [i-1 for i in attr_cor_to_V]
        Voltages: list[list[list[float]]] = [self.V1, self.V2, self.V3, self.V4, self.V5, self.V6]
        self.cernox_name: str = cernox_name
        self.attr_cor_to_V: list[int] = attr_cor_to_V
        self._set_CernoxTemp(cernox_name, Voltages[attr_cor_to_V[0]])
        self.dTx: list[float] = [v/s for (v,_), s in zip(Voltages[attr_cor_to_V[1]], self.S_TC)]
        self.dVx: list[float] = [v for (v,_) in Voltages[attr_cor_to_V[2]]]
        self.dVy: list[float] = [v for (v,_) in Voltages[attr_cor_to_V[3]]]

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
        self._operated_filename: str = filename
        self.filename_Seebeck: str = filename_Seebeck
        with open(file=filename, mode="r") as f:
            self.full_contents: list[str] = f.readlines()

        self.TC_TS: list[list[float]] = []
        with open(file=filename_Seebeck, mode="r") as f:
            for line in f.readlines()[1:]:
                s,t = map(float, line.split())
                self.TC_TS.append([t,s])
        self.TC_TS = sorted(self.TC_TS, key=lambda x:x[0]) # 温度順にソート
        
        self.header_length: int = 14

        filename_raw: str = re.sub(r"Exp", r"Raw", self.filename)
        with open(file=filename_raw, mode="r") as f:
            self.len_Index: int = len(f.readlines()) - self.header_length

        self.StartTime: datetime.datetime = datetime.datetime.strptime(re.sub(r"[^\d]+?:", r"", self.full_contents[0]).strip(), '%Y/%m/%d %H:%M:%S')
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

        idx: int = self.header_length
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
        self.Q: list[float] = [self.R*(i**2)/1000 for i in self.Current] # mW
    
    def check_same_condition(self, other: ExpDataExpander) -> bool:
        return self.LTx == other.LTx and self.LTy == other.LTy and self.LVx == other.LVx and \
            self.LVy == other.LVy and self.Width == other.Width and self.Thickness == other.Thickness
    
    def __add__(self, other: ExpDataExpander) -> ExpDataExpander:
        new: ExpDataExpander = self.__class__(self.filename, self.filename_Seebeck)
        if self.check_same_condition(other):
            if self.StartTime < other.StartTime:
                new._operated_filename = self.filename + other.filename
                new.StartTime = self.StartTime
                new.len_Index = self.len_Index + other.len_Index
                new.Index = self.Index + [(s0+self.len_Index, e0+self.len_Index, s1+self.len_Index, e1+self.len_Index) for (s0,e0,s1,e1) in other.Index]
                new.T_PPMS = self.T_PPMS + other.T_PPMS
                new.Field = self.Field + other.Field
                new.Current = self.Current + other.Current
                new.T_Cernox = self.T_Cernox + other.T_Cernox
                new.dVx = self.dVx + other.dVx
                new.dVy = self.dVy + other.dVy
                new.dTx = self.dTx + other.dTx
                new.dTy = self.dTy + other.dTy
                new.dTx0 = self.dTx0 + other.dTx0
                new.dTy0 = self.dTy0 + other.dTy0
                new.dTx1 = self.dTx1 + other.dTx1
                new.dTy1 = self.dTy1 + other.dTy1
                new.errdTx = self.errdTx + other.errdTx
                new.errdTy = self.errdTy + other.errdTy
                new.errdTx0 = self.errdTx0 + other.errdTx0
                new.errdTy0 = self.errdTy0 + other.errdTy0
                new.errdTx1 = self.errdTx1 + other.errdTx1
                new.errdTy1 = self.errdTy1 + other.errdTy1
                new.kxx = self.kxx + other.kxx
                new.errkxx = self.errkxx + other.errkxx
                new.Q = self.Q + other.Q

            else:
                new._operated_filename = other.filename + self.filename
                new.StartTime = other.StartTime
                new.len_Index = other.len_Index + self.len_Index
                new.Index = other.Index + [(s0+other.len_Index, e0+other.len_Index, s1+other.len_Index, e1+other.len_Index) for (s0,e0,s1,e1) in self.Index]
                new.T_PPMS = other.T_PPMS + self.T_PPMS
                new.Field = other.Field + self.Field
                new.Current = other.Current + self.Current
                new.T_Cernox = other.T_Cernox + self.T_Cernox
                new.dVx = other.dVx + self.dVx
                new.dVy = other.dVy + self.dVy
                new.dTx = other.dTx + self.dTx
                new.dTy = other.dTy + self.dTy
                new.dTx0 = other.dTx0 + self.dTx0
                new.dTy0 = other.dTy0 + self.dTy0
                new.dTx1 = other.dTx1 + self.dTx1
                new.dTy1 = other.dTy1 + self.dTy1
                new.errdTx = other.errdTx + self.errdTx
                new.errdTy = other.errdTy + self.errdTy
                new.errdTx0 = other.errdTx0 + self.errdTx0
                new.errdTy0 = other.errdTy0 + self.errdTy0
                new.errdTx1 = other.errdTx1 + self.errdTx1
                new.errdTy1 = other.errdTy1 + self.errdTy1
                new.kxx = other.kxx + self.kxx
                new.errkxx = other.errkxx + self.errkxx
                new.Q = other.Q + self.Q
        else:
            raise ValueError()
        return new

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

        self.header_length: int = 14

        self.StartTime: datetime.datetime = datetime.datetime.strptime(re.sub(r"[^\d]+?:", r"", self.full_contents[0]).strip(), '%Y/%m/%d %H:%M:%S')
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


        idx: int = self.header_length
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
        kxx_symm: list[float] = [(self.kxx[i]+self.kxx[N-1-i])/2 for i in range(N)]
        kxx_symm_err: list[float] = [np.sqrt(self.errkxx[i]**2 + self.errkxx[N-1-i]**2)/2 for i in range(N)]

        Sxx_symm: list[float] = [(self.Sxx[i]+self.Sxx[N-1-i])/2 for i in range(N)]
        Sxx_symm_err: list[float] = [np.sqrt(self.errSxx[i]**2 + self.errSxx[N-1-i]**2)/2 for i in range(N)]

        Sxy_no_symm: list[float] = [ey/dtx/self.LVy*self.LTx * 1e6 for ey,dtx in zip(self.Ey,self.dTx)] # (uV/K)
        Sxy_no_symm_err: list[float] = [abs(sxy) * ((eey/ey)**2 + (edtx/dtx)**2)**0.5 for sxy,ey,eey,dtx,edtx in zip(Sxy_no_symm, self.Ex, self.errEx, self.dTx, self.errdTx)]
        H_: list[float]
        if N % 2 == 1:
            H_ = [-h for h in self.Field[N//2+1:][::-1]] + self.Field[N//2:]
        else:
            H_ = [-h for h in self.Field[N//2:][::-1]] + self.Field[N//2:]
        Sxy_symm = [(Sxy_no_symm[i]-Sxy_no_symm[N-1-i])/2 for i in range(N)]
        Sxy_symm_err: list[float] = [np.sqrt(Sxy_no_symm_err[i]**2 + Sxy_no_symm_err[N-1-i]**2)/2 for i in range(N)]

        return H_, kxx_symm, kxx_symm_err, Sxx_symm, Sxx_symm_err, Sxy_symm, Sxy_symm_err
    
    def symmetrize_positive_half(self) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
        H_, kxx, kxx_err, Sxx, Sxx_err, Sxy, Sxy_err = self.symmetrize()
        N: int = len(H_)
        if H_[0] < 0:
            return H_[N//2:], kxx[N//2:], kxx_err[N//2:], Sxx[N//2:], Sxx_err[N//2:], Sxy[N//2:], Sxy_err[N//2:]
        else:
            return H_[:(N+1)//2][::-1], kxx[:(N+1)//2][::-1], kxx_err[:(N+1)//2][::-1], Sxx[:(N+1)//2][::-1], Sxx_err[:(N+1)//2][::-1], Sxy[:(N+1)//2][::-1], Sxy_err[:(N+1)//2][::-1]
    
    def symmetrize_dT(self) -> tuple[list[float], list[float], list[float]]:
        N: int = len(self.Field)
        dTx_symm: list[float] = [(self.dTx[i]+self.dTx[N-1-i])/2 for i in range(N)]
        dTx_symm_err: list[float] = [np.sqrt(self.errdTx[i]**2 + self.errdTx[N-1-i]**2)/2 for i in range(N)]

        H_: list[float]
        if N % 2 == 1:
            H_ = [-h for h in self.Field[N//2+1:][::-1]] + self.Field[N//2:]
        else:
            H_ = [-h for h in self.Field[N//2:][::-1]] + self.Field[N//2:]
        return H_, dTx_symm, dTx_symm_err


class RemakeExpFromRaw:
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

        self.StartTime: datetime.datetime = self.RawData.StartTime
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
        for sidx0, eidx0, sidx1, eidx1 in self.Index:
            tp0, etp0, h0, eh0, cur0, ecur0, tc0, etc0, vx0, evx0, vy0, evy0 = self.ave_std(eidx0-59, eidx0)
            tp1, etp1, h1, eh1, cur1, ecur1, tc1, etc1, vx1, evx1, vy1, evy1 = self.ave_std(eidx1-59, eidx1)
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


    def ave_std(self, sidx: int, eidx: int) -> tuple[float, ...]:
        slc: slice = slice(sidx, eidx+1)
        attr_cor_to_V: list[int] = self.RawData.attr_cor_to_V
        aveT_PPMS: float = np.average(self.RawData.PPMSTemp[slc])
        errT_PPMS: float = np.std(self.RawData.PPMSTemp[slc])
        aveH: float = np.average(self.RawData.Field[slc])
        errH: float = np.std(self.RawData.Field[slc])
        aveCurrent: float = np.average(self.RawData.HeaterCurrent[slc])
        errCurrent: float = np.std(self.RawData.HeaterCurrent[slc])
        aveT_Cernox: float = np.average(self.RawData.CernoxTemp[slc])
        errT_Cernox: float = np.std(self.RawData.CernoxTemp[slc])
        Vs = [self.RawData.V1, self.RawData.V2, self.RawData.V3, self.RawData.V4, self.RawData.V5, self.RawData.V6]
        aveVx: float = np.average(Vs[attr_cor_to_V[1]][slc])
        errVx: float = np.std(Vs[attr_cor_to_V[1]][slc])
        aveVy: float = np.average(Vs[attr_cor_to_V[2]][slc])
        errVy: float = np.std(Vs[attr_cor_to_V[2]][slc])
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


    def make_new_exp_data_file(self) -> None:
        filename_new: str = re.sub(r"Raw", r"NewExp", self.filename)
        with open(filename_new, mode="w") as f:
            f.write("".join(self.ExpData.full_contents[:self.RawData.header_length]))
            for i, (sidx0, eidx0, sidx1, eidx1) in enumerate(self.Index):
                line = (sidx0, eidx0) + self.ave_std(sidx0, eidx0) + (sidx1, eidx1) + self.ave_std(sidx1, eidx1) + (self.dTx[i], self.errdTx[i], self.kxx[i], self.errkxx[i])
                f.write("\t".join([f"{v:.9e}" for v in line]) + "\n")


            
class RemakeExpFromRawSeebeck:
    def __init__(self, filename: str, filename_Seebeck: str, cernox_name: str = "X173409", attr_cor_to_V: list[int] | None = None) -> None:
        self.filename: str = filename
        self.TC_TS: list[list[float]] = []
        with open(file=filename_Seebeck, mode="r") as f:
            for line in f.readlines()[1:]:
                s,t = map(float, line.split())
                self.TC_TS.append([t,s])
        self.TC_TS = sorted(self.TC_TS, key=lambda x:x[0]) # 温度順にソート

        self.RawData: RawDataExpander = RawDataExpander(filename, filename_Seebeck, attr_cor_to_V)
        self.RawData.SxxSxy_mode(cernox_name, attr_cor_to_V)
        filename_exp: str = re.sub(r"Raw", r"Exp", self.filename)
        self.ExpData: ExpDataExpander = ExpDataExpander(filename_exp, filename_Seebeck)

        self.StartTime: datetime.datetime = self.RawData.StartTime
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
        
        self.R: float = 1000.0
        for sidx0, eidx0, sidx1, eidx1 in self.Index:
            tp0, etp0, h0, eh0, cur0, ecur0, tc0, etc0, vx0, evx0, ex0, eex0, ey0, eey0 = self.ave_std(eidx0-59, eidx0)
            tp1, etp1, h1, eh1, cur1, ecur1, tc1, etc1, vx1, evx1, ex1, eex1, ey1, eey1 = self.ave_std(eidx1-59, eidx1)
            self.T_PPMS.append(tp1)
            self.Field.append(h1)
            self.Current.append(cur1)
            self.T_Cernox.append(tc1)

            self.dVx.append(vx1-vx0)
            dTx: float = (vx1-vx0) / self.Seebeck_at_T(tp1)
            edTx: float = (evx0**2+evx1**2)**0.5 / self.Seebeck_at_T(tp1)
            self.dTx.append(dTx)
            self.errdTx.append(edTx)
            self.dTx0.append((vx0) / self.Seebeck_at_T(tp0))
            self.dTx1.append((vx1) / self.Seebeck_at_T(tp1))
            self.errdTx0.append((evx0) / self.Seebeck_at_T(tp0))
            self.errdTx1.append((evx1) / self.Seebeck_at_T(tp1))

            ex: float = ex1-ex0
            ey: float = ey1-ey0
            eex: float = (eex0**2+eex1**2)**0.5
            eey: float = (eey0**2+eey1**2)**0.5
            self.Ex.append(ex)
            self.Ey.append(ey)
            self.errEx.append(eex)
            self.errEy.append(eey)
            self.Ex0.append(ex0)
            self.Ex1.append(ex1)
            self.Ey0.append(ey0)
            self.Ey1.append(ey1)
            self.errEx0.append(eex0)
            self.errEx1.append(eex1)
            self.errEy0.append(eey0)
            self.errEy1.append(eey1)
            
            # self.Sxx.append(Sxx)
            # self.errSxx.append(eSxx)
            Sxx: float = ex / dTx / self.Lvx * self.LTx * 1e6 # (uv/K)
            self.Sxx.append(Sxx)
            self.errSxx.append(abs(Sxx) * ((eex/ex)**2 + (edTx/dTx)**2)**0.5)
        
        self.kxx: list[float] = [self.R*(i**2)/self.Width/self.Thickness / (dt/self.LTx) for i,dt in zip(self.Current,self.dTx)]
        self.errkxx: list[float] = [k * edtx / dtx for k,dtx,edtx in zip(self.kxx,self.dTx,self.errdTx)]


    def ave_std(self, sidx: int, eidx: int) -> tuple[float, ...]:
        slc: slice = slice(sidx, eidx+1)
        attr_cor_to_V: list[int] = self.RawData.attr_cor_to_V
        aveT_PPMS: float = np.average(self.RawData.PPMSTemp[slc])
        errT_PPMS: float = np.std(self.RawData.PPMSTemp[slc])
        aveH: float = np.average(self.RawData.Field[slc])
        errH: float = np.std(self.RawData.Field[slc])
        aveCurrent: float = np.average(self.RawData.HeaterCurrent[slc])
        errCurrent: float = np.std(self.RawData.HeaterCurrent[slc])
        aveT_Cernox: float = np.average(self.RawData.CernoxTemp[slc])
        errT_Cernox: float = np.std(self.RawData.CernoxTemp[slc])
        Vs = [self.RawData.V1, self.RawData.V2, self.RawData.V3, self.RawData.V4, self.RawData.V5, self.RawData.V6]
        aveVx: float = np.average(Vs[attr_cor_to_V[1]][slc])
        errVx: float = np.std(Vs[attr_cor_to_V[1]][slc])
        aveEx: float = np.average(Vs[attr_cor_to_V[2]][slc])
        errEx: float = np.std(Vs[attr_cor_to_V[2]][slc])
        aveEy: float = np.average(Vs[attr_cor_to_V[3]][slc])
        errEy: float = np.std(Vs[attr_cor_to_V[3]][slc])
        return aveT_PPMS, errT_PPMS, aveH, errH, aveCurrent, errCurrent, aveT_Cernox, errT_Cernox, aveVx, errVx, aveEx, errEx, aveEy, errEy

    def Seebeck_at_T(self, T: float):
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
        kxx_symm: list[float] = [(self.kxx[i]+self.kxx[N-1-i])/2 for i in range(N)]
        kxx_symm_err: list[float] = [np.sqrt(self.errkxx[i]**2 + self.errkxx[N-1-i]**2)/2 for i in range(N)]

        Sxx_symm: list[float] = [(self.Sxx[i]+self.Sxx[N-1-i])/2 for i in range(N)]
        Sxx_symm_err: list[float] = [np.sqrt(self.errSxx[i]**2 + self.errSxx[N-1-i]**2)/2 for i in range(N)]

        Sxy_no_symm: list[float] = [ey/dtx/self.LVy*self.LTx * 1e6 for ey,dtx in zip(self.Ey,self.dTx)] # (uV/K)
        Sxy_no_symm_err: list[float] = [abs(sxy) * ((eey/ey)**2 + (edtx/dtx)**2)**0.5 for sxy,ey,eey,dtx,edtx in zip(Sxy_no_symm, self.Ex, self.errEx, self.dTx, self.errdTx)]
        H_: list[float]
        if N % 2 == 1:
            H_ = [-h for h in self.Field[N//2+1:][::-1]] + self.Field[N//2:]
        else:
            H_ = [-h for h in self.Field[N//2:][::-1]] + self.Field[N//2:]
        Sxy_symm = [(Sxy_no_symm[i]-Sxy_no_symm[N-1-i])/2 for i in range(N)]
        Sxy_symm_err: list[float] = [np.sqrt(Sxy_no_symm_err[i]**2 + Sxy_no_symm_err[N-1-i]**2)/2 for i in range(N)]

        return H_, kxx_symm, kxx_symm_err, Sxx_symm, Sxx_symm_err, Sxy_symm, Sxy_symm_err
    
    def symmetrize_positive_half(self) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
        H_, kxx, kxx_err, Sxx, Sxx_err, Sxy, Sxy_err = self.symmetrize()
        N: int = len(H_)
        if H_[0] < 0:
            return H_[N//2:], kxx[N//2:], kxx_err[N//2:], Sxx[N//2:], Sxx_err[N//2:], Sxy[N//2:], Sxy_err[N//2:]
        else:
            return H_[:(N+1)//2][::-1], kxx[:(N+1)//2][::-1], kxx_err[:(N+1)//2][::-1], Sxx[:(N+1)//2][::-1], Sxx_err[:(N+1)//2][::-1], Sxy[:(N+1)//2][::-1], Sxy_err[:(N+1)//2][::-1]
    
    def symmetrize_dT(self) -> tuple[list[float], list[float], list[float]]:
        N: int = len(self.Field)
        dTx_symm: list[float] = [(self.dTx[i]+self.dTx[N-1-i])/2 for i in range(N)]
        dTx_symm_err: list[float] = [np.sqrt(self.errdTx[i]**2 + self.errdTx[N-1-i]**2)/2 for i in range(N)]

        H_: list[float]
        if N % 2 == 1:
            H_ = [-h for h in self.Field[N//2+1:][::-1]] + self.Field[N//2:]
        else:
            H_ = [-h for h in self.Field[N//2:][::-1]] + self.Field[N//2:]
        return H_, dTx_symm, dTx_symm_err

    def make_new_exp_data_file(self) -> None:
        filename_new: str = re.sub(r"Raw", r"NewExp", self.filename)
        with open(filename_new, mode="w") as f:
            f.write("".join(self.ExpData.full_contents[:self.RawData.header_length]))
            for i, (sidx0, eidx0, sidx1, eidx1) in enumerate(self.Index):
                line = (sidx0, eidx0) + self.ave_std(sidx0, eidx0) + (sidx1, eidx1) + self.ave_std(sidx1, eidx1) + (self.dTx[i], self.errdTx[i], self.kxx[i], self.errkxx[i], self.Sxx[i], self.errSxx[i],)
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
        self.ax1.set_ylabel(ylabel=r"$H$ (Oe)")
        self.ax1.xaxis.set_ticklabels([]) # 目盛を削除
        self.ax1.set_title(self.filename)

        self.ax2: plt.Subplot = fig.add_subplot(223)
        self.ax2.xaxis.set_ticks_position('both')
        self.ax2.yaxis.set_ticks_position('both')
        self.ax2.plot(self.Time, self.CernoxTemp, marker='o', color="blue", markersize=2)
        self.ax2.plot(self.Time, self.PPMSTemp, marker='o', color="red", markersize=2)
        self.ax2.set_xlabel(xlabel=r"Time (sec)")
        self.ax2.set_ylabel(ylabel=r"$T$ (K)"+"\n Cernox:blue\n PPMS:red)")

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

        self.expfit_dTx, = self.ax3.plot([],[], color="cyan", linewidth=2, zorder=1000)

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

        # スライダーの値を直接入力
        def slider_left_changer(var: str, idx: str, mode: str) -> None:
            self.sliders[0].set(self.slider_left_value.get())
            self._slider_scroll()
        def slider_right_changer(var: str, idx: str, mode: str) -> None:
            self.sliders[1].set(self.slider_right_value.get())
            self._slider_scroll()
        self.slider_left_value: tk.DoubleVar = tk.DoubleVar()
        self.slider_left_value.trace_add(mode="write", callback=slider_left_changer)
        self.slider_left: tk.Entry = tk.Entry(slider_control_frame, width=8, textvariable=self.slider_left_value)
        self.slider_left.grid(row=0, column=2)
        self.slider_right_value: tk.DoubleVar = tk.DoubleVar()
        self.slider_right_value.trace_add(mode="write", callback=slider_right_changer)
        self.slider_right: tk.Entry = tk.Entry(slider_control_frame, width=8, textvariable=self.slider_right_value)
        self.slider_right.grid(row=1, column=2)

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
        range_frame: tk.Frame = tk.Frame(analysis_frame, borderwidth=3, relief="ridge")
        lbl_start: tk.Label = tk.Label(range_frame, text="start (s)", foreground="green")
        lbl_start.grid(row=0, column=0)
        self.time_start: tk.Entry = tk.Entry(range_frame, width=8)
        self.time_start.grid(row=1, column=0)
        lbl_end: tk.Label = tk.Label(range_frame, text="end (s)", foreground="red")
        lbl_end.grid(row=0, column=1)
        self.time_end: tk.Entry = tk.Entry(range_frame, width=8)
        self.time_end.grid(row=1, column=1)
        lbl_dt: tk.Label = tk.Label(range_frame, text="end-start (s)", foreground="black")
        lbl_dt.grid(row=0, column=2)
        self.time_dt_value: tk.StringVar = tk.StringVar()
        self.lbl_time_dt_value: tk.Label = tk.Label(range_frame, textvariable=self.time_dt_value, relief="sunken", width=8)
        self.lbl_time_dt_value.grid(row=1, column=2)

        self.start_or_end: int = 0
        self.is_select_range_by_click: tk.BooleanVar = tk.BooleanVar()
        self.is_select_range_by_click.set(True)
        select_range_by_click_cbutton: tk.Checkbutton = tk.Checkbutton(range_frame, variable=self.is_select_range_by_click, text="select range by click", command=self._select_range_by_click_click)
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
        calc_button: tk.Button = tk.Button(analysis_frame, text="Calc", command=self._calc_click)
        calc_button.pack()

        # 計算したデータをsaveさせるボタン
        save_button: tk.Button = tk.Button(analysis_frame, text="Save", command=self._save_click)
        save_button.pack()

        # 選択した範囲から計算したExp形式のデータを標準出力させるFrame
        print_frame: tk.Frame = tk.Frame(analysis_frame, borderwidth=3, relief="ridge")
        self.print_mode: int = 0
        self.t0: float | None = None
        self.t1: float | None = None
        self.data0: list[float] | None = None
        self.data1: list[float] | None = None

        print_button: tk.Button = tk.Button(print_frame, text="Print", command=self._print_click)
        print_button.grid(row=0, column=0)
        lbl_sidx0: tk.Label = tk.Label(print_frame, text="t_s0")
        lbl_sidx0.grid(row=0, column=1)
        lbl_eidx0: tk.Label = tk.Label(print_frame, text="t_e0")
        lbl_eidx0.grid(row=0, column=2)
        lbl_sidx1: tk.Label = tk.Label(print_frame, text="t_s1")
        lbl_sidx1.grid(row=0, column=3)
        lbl_eidx1: tk.Label = tk.Label(print_frame, text="t_e1")
        lbl_eidx1.grid(row=0, column=4)
        self.t_s0: tk.StringVar = tk.StringVar()
        self.lbl_t_s0_value: tk.Label = tk.Label(print_frame, textvariable=self.t_s0, relief="sunken", width=6)
        self.lbl_t_s0_value.grid(row=1, column=1)
        self.t_e0: tk.StringVar = tk.StringVar()
        self.lbl_t_e0_value: tk.Label = tk.Label(print_frame, textvariable=self.t_e0, relief="sunken", width=6)
        self.lbl_t_e0_value.grid(row=1, column=2)
        self.t_s1: tk.StringVar = tk.StringVar()
        self.lbl_t_s1_value: tk.Label = tk.Label(print_frame, textvariable=self.t_s1, relief="sunken", width=6)
        self.lbl_t_s1_value.grid(row=1, column=3)
        self.t_e1: tk.StringVar = tk.StringVar()
        self.lbl_t_e1_value: tk.Label = tk.Label(print_frame, textvariable=self.t_e1, relief="sunken", width=6)
        self.lbl_t_e1_value.grid(row=1, column=4)
        print_frame.pack()

        # 指定されている範囲を f(t) := A exp(-t/τ) でフィッティングするFrame
        expfit_frame: tk.Frame = tk.Frame(analysis_frame, borderwidth=3, relief="ridge")
        expfit_button: tk.Button = tk.Button(expfit_frame, text="ExpFit", command=self._expfit_click)
        expfit_button.grid(row=0, column=0)
        lbl_relaxation_time: tk.Label = tk.Label(expfit_frame, text="τ ln(100) (s)")
        lbl_relaxation_time.grid(row=0, column=1)
        self.relaxation_time: tk.StringVar = tk.StringVar()
        self.lbl_relaxation_time_value: tk.Label = tk.Label(expfit_frame, textvariable=self.relaxation_time, relief="sunken", width=15)
        self.lbl_relaxation_time_value.grid(row=1, column=1)
        expfit_frame.pack()


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
        yax1m, yax1M = min(yax1), max(yax1)
        self.ax1.set_ylim(yax1m-(yax1M-yax1m)*0.05, yax1M+(yax1M-yax1m)*0.05)
        lineax2 = self.ax2.lines[0]
        yax2 = lineax2._yorig[bisect_left(lineax2._xorig,t1):bisect_left(lineax2._xorig,t2)]
        lineax2_2 = self.ax2.lines[1]
        yax2_2 = lineax2_2._yorig[bisect_left(lineax2_2._xorig,t1):bisect_left(lineax2_2._xorig,t2)]
        yax2m, yax2M = min(min(yax2),min(yax2_2)), max(max(yax2),max(yax2_2))
        self.ax2.set_ylim(yax2m-(yax2M-yax2m)*0.05, yax2M+(yax2M-yax2m)*0.05)
        lineax3 = self.ax3.lines[0]
        yax3 = lineax3._yorig[bisect_left(lineax3._xorig,t1):bisect_left(lineax3._xorig,t2)]
        yax3m, yax3M = min(yax3), max(yax3)
        self.ax3.set_ylim(yax3m-(yax3M-yax3m)*0.05, yax3M+(yax3M-yax3m)*0.05)
        lineax4 = self.ax4.lines[0]
        yax4 = lineax4._yorig[bisect_left(lineax4._xorig,t1):bisect_left(lineax4._xorig,t2)]
        yax4m, yax4M = min(yax4), max(yax4)
        self.ax4.set_ylim(yax4m-(yax4M-yax4m)*0.05, yax4M+(yax4M-yax4m)*0.05)

    def _update_start_time(self, x_start: float) -> None:
        self.time_start.delete(0, tk.END)
        self.time_start.insert(tk.END, f"{x_start:.1f}")
        self.start_or_end = 1
        self.ln1_start.set_data([x_start,x_start], self.ax1.get_ylim())
        self.ln2_start.set_data([x_start,x_start], self.ax2.get_ylim())
        self.ln3_start.set_data([x_start,x_start], self.ax3.get_ylim())
        self.ln4_start.set_data([x_start,x_start], self.ax4.get_ylim())
        if self.time_start.get() and self.time_end.get():
            self.time_dt_value.set(f"{float(self.time_end.get())-float(self.time_start.get()):.1f}")

    def _update_end_time(self, x_end: float) -> None:
        self.time_end.delete(0, tk.END)
        self.time_end.insert(tk.END, f"{x_end:.1f}")
        self.start_or_end = 0
        self.ln1_end.set_data([x_end,x_end], self.ax1.get_ylim())
        self.ln2_end.set_data([x_end,x_end], self.ax2.get_ylim())
        self.ln3_end.set_data([x_end,x_end], self.ax3.get_ylim())
        self.ln4_end.set_data([x_end,x_end], self.ax4.get_ylim())
        if self.time_start.get() and self.time_end.get():
            self.time_dt_value.set(f"{float(self.time_end.get())-float(self.time_start.get()):.1f}")

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

    def _print_click(self) -> None:
        """'Print'ボタンをクリックしたときに'start time'から'end time'の各物理量の平均値と
            標準偏差を所定の形式で出力
        """
        self.print_mode = 1
        self.t_s0.set("")
        self.t_e0.set("")
        self.t_s1.set("")
        self.t_e1.set("")

    def _expfit_click(self) -> None:
        """'ExpFit'ボタンをクリックしたときに'start time'から'end time'のdTxを
            f(t) := A exp(-(t-t_start)/τ) + B
            でフィッティングしたときの緩和時間τの計算
        """
        try:
            t_start: float = float(self.time_start.get())
            t_end: float = float(self.time_end.get())
            idx: list[int] = [i for i,t in enumerate(self.Time) if t_start <= t <= t_end]

            
            X: npt.NDArray = np.array([self.Time[i] for i in idx])
            Y: npt.NDArray = np.array([self.dTx[i] for i in idx])
            def f(t: float, A: float, B: float, tau: float) -> float:
                return A * np.exp(-(t-t_start)/tau) + B

            param: tuple[float, float, float] = optimize.curve_fit(f, X, Y)[0] # 返り値はtuple(np.ndarray(#パラメータの値),np.ndarray(#パラメータの標準偏差))
            A, B, tau = param
            tau_1percent: float = -tau * np.log(0.01) # 収束先からの偏差が1％になるまでの時間
            self.relaxation_time.set(f"{tau_1percent:.2f}")
            self.expfit_dTx.set_data(X, f(X, *param))
            print(f"parameters: A:{A:.3f} (K), B:{B:.3f} (K), tau:{tau:.3f}, <1%: {tau_1percent:.3f}")
        except:
            print("failed: fit data")

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
        if self.print_mode == 1:
            if self.t0 is None and self.is_select_range_by_click.get() and self.start_or_end == 0:
                self.t0 = x
                self.t_s0.set(f"{x:.1f}")
            if self.t1 is None and self.is_select_range_by_click.get() and self.start_or_end == 1:
                self.t1 = x
                self.t_e0.set(f"{x:.1f}")
            if self.t0 is not None and self.t1 is not None:
                idx: list[int] = [i for i,t in enumerate(self.Time) if self.t0 <= t <= self.t1]
                sidx: int = idx[0]
                eidx: int = idx[-1]

                self.data0 = [sidx, eidx] + list(self.ave_std(sidx, eidx))
                self.t0 = None
                self.t1 = None
                self.print_mode = 2
        elif self.print_mode == 2:
            if self.t0 is None and self.is_select_range_by_click.get() and self.start_or_end == 0:
                self.t0 = x
                self.t_s1.set(f"{x:.1f}")
            if self.t1 is None and self.is_select_range_by_click.get() and self.start_or_end == 1:
                self.t1 = x
                self.t_e1.set(f"{x:.1f}")
            if self.t0 is not None and self.t1 is not None:
                idx: list[int] = [i for i,t in enumerate(self.Time) if self.t0 <= t <= self.t1]
                sidx: int = idx[0]
                eidx: int = idx[-1]

                self.data1 = [sidx, eidx] + list(self.ave_std(sidx, eidx))
                self.t0 = None
                self.t1 = None
                self.print_mode = 0
                sidx0, eidx0, tp0, etp0, h0, eh0, cur0, ecur0, tc0, etc0, vx0, evx0, vy0, evy0 = self.data0
                sidx1, eidx1, tp1, etp1, h1, eh1, cur1, ecur1, tc1, etc1, vx1, evx1, vy1, evy1 = self.data1
                dtx: float = (vx1-vx0) / self.Seebeck_at_T(tp1)
                errdtx: float = (evx0**2+evx1**2)**0.5 / self.Seebeck_at_T(tp1)
                kxx: float = self.R*(cur1**2)/self.Width/self.Thickness / (dtx / self.LTx)
                errkxx: float = kxx * errdtx / dtx
                print("Start Index0	End Index0	ave T_PPMS0 (K)	err T_PPMS0 (K)	ave H0 (Oe)	err H0 (Oe)	ave Current0 (mA)	err Current0 (mA)	ave T_Cernox0 (K)	err T_Cernox0 (K)	ave Vx0 (V)	err Vx0 (V)	ave Vy0 (V)	err Vy0 (V)	Start Index1	End Index1	ave T_PPMS1 (K)	err T_PPMS1 (K)	ave H1 (Oe)	err H1 (Oe)	ave Current1 (mA)	err Current1 (mA)	ave T_Cernox1 (K)	err T_Cernox1 (K)	ave Vx1 (V)	err Vx1 (V)	ave Vy1 (V)	err Vy1 (V)	dTx (K)	err dTx (K)	kxx (W/Km)	err kxx (W/Km)")
                print("\t".join(map(str, self.data0+self.data1+[dtx, errdtx, kxx, errkxx])))

        if self.is_select_range_by_click.get():
            if self.start_or_end == 0:
                self._update_start_time(x)
                self.fig_canvas.draw()
            else:
                self._update_end_time(x)
                self.fig_canvas.draw()
    
    def ave_std(self, sidx: int, eidx: int) -> tuple[float, ...]:
        slc: slice = slice(sidx, eidx+1)
        attr_cor_to_V: list[int] = self.attr_cor_to_V
        aveT_PPMS: float = np.average(self.PPMSTemp[slc])
        errT_PPMS: float = np.std(self.PPMSTemp[slc])
        aveH: float = np.average(self.Field[slc])
        errH: float = np.std(self.Field[slc])
        aveCurrent: float = np.average(self.HeaterCurrent[slc])
        errCurrent: float = np.std(self.HeaterCurrent[slc])
        aveT_Cernox: float = np.average(self.CernoxTemp[slc])
        errT_Cernox: float = np.std(self.CernoxTemp[slc])
        Vs: list[list[list[float]]] = [self.V1, self.V2, self.V3, self.V4, self.V5, self.V6]
        aveVx: float = np.average(Vs[attr_cor_to_V[1]][slc])
        errVx: float = np.std(Vs[attr_cor_to_V[1]][slc])
        aveVy: float = np.average(Vs[attr_cor_to_V[2]][slc])
        errVy: float = np.std(Vs[attr_cor_to_V[2]][slc])
        return aveT_PPMS, errT_PPMS, aveH, errH, aveCurrent, errCurrent, aveT_Cernox, errT_Cernox, aveVx, errVx, aveVy, errVy

    def excute(self, save_filename: str | None = None, Tx_gain: float = 1) -> None:
        """アプリを実行
        """
        if save_filename is not None:
            self.filename_to_save.insert(tk.END, save_filename)

        self.mainloop()
    
    def delete(self) -> None:
        self.master.destroy()


class HistoryOfTTM(tk.Frame):
    """History of Thermal Transport Measurement
    """
    def __init__(self, filename_Seebeck: str, cernox_name: str, foldername: str | None = None, attr_cor_to_V: list[int] | None = None) -> None:
        if foldername is None:
            foldername = os.getcwd()
        filenames: list[str] = []
        for filename in glob.glob(foldername+"/*"):
            if "Raw" in filename:
                filenames.append(filename)
        filenames = sorted(filenames)
        
        self.start_time_list: list[float] = []
        RDE: RawDataExpander = RawDataExpander(filenames[0], filename_Seebeck)
        RDE.kxxkxy_mode(cernox_name, attr_cor_to_V)
        EDE: ExpDataExpander = ExpDataExpander(re.sub(r"Raw", r"Exp", filenames[0]), filename_Seebeck)
        for filename in filenames[1:]:
            R: RawDataExpander = RawDataExpander(filename, filename_Seebeck)
            R.kxxkxy_mode(cernox_name, attr_cor_to_V)
            E: ExpDataExpander = ExpDataExpander(re.sub(r"Raw", r"Exp", filename), filename_Seebeck)
            RDE = RDE + R
            EDE = EDE + E
            self.start_time_list.append((R.StartTime-RDE.StartTime).total_seconds())
        self.RawData: RawDataExpander = RDE
        self.ExpData: ExpDataExpander = EDE

        root: tk.Tk = tk.Tk()
        tk.Frame.__init__(self, root)

        self.master: tk.Tk = root
        self.master.title(foldername.split('/')[-1])
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

        figsize: tuple[int, int] = (11,10)
        fig: plt.Figure = plt.figure(figsize=figsize)
        plt.subplots_adjust(wspace=0.2, hspace=0, bottom=0.20, top=0.95)
        
        # 磁場
        self.ax1: plt.Subplot = fig.add_subplot(411)
        self.ax1.xaxis.set_ticks_position('both')
        self.ax1.yaxis.set_ticks_position('both')
        H_threshold: float = 1e5
        H: list[float] = [h if -H_threshold < h < H_threshold else H_threshold for h in self.RawData.Field]
        self.ax1.plot(self.RawData.Time, H, marker='o', color="blue", markersize=2)
        self.ax1.set_xlabel(xlabel=r"Time (sec)")
        self.ax1.set_ylabel(ylabel=r"$H$ (Oe)")
        self.ax1.xaxis.set_ticklabels([]) # 目盛を削除
        self.ax1.set_title(foldername.split("/")[-1])
        # 温度
        self.ax2: plt.Subplot = fig.add_subplot(412)
        self.ax2.xaxis.set_ticks_position('both')
        self.ax2.yaxis.set_ticks_position('both')
        T_threshold: float = 500
        T_cernox: list[float] = [t if 0 < t < T_threshold else T_threshold for t in self.RawData.CernoxTemp]
        T_ppms: list[float] = [t if 0 < t < T_threshold else T_threshold for t in self.RawData.PPMSTemp]
        self.ax2.plot(self.RawData.Time, T_cernox, marker='o', color="blue", markersize=2)
        self.ax2.plot(self.RawData.Time, T_ppms, marker='o', color="red", markersize=2)
        self.ax2.set_xlabel(xlabel=r"Time (sec)")
        self.ax2.set_ylabel(ylabel=r"$T$ (K)"+"\n Cernox:blue\n PPMS:red")
        self.ax2.xaxis.set_ticklabels([]) # 目盛を削除
        # dTx
        self.ax3: plt.Subplot = fig.add_subplot(413)
        self.ax3.xaxis.set_ticks_position('both')
        self.ax3.yaxis.set_ticks_position('both')
        self.ax3.plot(self.RawData.Time, self.RawData.dTx, marker='o', color="blue", markersize=2)
        self.ax3.set_xlabel(xlabel=r"Time (sec)")
        self.ax3.set_ylabel(ylabel=r"$\Delta T_{x}$ (K)")
        self.ax3.xaxis.set_ticklabels([]) # 目盛を削除
        # dTy
        self.ax4: plt.Subplot = fig.add_subplot(414)
        self.ax4.xaxis.set_ticks_position('both')
        self.ax4.yaxis.set_ticks_position('both')
        self.ax4.plot(self.RawData.Time, self.RawData.dTy, marker='o', color="blue", markersize=2)
        self.ax4.set_xlabel(xlabel=r"Time (sec)")
        self.ax4.set_ylabel(ylabel=r"$\Delta T_{y}$ (K)")
        self.adjust_tickslabel(0, max(self.RawData.Time))

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

        # 指数関数fitting用
        self.expfit_dTx, = self.ax3.plot([],[], color="cyan", linewidth=2, zorder=1000)

        # Rawファイルごとに区切る線
        ylim1: tuple[float, float] = self.ax1.get_ylim()
        ylim2: tuple[float, float] = self.ax2.get_ylim()
        ylim3: tuple[float, float] = self.ax3.get_ylim()
        ylim4: tuple[float, float] = self.ax4.get_ylim()
        for s in self.start_time_list:
            self.ax1.plot([s,s], ylim1, color="black", linewidth=1)
            self.ax2.plot([s,s], ylim2, color="black", linewidth=1)
            self.ax3.plot([s,s], ylim3, color="black", linewidth=1)
            self.ax4.plot([s,s], ylim4, color="black", linewidth=1)
        self.ax1.set_ylim(ylim1)
        self.ax2.set_ylim(ylim2)
        self.ax3.set_ylim(ylim3)
        self.ax4.set_ylim(ylim4)

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
                    to = max(self.RawData.Time)+1,
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
                    to = max(self.RawData.Time)+1,
                    resolution = 1,
                    tickinterval = 0
                    )
        slider1.pack()
        self.sliders[0].set(0)
        self.sliders[1].set(max(self.RawData.Time)+1)

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
        increment_frame.grid(rowspan=2, column=1, row=0, sticky=tk.N+tk.S)

        # スライダーの値を直接入力
        def slider_left_changer(var: str, idx: str, mode: str) -> None:
            self.sliders[0].set(self.slider_left_value.get())
            self._slider_scroll()
        def slider_right_changer(var: str, idx: str, mode: str) -> None:
            self.sliders[1].set(self.slider_right_value.get())
            self._slider_scroll()
        self.slider_left_value: tk.DoubleVar = tk.DoubleVar()
        self.slider_left_value.trace_add(mode="write", callback=slider_left_changer)
        self.slider_left: tk.Entry = tk.Entry(slider_control_frame, width=8, textvariable=self.slider_left_value)
        self.slider_left.grid(row=0, column=2)
        self.slider_right_value: tk.DoubleVar = tk.DoubleVar()
        self.slider_right_value.trace_add(mode="write", callback=slider_right_changer)
        self.slider_right: tk.Entry = tk.Entry(slider_control_frame, width=8, textvariable=self.slider_right_value)
        self.slider_right.grid(row=1, column=2)

        slider_control_frame.pack()


        #-----------------------------------------------
        ### データ解析のFrame
        analysis_frame: tk.Frame = tk.Frame(self.master, borderwidth=3, relief="ridge")

        # # データsave先のfilename
        # lbl_filename: tk.Label = tk.Label(analysis_frame, text="save filename")
        # lbl_filename.pack()
        # self.filename_to_save: tk.Entry = tk.Entry(analysis_frame, width=20)
        # self.filename_to_save.pack()

        # 測定番号を元に時間範囲を指定するFrame
        exp_idx_frame: tk.Frame = tk.Frame(analysis_frame, borderwidth=3, relief="ridge")
        lbl_exp_idx: tk.Label = tk.Label(exp_idx_frame, text="Exp index", foreground="black")
        lbl_exp_idx.grid(row=0, column=0)
        self.exp_idx: tk.Entry = tk.Entry(exp_idx_frame, width=8)
        self.exp_idx.grid(row=1, column=0)
        lbl_now_idx: tk.Label = tk.Label(exp_idx_frame, text="now index", foreground="black")
        lbl_now_idx.grid(row=0, column=1)
        self.now_idx_value: tk.StringVar = tk.StringVar()
        self.now_idx_value.set(f"0")
        self.lbl_now_idx_value = tk.Label(exp_idx_frame, textvariable=self.now_idx_value, relief="sunken", width=8)
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
        self.lbl_kxx_value: tk.Label = tk.Label(exp_idx_frame, textvariable=self.kxx_value, relief="sunken", width=8)
        self.lbl_kxx_value.grid(row=4, column=0)
        exp_idx_frame.pack(pady=10)

        # データとして使う時間範囲の設定をするためのFrame
        range_frame: tk.Frame = tk.Frame(analysis_frame, borderwidth=3, relief="ridge")
        lbl_start: tk.Label = tk.Label(range_frame, text="start (s)", foreground="green")
        lbl_start.grid(row=0, column=0)
        self.time_start: tk.Entry = tk.Entry(range_frame, width=8)
        self.time_start.grid(row=1, column=0)
        lbl_end: tk.Label = tk.Label(range_frame, text="end (s)", foreground="red")
        lbl_end.grid(row=0, column=1)
        self.time_end: tk.Entry = tk.Entry(range_frame, width=8)
        self.time_end.grid(row=1, column=1)
        lbl_dt: tk.Label = tk.Label(range_frame, text="end-start (s)", foreground="black")
        lbl_dt.grid(row=0, column=2)
        self.time_dt_value: tk.StringVar = tk.StringVar()
        self.lbl_time_dt_value: tk.Label = tk.Label(range_frame, textvariable=self.time_dt_value, relief="sunken", width=8)
        self.lbl_time_dt_value.grid(row=1, column=2)

        self.start_or_end: int = 0
        self.is_select_range_by_click: tk.BooleanVar = tk.BooleanVar()
        self.is_select_range_by_click.set(True)
        select_range_by_click_cbutton: tk.Checkbutton = tk.Checkbutton(range_frame, variable=self.is_select_range_by_click, text="select range by click", command=self._select_range_by_click_click)
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
        calc_button: tk.Button = tk.Button(analysis_frame, text="Calc", command=self._calc_click)
        calc_button.pack()

        # # 計算したデータをsaveさせるボタン
        # save_button: tk.Button = tk.Button(analysis_frame, text="Save", command=self._save_click)
        # save_button.pack()

        # 選択した範囲から計算したExp形式のデータを標準出力させるFrame
        print_frame: tk.Frame = tk.Frame(analysis_frame, borderwidth=3, relief="ridge")
        self.print_mode: int = 0
        self.t0: float | None = None
        self.t1: float | None = None
        self.data0: list[float] | None = None
        self.data1: list[float] | None = None
        print_button: tk.Button = tk.Button(print_frame, text="Print", command=self._print_click)
        print_button.grid(row=0, column=0)
        lbl_sidx0: tk.Label = tk.Label(print_frame, text="t_s0")
        lbl_sidx0.grid(row=0, column=1)
        lbl_eidx0: tk.Label = tk.Label(print_frame, text="t_e0")
        lbl_eidx0.grid(row=0, column=2)
        lbl_sidx1: tk.Label = tk.Label(print_frame, text="t_s1")
        lbl_sidx1.grid(row=0, column=3)
        lbl_eidx1: tk.Label = tk.Label(print_frame, text="t_e1")
        lbl_eidx1.grid(row=0, column=4)
        self.t_s0: tk.StringVar = tk.StringVar()
        self.lbl_t_s0_value: tk.Label = tk.Label(print_frame, textvariable=self.t_s0, relief="sunken", width=6)
        self.lbl_t_s0_value.grid(row=1, column=1)
        self.t_e0: tk.StringVar = tk.StringVar()
        self.lbl_t_e0_value: tk.Label = tk.Label(print_frame, textvariable=self.t_e0, relief="sunken", width=6)
        self.lbl_t_e0_value.grid(row=1, column=2)
        self.t_s1: tk.StringVar = tk.StringVar()
        self.lbl_t_s1_value: tk.Label = tk.Label(print_frame, textvariable=self.t_s1, relief="sunken", width=6)
        self.lbl_t_s1_value.grid(row=1, column=3)
        self.t_e1: tk.StringVar = tk.StringVar()
        self.lbl_t_e1_value: tk.Label = tk.Label(print_frame, textvariable=self.t_e1, relief="sunken", width=6)
        self.lbl_t_e1_value.grid(row=1, column=4)
        print_frame.pack()

        # 指定されている範囲を f(t) := A exp(-t/τ) でフィッティングするFrame
        expfit_frame: tk.Frame = tk.Frame(analysis_frame, borderwidth=3, relief="ridge")
        expfit_button: tk.Button = tk.Button(expfit_frame, text="ExpFit", command=self._expfit_click)
        expfit_button.grid(row=0, column=0)
        lbl_relaxation_time: tk.Label = tk.Label(expfit_frame, text="τ ln(100) (s)")
        lbl_relaxation_time.grid(row=0, column=1)
        self.relaxation_time: tk.StringVar = tk.StringVar()
        self.lbl_relaxation_time_value: tk.Label = tk.Label(expfit_frame, textvariable=self.relaxation_time, relief="sunken", width=15)
        self.lbl_relaxation_time_value.grid(row=1, column=1)
        expfit_frame.pack()

        analysis_frame.pack(pady=20)
        #-----------------------------------------------
    
    def _update_xlim(self, t1: float, t2: float) -> None:
        if t1 > t2:
            return
        self.ax1.set_xlim(t1,t2)
        self.ax2.set_xlim(t1,t2)
        self.ax3.set_xlim(t1,t2)
        self.ax4.set_xlim(t1,t2)

    def _update_ylim(self, t1: float, t2: float) -> None:
        if t1 > t2:
            return
        try:
            lineax1 = self.ax1.lines[0]
            yax1 = lineax1._yorig[bisect_left(lineax1._xorig,t1):bisect_left(lineax1._xorig,t2)]
            yax1m, yax1M = min(yax1), max(yax1)
            if yax1m == yax1M:
                pass
            else:
                self.ax1.set_ylim(yax1m-(yax1M-yax1m)*0.05, yax1M+(yax1M-yax1m)*0.05)
        except:
            print("error in _update_ylim: ax1")
        try:
            lineax2 = self.ax2.lines[0]
            yax2 = lineax2._yorig[bisect_left(lineax2._xorig,t1):bisect_left(lineax2._xorig,t2)]
            lineax2_2 = self.ax2.lines[1]
            yax2_2 = lineax2_2._yorig[bisect_left(lineax2_2._xorig,t1):bisect_left(lineax2_2._xorig,t2)]
            yax2m, yax2M = min(min(yax2),min(yax2_2)), max(max(yax2),max(yax2_2))
            self.ax2.set_ylim(yax2m-(yax2M-yax2m)*0.05, yax2M+(yax2M-yax2m)*0.05)
        except:
            print("error in _update_ylim: ax2")
        try:
            lineax3 = self.ax3.lines[0]
            yax3 = lineax3._yorig[bisect_left(lineax3._xorig,t1):bisect_left(lineax3._xorig,t2)]
            yax3m, yax3M = min(yax3), max(yax3)
            self.ax3.set_ylim(yax3m-(yax3M-yax3m)*0.05, yax3M+(yax3M-yax3m)*0.05)
        except:
            print("error in _update_ylim: ax3")
        try:
            lineax4 = self.ax4.lines[0]
            yax4 = lineax4._yorig[bisect_left(lineax4._xorig,t1):bisect_left(lineax4._xorig,t2)]
            yax4m, yax4M = min(yax4), max(yax4)
            self.ax4.set_ylim(yax4m-(yax4M-yax4m)*0.05, yax4M+(yax4M-yax4m)*0.05)
        except:
            print("error in _update_ylim: ax4")

    def _update_start_time(self, x_start: float) -> None:
        self.time_start.delete(0, tk.END)
        self.time_start.insert(tk.END, f"{x_start:.1f}")
        self.start_or_end = 1
        self.ln1_start.set_data([x_start,x_start], self.ax1.get_ylim())
        self.ln2_start.set_data([x_start,x_start], self.ax2.get_ylim())
        self.ln3_start.set_data([x_start,x_start], self.ax3.get_ylim())
        self.ln4_start.set_data([x_start,x_start], self.ax4.get_ylim())
        if self.time_start.get() and self.time_end.get():
            self.time_dt_value.set(f"{float(self.time_end.get())-float(self.time_start.get()):.1f}")

    def _update_end_time(self, x_end: float) -> None:
        self.time_end.delete(0, tk.END)
        self.time_end.insert(tk.END, f"{x_end:.1f}")
        self.start_or_end = 0
        self.ln1_end.set_data([x_end,x_end], self.ax1.get_ylim())
        self.ln2_end.set_data([x_end,x_end], self.ax2.get_ylim())
        self.ln3_end.set_data([x_end,x_end], self.ax3.get_ylim())
        self.ln4_end.set_data([x_end,x_end], self.ax4.get_ylim())
        if self.time_start.get() and self.time_end.get():
            self.time_dt_value.set(f"{float(self.time_end.get())-float(self.time_start.get()):.1f}")

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
        t2: float = max(self.RawData.Time)+1
        self.sliders[0].set(t1)
        self.sliders[1].set(t2)
        self._update_xlim(t1,t2)
        self._update_ylim(t1,t2)
        self.adjust_tickslabel(t1, t2)
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
        self.adjust_tickslabel(t1, t2)
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
        t1: float = self.RawData.Time[sidx0]
        t2: float = self.RawData.Time[eidx0]
        t3: float = self.RawData.Time[sidx1]
        t4: float = self.RawData.Time[eidx1]
        self.sliders[0].set(t1-30)
        self.sliders[1].set(t4+30)
        self._update_xlim(t1-30,t4+30)
        self._update_ylim(t1-30,t4+30)
        self.adjust_tickslabel(t1-30, t4+30)
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
        idx: list[int] = [i for i,t in enumerate(self.RawData.Time) if t_start <= t <= t_end]
        now_H: list[float] = [self.RawData.Field[i] for i in idx]
        ave_H: float = np.average(now_H)
        std_H: float = np.std(now_H)

        now_Cernox_temp: list[float] = [self.RawData.CernoxTemp[i] for i in idx]
        ave_Cernox_temp: float = np.average(now_Cernox_temp)
        std_Cernox_temp: float = np.std(now_Cernox_temp)
        
        now_dTx: list[float] = [self.RawData.dTx[i] for i in idx]
        ave_dTx: float = np.average(now_dTx)
        std_dTx: float = np.std(now_dTx)

        now_dTy: list[float] = [self.RawData.dTy[i] for i in idx]
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
            idx: list[int] = [i for i,t in enumerate(self.RawData.Time) if t_start <= t <= t_end]

            now_H: list[float] = [self.RawData.Field[i] for i in idx]
            ave_H: float = np.average(now_H)
            std_H: float = np.std(now_H)

            now_Cernox_temp: list[float] = [self.RawData.CernoxTemp[i] for i in idx]
            ave_Cernox_temp: float = np.average(now_Cernox_temp)
            std_Cernox_temp: float = np.std(now_Cernox_temp)
            
            now_dTx: list[float] = [self.RawData.dTx[i] for i in idx]
            ave_dTx: float = np.average(now_dTx)
            std_dTx: float = np.std(now_dTx)

            now_dTy: list[float] = [self.RawData.dTy[i] for i in idx]
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

    def _print_click(self) -> None:
        """'Print'ボタンをクリックしたときに'start time'から'end time'の各物理量の平均値と
            標準偏差を所定の形式で出力
        """
        self.print_mode = 1
        self.t_s0.set("")
        self.t_e0.set("")
        self.t_s1.set("")
        self.t_e1.set("")

    def _expfit_click(self) -> None:
        """'ExpFit'ボタンをクリックしたときに'start time'から'end time'のdTxを
            f(t) := A exp(-(t-t_start)/τ) + B
            でフィッティングしたときの緩和時間τの計算
        """
        try:
            t_start: float = float(self.time_start.get())
            t_end: float = float(self.time_end.get())
            idx: list[int] = [i for i,t in enumerate(self.RawData.Time) if t_start <= t <= t_end]

            X: npt.NDArray = np.array([self.RawData.Time[i] for i in idx])
            Y: npt.NDArray = np.array([self.RawData.dTx[i] for i in idx])
            def f(t: float, A: float, B: float, tau: float) -> float:
                return A * np.exp(-(t-t_start)/tau) + B

            param: tuple[float, float, float] = optimize.curve_fit(f, X, Y)[0] # 返り値はtuple(np.ndarray(#パラメータの値),np.ndarray(#パラメータの標準偏差))
            A, B, tau = param
            tau_1percent: float = -tau * np.log(0.01) # 収束先からの偏差が1％になるまでの時間
            self.relaxation_time.set(f"{tau_1percent:.2f}")
            self.expfit_dTx.set_data(X, f(X, *param))
            print(f"parameters: A:{A:.3f} (K), B:{B:.3f} (K), tau:{tau:.3f}, <1%: {tau_1percent:.3f}")
        except:
            print("failed: fit data")

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
        self.adjust_tickslabel(t1, t2)
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
        if self.print_mode == 1:
            if self.t0 is None and self.is_select_range_by_click.get() and self.start_or_end == 0:
                self.t0 = x
                self.t_s0.set(f"{x:.1f}")
            if self.t1 is None and self.is_select_range_by_click.get() and self.start_or_end == 1:
                self.t1 = x
                self.t_e0.set(f"{x:.1f}")
            if self.t0 is not None and self.t1 is not None:
                idx: list[int] = [i for i,t in enumerate(self.RawData.Time) if self.t0 <= t <= self.t1]
                sidx: int = idx[0]
                eidx: int = idx[-1]

                self.data0 = [sidx, eidx] + list(self.ave_std(sidx, eidx))
                self.t0 = None
                self.t1 = None
                self.print_mode = 2
        elif self.print_mode == 2:
            if self.t0 is None and self.is_select_range_by_click.get() and self.start_or_end == 0:
                self.t0 = x
                self.t_s1.set(f"{x:.1f}")
            if self.t1 is None and self.is_select_range_by_click.get() and self.start_or_end == 1:
                self.t1 = x
                self.t_e1.set(f"{x:.1f}")
            if self.t0 is not None and self.t1 is not None:
                idx: list[int] = [i for i,t in enumerate(self.RawData.Time) if self.t0 <= t <= self.t1]
                sidx: int = idx[0]
                eidx: int = idx[-1]

                self.data1 = [sidx, eidx] + list(self.ave_std(sidx, eidx))
                self.t0 = None
                self.t1 = None
                self.print_mode = 0
                sidx0, eidx0, tp0, etp0, h0, eh0, cur0, ecur0, tc0, etc0, vx0, evx0, vy0, evy0 = self.data0
                sidx1, eidx1, tp1, etp1, h1, eh1, cur1, ecur1, tc1, etc1, vx1, evx1, vy1, evy1 = self.data1
                dtx: float = (vx1-vx0) / self.RawData.Seebeck_at_T(tp1)
                errdtx: float = (evx0**2+evx1**2)**0.5 / self.RawData.Seebeck_at_T(tp1)
                kxx: float = self.RawData.R*(cur1**2)/self.RawData.Width/self.RawData.Thickness / (dtx / self.RawData.LTx)
                errkxx: float = kxx * errdtx / dtx
                print("Start Index0	End Index0	ave T_PPMS0 (K)	err T_PPMS0 (K)	ave H0 (Oe)	err H0 (Oe)	\
                    ave Current0 (mA)	err Current0 (mA)	ave T_Cernox0 (K)	err T_Cernox0 (K)	\
                    ave Vx0 (V)	err Vx0 (V)	ave Vy0 (V)	err Vy0 (V)	\
                    Start Index1	End Index1	ave T_PPMS1 (K)	err T_PPMS1 (K)	ave H1 (Oe)	err H1 (Oe)	\
                    ave Current1 (mA)	err Current1 (mA)	ave T_Cernox1 (K)	err T_Cernox1 (K)	\
                    ave Vx1 (V)	err Vx1 (V)	ave Vy1 (V)	err Vy1 (V)	\
                    dTx (K)	err dTx (K)	kxx (W/Km)	err kxx (W/Km)")
                print("\t".join(map(str, self.data0+self.data1+[dtx, errdtx, kxx, errkxx])))

        if self.is_select_range_by_click.get():
            if self.start_or_end == 0:
                self._update_start_time(x)
                self.fig_canvas.draw()
            else:
                self._update_end_time(x)
                self.fig_canvas.draw()
    
    def ave_std(self, sidx: int, eidx: int) -> tuple[float, ...]:
        slc: slice = slice(sidx, eidx+1)
        attr_cor_to_V: list[int] = self.RawData.attr_cor_to_V
        aveT_PPMS: float = np.average(self.RawData.PPMSTemp[slc])
        errT_PPMS: float = np.std(self.RawData.PPMSTemp[slc])
        aveH: float = np.average(self.RawData.Field[slc])
        errH: float = np.std(self.RawData.Field[slc])
        aveCurrent: float = np.average(self.RawData.HeaterCurrent[slc])
        errCurrent: float = np.std(self.RawData.HeaterCurrent[slc])
        aveT_Cernox: float = np.average(self.RawData.CernoxTemp[slc])
        errT_Cernox: float = np.std(self.RawData.CernoxTemp[slc])
        Vs: list[list[list[float]]] = [self.RawData.V1, self.RawData.V2, self.RawData.V3, self.RawData.V4, self.RawData.V5, self.RawData.V6]
        aveVx: float = np.average(Vs[attr_cor_to_V[1]][slc])
        errVx: float = np.std(Vs[attr_cor_to_V[1]][slc])
        aveVy: float = np.average(Vs[attr_cor_to_V[2]][slc])
        errVy: float = np.std(Vs[attr_cor_to_V[2]][slc])
        return aveT_PPMS, errT_PPMS, aveH, errH, aveCurrent, errCurrent, aveT_Cernox, errT_Cernox, aveVx, errVx, aveVy, errVy

    def adjust_tickslabel(self, t1: float, t2: float) -> None:
        """目盛りのラベルを変更"""
        time_origin_tranformation: int = 60 * self.RawData.StartTime.minute + self.RawData.StartTime.second # 目盛を0分0秒を基準として見やすくする
        t1_int: int = int(t1)
        t2_int: int = int(t2)
        t_range: float = t2_int - t1_int
        dt_list: list[int] = [86400, 43200, 21600, 10800, 7200, 3600, 1800, 900, 600, 300, 180, 120, 60, 30, 15, 10, 5, 3, 2, 1]
        dt_list = dt_list[::-1]
        xticks: list[int] = []
        for dt in dt_list:
            if t_range // dt <= 10:
                xticks = [t for t in range(((t1_int-1)//dt+1)*dt-time_origin_tranformation, t2_int+1, dt) if t1 <= t <= t2]
                break
            elif dt == 86400:
                xticks = [t for t in range(((t1_int-1)//dt+1)*dt-time_origin_tranformation, t2_int+1, dt) if t1 <= t <= t2]
                break
            else:
                continue
        xtickslabel: list[str] = [(self.RawData.StartTime + datetime.timedelta(seconds=t)).strftime('%Y/%m/%d %H:%M:%S') for t in xticks]
        self.ax1.xaxis.set_ticks(xticks)
        self.ax2.xaxis.set_ticks(xticks)
        self.ax3.xaxis.set_ticks(xticks)
        self.ax4.xaxis.set_ticks(xticks)
        self.ax1.xaxis.set_ticklabels([]) # 目盛を削除
        self.ax2.xaxis.set_ticklabels([]) # 目盛を削除
        self.ax3.xaxis.set_ticklabels([]) # 目盛を削除
        self.ax4.xaxis.set_ticklabels(xtickslabel, Rotation=90)

    def excute(self, save_filename: str | None = None) -> None:
        """アプリを実行
        """
        if save_filename is not None:
            self.filename_to_save.insert(tk.END, save_filename)

        self.mainloop()
    
    def delete(self) -> None:
        self.master.destroy()


def triu_inv(U: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    """Upper triangular matrix linear simultaneous equation.

    Args:
        U (npt.NDArray): Upper triangular matrix.
        b (npt.NDArray): Vector.

    Returns:
        npt.NDArray: Answer of the linear simultaneous equation.
    """
    n: int = len(U)
    x: npt.NDArray = np.zeros(n)
    for i in reversed(range(n)):
        s: float = 0.
        for j in range(i+1,n):
            s += U[i][j] * x[j]
        x[i] = (b[i]-s) / U[i][i]
    return x

def tril_inv(L: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    """Lower triangular matrix linear simultaneous equation.

    Args:
        L (npt.NDArray): Lower triangular matrix.
        b (npt.NDArray): Vector.

    Returns:
        npt.NDArray: Answer of the linear simultaneous equation.
    """
    n: int = len(L)
    x: npt.NDArray = np.zeros(n)
    for i in range(n):
        s: float = 0.
        for j in range(i):
            s += L[i][j] * x[j]
        x[i] = (b[i]-s) / L[i][i]
    return x

def Jacobi(A: npt.NDArray, b: npt.NDArray, tol: float = 1e-9):
    """Jacobi method.

    Args:
        A (npt.NDArray): Coefficient matrix.
        b (npt.NDArray): Vector.
        tol (float, optional): Tolerance. Defaults to 1e-9.

    Returns:
        npt.NDArray: Answer of the linear simultaneous equation.
    
    ToDo:
        Pivoting for 0 elements in D.
    """
    k: int = 0
    x_k: npt.NDArray = np.empty_like(b, dtype=np.float64)
    error: float = float('inf')

    A_diag_vector: npt.NDArray = np.diag(A)
    D: npt.NDArray = np.diag(A_diag_vector)
    LU: npt.NDArray = A-D # LU分解ではなく、LU==L+U==A-D
    D_inv: npt.NDArray = np.diag(1/A_diag_vector) # Dの中に0があったらどうするの？

    #while error  > tol: # 更新量がtol以下になったら終了
    while np.linalg.norm(b-np.dot(A,x_k)) > tol: # 残差がtol以下になったら終了
        x: npt.NDArray = np.dot(D_inv, b-np.dot(LU, x_k))
        k += 1
        error = np.linalg.norm(x-x_k)/np.linalg.norm(x)
        x_k = x
    return x

def GaussSeidel(A: npt.NDArray, b: npt.NDArray, tol: float = 1e-9) -> npt.NDArray:
    """Gauss-Seidel method.

    Args:
        A (npt.NDArray): Coefficient matrix.
        b (npt.NDArray): Vector.
        tol (float, optional): Tolerance. Defaults to 1e-9.

    Returns:
        npt.NDArray: Answer of the linear simultaneous equation.
    """
    k: int = 0
    x_k: npt.NDArray = np.empty_like(b, dtype=np.float64)
    error: float = float('inf')

    L: npt.NDArray = np.tril(A) # 下三角行列(対角成分含む)
    U: npt.NDArray = A - L # 上三角行列
    
    # while error > tol: # 更新量がtol以下になったら終了
    while np.linalg.norm(b-np.dot(A,x_k)) > tol: # 残差がtol以下になったら終了
        x: npt.NDArray = tril_inv(L, b-np.dot(U, x_k))
        k += 1
        # error = np.linalg.norm(x-x_k)/np.linalg.norm(x)
        x_k = x
    return x

def TDMA(d: npt.NDArray, u: npt.NDArray, l: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    """Tri-Diagonal Matrix Algorithm for linear simultaneous equation.

    Args:
        d (npt.NDArray): Diagonal elements.
        u (npt.NDArray): Upper diagonal elements.
        l (npt.NDArray): Lower diagonal elements.
        b (npt.NDArray): Right side vector.

    Returns:
        npt.NDArray: Answer of the linear simultaneous equation.
    """
    n: int = len(d)
    P: npt.NDArray = np.zeros(n)
    Q: npt.NDArray = np.zeros(n)
    x: npt.NDArray = np.zeros(n)
    for i in range(n):
        P[i] = -u[i] / (d[i]+l[i]*P[i-1])
        Q[i] = (b[i]-l[i]*Q[i-1]) / (d[i]+l[i]*P[i-1])
    x[-1] = Q[-1]
    for i in range(n-2,-1,-1):
        x[i] = P[i] * x[i+1] + Q[i]
    return x


class ThermalDiffusionSimulation:
    def __init__(self, 
            Lx: float, 
            Ly: float,
            Lz: float,
            Lt: float,
            mol_density: float,
            dx: float = 1.,
            dt: float = 0.01,
        ) -> None:
        """initialize

        Args:
            Lx (float): Length of the sample (μm).
            Ly (float): Width of the sample (μm).
            Lz (float): Thickness of the sample (μm).
            Lt (float): Time length of simulation (s).
            mol_density (float): Mol density of the sample (mol/cm^3).
            dx (float, optional): Discretization length of x direction (μm). Defaluts to 1.
            dt (float, optional): Discretization length of time (s). Defaluts to 0.01.
        """
        self.Lx: float = Lx
        self.Ly: float = Ly
        self.Lz: float = Lz
        self.Lt: float = Lt
        self.mol_density: float = mol_density

        self.dx: float = dx # x方向離散化長さ(μm)
        self.dt: float = dt # 離散化時間単位(s)

        self.Nx: int = int(Lx/self.dx) # x方向グリッド数
        self.Nt: int = int(Lt/self.dt) # t方向グリッド数

    def excute_1D_FTCS(self, T_init: float, Q: float, kappa: float, c: float) -> npt.NDArray:
        """1D thermal diffusion simulation by Forward Time Center Space explicit method.

        Note:
            The discretization width (dx,dt) must be satisfied the CFL condition as below:
                dt/(dx**2) < 1/(2*D),
            where D is diffusion coefficient.
            The boundary conditions are:
                x=0: Neumann condition under steady thermal flux flowing.
                x=L: Dirichlet condition at the constant temperature T_init.
            T_{n}^(t+1) = (1-2α)T_{n}^(t) + α(T_{n-1}^(t) + T_{n+1}^(t))
            
        Args:
            T_init (float): Initial temperature (K).
            Q (float): Heater power (mW).
            kappa (float): Thermal conductivity (W/Km).
            c (float): Specific heat (J/molK).

        Returns:
            npt.NDArray: Temperature distribution T(x,t).
        """
        # 時間前進-空間中心の陽的解法
        Nx: int = self.Nx
        Nt: int = self.Nt
        T: npt.NDarray = np.zeros((Nx, Nt))
        T[:, 0] = T_init
    
        dtdxdx: float = self.dt / (self.dx)**2 # s / (μm)^2
        D: float = kappa / self.mol_density / c * 1e6 # (W/Km) / (mol/cm^3) / (J/molK) = 10^6 * (μm)^2 / s
        alpha: float = D * dtdxdx
        beta: float = Q/self.Ly/self.Lz * self.dx / kappa * 1e3 # mW/μm/μm * μm / (W/Km) = 10^3 * K
        print(dtdxdx, 1/(2*D))
        assert dtdxdx <= 1/(2*D) # 拡散方程式の安定条件
        for t in tqdm.tqdm(range(Nt-1)):
            T[1:Nx-1, t+1] = T[1:Nx-1, t] + alpha * (T[0:Nx-2, t] - 2*T[1:Nx-1, t] + T[2:Nx, t])
        
            # Neumann条件(定常熱流印加バージョン)
            # T[0, t+1] = T[0, t] + Q/self.Ly/self.Lz * self.dx / kappa * 1e3 # mW/μm/μm * μm / (W/Km) = 10^3 * K
            T[0, t+1] = T[1, t+1] + beta
            # Dirichlet条件
            T[Nx-1, t+1] = T_init
        self.T: npt.NDArray = T
        return T
    
    def _excute_1D_implicit2(self, T_init: float, Q: float, kappa: float, c: float) -> npt.NDArray:
        """1D thermal diffusion simulation by iterative implicit method.

        Note:
            The constraint of discretization width (dx,dt) is nothing.
            The boundary conditions are:
                x=0: Neumann condition under steady thermal flux flowing.
                x=L: Dirichlet condition at the constant temperature T_init.
            
        Args:
            T_init (float): Initial temperature (K).
            Q (float): Heater power (mW).
            kappa (float): Thermal conductivity (W/Km).
            c (float): Specific heat (J/molK).

        Returns:
            npt.NDArray: Temperature distribution T(x,t).
        """
        # 反復法の陰的解法
        Nx: int = self.Nx
        Nt: int = self.Nt
        T: npt.NDarray = np.zeros((Nx, Nt))
        T[:, 0] = T_init
    
        dtdxdx: float = self.dt / (self.dx)**2 # s / (μm)^2
        D: float = kappa / self.mol_density / c * 1e6 # (W/Km) / (mol/cm^3) / (J/molK) = 10^6 * (μm)^2 / s
        alpha: float = D * dtdxdx # dimensionless
        beta: float = Q/self.Ly/self.Lz * self.dx / kappa * 1e3 # mW/μm/μm * μm / (W/Km) = 10^3 * K
        for t in tqdm.tqdm(range(Nt-1)):
            v: npt.NDArray = np.copy(T[:, t])
            for n in range(1000):
                w: npt.NDArray = np.copy(T[:, t+1])
                T[1:Nx-1, t+1] = 1 / (1+2*alpha) * (v[1:Nx-1] + alpha*(w[2:Nx]+w[0:Nx-2]))
                if n % 10 == 0:
                    if np.sqrt(np.sum(T[:, t+1] - w) ** 2) / np.sum(w ** 2) < 1e-6:
                        break

                # boundary condition
                # Neumann条件(定常熱流印加バージョン)
                T[0, t+1] = T[1, t+1] + beta
                # Dirichlet条件
                T[Nx-1, t+1] = T_init
        self.T: npt.NDArray = T
        return T

    def excute_1D_implicit(self, T_init: float, Q: float, kappa: float, c: float) -> npt.NDArray:
        """1D thermal diffusion simulation by implicit method with TDMA.

        Note:
            The constraint of discretization width (dx,dt) is nothing.
            The boundary conditions are:
                x=0: Neumann condition under steady thermal flux flowing.
                x=L: Dirichlet condition at the constant temperature T_init.
            (1+2α)T_{n}^(t+1) - α(T_{n-1}^(t+1) + T_{n+1}^(t+1)) = T_{n}^(t)
            
        Args:
            T_init (float): Initial temperature (K).
            Q (float): Heater power (mW).
            kappa (float): Thermal conductivity (W/Km).
            c (float): Specific heat (J/molK).

        Returns:
            npt.NDArray: Temperature distribution T(x,t).
        """
        # 三重対角行列の陰的解法
        Nx: int = self.Nx
        Nt: int = self.Nt
        T: npt.NDarray = np.zeros((Nx, Nt))
        T[:, 0] = T_init
    
        dtdxdx: float = self.dt / (self.dx)**2 # s / (μm)^2
        D: float = kappa / self.mol_density / c * 1e6 # (W/Km) / (mol/cm^3) / (J/molK) = 10^6 * (μm)^2 / s
        alpha: float = D * dtdxdx # dimensionless
        beta: float = Q/self.Ly/self.Lz * self.dx / kappa * 1e3 # mW/μm/μm * μm / (W/Km) = 10^3 * K
        d: npt.NDArray = np.array([1.]+[1+2*alpha]*(Nx-2)+[1.])
        u: npt.NDArray = np.array([-1.]+[-alpha]*(Nx-2)+[0.])
        l: npt.NDArray = np.array([0.]+[-alpha]*(Nx-2)+[0.])
        for t in tqdm.tqdm(range(Nt-1)):
            b: npt.NDArray = np.copy(T[:, t])
            b[0] = beta
            T[:, t+1] = TDMA(d, u, l, b)
        self.T = T
        return T

    def excute_1D_Crank_Nicolson(self, T_init: float, Q: float, kappa: float, c: float) -> npt.NDArray:
        """1D thermal diffusion simulation by Crank-Nicolson method.

        Note:
            The constraint of discretization width (dx,dt) is nothing.
            The boundary conditions are:
                x=0: Neumann condition under steady thermal flux flowing.
                x=L: Dirichlet condition at the constant temperature T_init.
            (1+2θα)T_{n}^(t+1) - θα(T_{n-1}^(t+1) + T_{n+1}^(t+1)) = (1-2(1-θ)α)T_{n}^(t) + (1-θ)α(T_{n-1}^(t) + T_{n+1}^(t))
            
        Args:
            T_init (float): Initial temperature (K).
            Q (float): Heater power (mW).
            kappa (float): Thermal conductivity (W/Km).
            c (float): Specific heat (J/molK).

        Returns:
            npt.NDArray: Temperature distribution T(x,t).
        """
        # Crank-Nicolson法の陰的解法
        Nx: int = self.Nx
        Nt: int = self.Nt
        T: npt.NDarray = np.zeros((Nx, Nt))
        T[:, 0] = T_init

        theta: float = 1/2
        dtdxdx: float = self.dt / (self.dx)**2 # s / (μm)^2
        D: float = kappa / self.mol_density / c * 1e6 # (W/Km) / (mol/cm^3) / (J/molK) = 10^6 * (μm)^2 / s
        alpha: float = D * dtdxdx # dimensionless
        beta: float = Q/self.Ly/self.Lz * self.dx / kappa * 1e3 # mW/μm/μm * μm / (W/Km) = 10^3 * K
        print(D, dtdxdx, alpha, beta)
        d: npt.NDArray = np.array([1.]+[1+2*theta*alpha]*(Nx-2)+[1.])
        u: npt.NDArray = np.array([-1.]+[-theta*alpha]*(Nx-2)+[0.])
        l: npt.NDArray = np.array([0.]+[-theta*alpha]*(Nx-2)+[0.])
        for t in tqdm.tqdm(range(Nt-1)):
            b: npt.NDArray = np.copy(T[:, t]) * (1 - 2*(1-theta)*alpha)
            b[1:] += T[:-1, t] * (1-theta)*alpha
            b[:-1] += T[1:, t] * (1-theta)*alpha
            b[0] = beta # boundary condition
            b[-1] = T_init # boundary condition
            T[:, t+1] = TDMA(d, u, l, b)
        self.T = T
        return T

    def _excutor(self, T_init: float, Q: float, kappa: float, c: float, mode: str = "Crank-Nicolson") -> tuple[npt.NDArray, npt.NDArray]:
        X: npt.NDArray = np.linspace(0, self.Lx, self.Nx)
        T: npt.NDArray
        if mode is None and "T" in dir(self):
            T = self.T
        else:
            if mode == "FTCS":
                T = self.excute_1D_FTCS(T_init, Q, kappa, c)
            elif mode == "implicit":
                T = self.excute_1D_implicit(T_init, Q, kappa, c)
            elif mode == "Crank-Nicolson":
                T = self.excute_1D_Crank_Nicolson(T_init, Q, kappa, c)
            else:
                raise ValueError
        return X, T
    
    def snapshot(self, t: float, T_init: float, Q: float, kappa: float, c: float, mode: str = "Crank-Nicolson") -> None:
        """Visualize temperature distribution at the selected time.

        Args:
            t (float): Time (s).
            T_init (float): Initial temperature (K).
            Q (float): Heater power (mW).
            kappa (float): Thermal conductivity (W/Km).
            mode (str, optional): Simulation mode. Defaults to "Crank-Nicolson".

        Raises:
            ValueError: mode must be in ["FTCS", "implicit", "Crank-Nicolson", "reuse"]
        """
        X, T = self._excutor(T_init, Q, kappa, c, mode)
        fig: plt.Figure = plt.figure()
        fig.set_dpi(100)
        ax: plt.Subplot = fig.add_subplot(1,1,1)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        min_y: float = np.min(T)
        max_y: float = np.max(T)*1.1
        ax.plot(X, T[:,t], color="blue")
        ax.text(0.05, 0.9, f"t = {(t+1)*self.dt:.2f}/{self.Lt:.2f},", transform=ax.transAxes)
        ax.set_xlim(0, self.Lx)
        ax.set_ylim(min_y, max_y)
        ax.set_xlabel(r"$x$ ($\mathrm{\mu}$m)")
        ax.set_ylabel(r"$T$ (K)")
        plt.show()

    def animation(self, T_init: float, Q: float, kappa: float, c: float, mode: str = "Crank-Nicolson") -> None:
        """Make an animation of temperature distribution.

        Args:
            T_init (float): Initial temperature (K).
            Q (float): Heater power (mW).
            kappa (float): Thermal conductivity (W/Km).
            c (float): Specific heat (J/molK).
            mode (str, optional): Simulation mode. Defaults to "Crank-Nicolson".

        Raises:
            ValueError: mode must be in ["FTCS", "implicit", "Crank-Nicolson", "reuse"]
        """
        X, T = self._excutor(T_init, Q, kappa, c, mode)
        fig: plt.Figure = plt.figure()
        fig.set_dpi(100)
        ax: plt.Subplot = fig.add_subplot(1,1,1)
        min_y: float = np.min(T)
        max_y: float = np.max(T)*1.1
        ax.set_xlim(0, self.Lx)
        ax.set_ylim(min_y, max_y)
        ax.set_xlabel(r"$x$ ($\mathrm{\mu}$m)")
        ax.set_ylabel(r"$T$ (K)")
        def animate(t):
            ax.clear()
            ax.plot(X, T[:,t])
            ax.text(0.05, 0.9, f"t = {(t+1)*self.dt:.2f}/{self.Lt:.2f},", transform=ax.transAxes)
            ax.set_ylim(min_y, max_y)
        anim = matplotlib.animation.FuncAnimation(fig, animate, frames=range(0,self.Nt,self.Nt//100), interval=1, repeat=True)
        plt.show()

    def exp_fit(self, X: npt.NDArray, Y: npt.NDArray) -> tuple[npt.NDArray, tuple[float, float, float]]:
        """Exponential fitting.

        Args:
            X (npt.NDArray): X.
            Y (npt.NDArray): Y.

        Returns:
            tuple[npt.NDArray, tuple[float, float, float]]: Result of fitting and used parameters.
        """
        def f(t: float, A: float, B: float, tau: float) -> float:
            return A * np.exp(-t/tau) + B
        param: tuple[float, float, float] = optimize.curve_fit(f, X, Y)[0]
        return f(X, *param), param
    
    def cal_tau(self, X: npt.NDArray, Y: npt.NDArray) -> float:
        """Get the relaxation time.

        Args:
            X (npt.NDArray): X.
            Y (npt.NDArray): Y.

        Returns:
            float: Relaxation time.
        """
        _, (_, _, tau) = self.exp_fit(X, Y)
        return tau
    
    def dT(self, lx1: float, lx2: float, T_init: float, Q: float, kappa: float, c: float, mode: str = "Crank-Nicolson") -> None:
        """Visualize dT vs time.

        Args:
            lx1 (float): Left side point (μm).
            lx2 (float): Right side point (μm).
            T_init (float): Initial temperature (K).
            Q (float): Heater power (mW).
            kappa (float): Thermal conductivity (W/Km).
            c (float): Specific heat (J/molK).
            mode (str, optional): Simulation mode. Defaults to "Crank-Nicolson".
        """
        _, T = self._excutor(T_init, Q, kappa, c, mode)
        idx1: int = int(lx1 / self.dx)
        idx2: int = int(lx2 / self.dx)

        X: npt.NDArray = np.linspace(0, self.Lt, self.Nt)
        Y: npt.NDArray = T[idx1].T - T[idx2].T

        fig: plt.Figure = plt.figure()
        ax: plt.Subplot = fig.add_subplot(1,1,1)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.plot(X, Y, marker="o", color="red")
        Y_fit, (_, _, tau) = self.exp_fit(X, Y)
        ax.plot(X, Y_fit, color="black", linewidth=1, label=f"τ*ln(100): {tau*np.log(100):.2f} (s)")
        ax.set_xlim(0, self.Lt)
        ax.set_xlabel(r"$t$ (s)")
        ax.set_ylabel(r"$\Delta T$ (K)")
        ax.legend()
        plt.show()
    
    def tau_vs_c(self, lx1: float, lx2: float, T_init: float, Q: float, kappa: float, mode: str = "Crank-Nicolson") -> None:
        """Visualize tau vs specific heat.

        Args:
            lx1 (float): Left side point (μm).
            lx2 (float): Right side point (μm).
            T_init (float): Initial temperature (K).
            Q (float): Heater power (mW).
            kappa (float): Thermal conductivity (W/Km).
            mode (str, optional): Simulation mode. Defaults to "Crank-Nicolson".
        """
        idx1: int = int(lx1 / self.dx)
        idx2: int = int(lx2 / self.dx)
        X: npt.NDArray = np.linspace(0, self.Lt, self.Nt)
        c_max: float = 100
        specific_heat: npt.NDArray = np.linspace(0.01, c_max, 50)
        tau_list: npt.NDArray = np.array([])
        for c in specific_heat:
            _, T = self._excutor(T_init, Q, kappa, c, mode)
            Y: npt.NDArray = T[idx1].T - T[idx2].T
            tau: float = self.cal_tau(X, Y)
            tau_list = np.append(tau_list, tau)
        
        fig: plt.Figure = plt.figure()
        ax: plt.Subplot = fig.add_subplot(1,1,1)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.plot(specific_heat, tau_list, marker="o", color="red")
        ax.set_title(
            fr"$T_{{\mathrm{{init}}}}$: {T_init}K, Q: {Q}mW, $\kappa$: {kappa}W/Km"+"\n"+
            fr"$L_{{x}}$: {self.Lx}, $L_{{y}}$: {self.Ly}, $L_{{z}}$: {self.Lz}, $L_{{t}}$: {self.Lt}"+"\n"+
            fr"$dx$: {self.dx}, $dt$: {self.dt}, $l_{{x1}}$: {lx1}, $l_{{x2}}$: {lx2}"
        )
        ax.set_xlim(0, c_max)
        ax.set_xlabel(r"$c$ (J/molK)")
        ax.set_ylabel(r"$\tau$ (s)")
        ax.legend()
        fig.savefig("./tau_vs_c.png", bbox_inches="tight", transparent=True, dpi=300)
        plt.show()

def main() -> None:
    pass
    return

if __name__ == "__main__":
    main()

