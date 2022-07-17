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
import re

import matplotlib.pyplot as plt # type: ignore
import numpy as np
import pymatgen # type: ignore
from pymatgen.io.cif import CifParser # type: ignore
import pymatgen.analysis.diffraction.xrd # type: ignore
from scipy.optimize import curve_fit # type: ignore
from scipy.stats import norm # type: ignore
from scipy import integrate # type: ignore

from ..core import Crystal, PhysicalConstant


plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["legend.framealpha"] = 0

def compare_powder_Xray_experiment_with_calculation(experimental_data_filename: str, cif_filename: str, material: Crystal | None = None, unbackground: bool = False, issave: bool = False) -> tuple[plt.Figure, plt.Subplot]:
    """Compare experimental intensity data of powder X-ray diffraction with theoretical intensity distribution.

    Notes:
        Removing background method should be improved.
        The argument 'material' will be removed in future.

    Args:
        experimental_data_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".ras".
        cif_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".cif".
        material (Crystal | None): `Crystal` instance of the measurement object.
        unbackground (bool): If True, remove the background with piecewise linear interpolation.

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
                flag = False
            elif flag:
                data.append(list(map(float, line.strip().split())))

    N: int = len(data)
    two_theta: list[float] = [d[0] for d in data] # データは2θ

    intensity: list[float] = [d[1] for d in data]
    
    neg: list[float] = [i for i in intensity if i<0]
    assert len(neg)==0 # 負のintensityをもつ壊れたデータがないことを確認
    
    neighbor_num: int = 50 # peak(極大値の中でも急激に増加するもの)判定で参照する近傍のデータ点数
    magnification: int = 4 # 周囲neighbor_num個の強度の最小値に比べて何倍大きければpeakと見なすかの閾値
    half: int = neighbor_num//2 # 中間点
    que: deque[float] = deque([])
    descending_intensity: list[tuple[float, int, float, float]] = []
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
    background_points: list[list[float]] = [[x,y] for _,_,x,y in sorted(descending_intensity[len(descending_intensity)//5*3:], key=lambda X:X[2])]
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
    background_x: list[float] = [two_theta[0]] + [x for x,y in background_points] + [two_theta[-1]]
    background_y: list[float] = [intensity[0]] + [y for x,y in background_points] + [intensity[-1]]
    
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
    # x = np.arange(10,90)
    # y = [interpolate_bg(i) for i in x]
    # plt.plot(x,y)
    # plt.scatter(background_x,background_y)
    # plt.show()
    if unbackground:
        intensity_unbackground: list[float] = [its-interpolate_bg(tht) for tht,its in zip(two_theta,intensity)]
    else:
        intensity_unbackground = intensity
    exp_tops: list[list[float]] = [[x,y] for _,_,x,y in sorted(descending_intensity[:display_num], key=lambda z:z[3], reverse=True)]
        
    # ここから粉末X線回折の理論計算
    parser: pymatgen.io.cif.CifParser = CifParser(cif_filename)
    structure: pymatgen.core.structure.Structure = parser.get_structures()[0]
    analyzer: pymatgen.analysis.diffraction.xrd.XRDCalculator = pymatgen.analysis.diffraction.xrd.XRDCalculator(wavelength='CuKa')
    diffraction_pattern: pymatgen.analysis.diffraction.core.DiffractionPattern = analyzer.get_pattern(structure)
    cal_tops: list[list[float]] = [[x,y] for _,_,x,y in sorted(zip(diffraction_pattern.d_hkls, diffraction_pattern.hkls, diffraction_pattern.x, diffraction_pattern.y), key=lambda z:z[3], reverse=True)[:display_num]]
    for d_hkl, hkl, x, y in sorted(zip(diffraction_pattern.d_hkls, diffraction_pattern.hkls, diffraction_pattern.x, diffraction_pattern.y), key=lambda z:z[3], reverse=True)[:display_num]:
        print(f"{x:.3f}, {hkl}, {d_hkl:.3f}")

    fig: plt.Figure = plt.figure(figsize=(12,6))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    def LSM(x, y, linear=False): # x: list, y: list
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
    normalized_intensity: list[float] = [a*x for x in intensity_unbackground]

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
            fig.savefig(f"./{material.name}_pXray_unbackground.png", transparent=True)
        else:
            fig.savefig(f"./{material.name}_pXray.png", transparent=True)
    return fig, ax

def compare_powder_Xray_experiment_with_calculation_of_some_materials(experimental_data_filename: str, cif_filename_list: list[str], unbackground: bool = False, issave: bool = False) -> tuple[plt.Figure, plt.Subplot]:
    """Compare experimental intensity data of powder X-ray diffraction with theoretical intensity distribution of some materials.

    Notes:
        Removing background method should be improved.

    Args:
        experimental_data_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".ras".
        cif_filename (list[str]): list of input file name (if necessary, add file path to the head). The suffix of `filename` must be ".cif".
        unbackground (bool): If True, remove the background with piecewise linear interpolation.

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
                flag = False
            elif flag:
                data.append(list(map(float, line.strip().split())))

    N: int = len(data)
    two_theta: list[float] = [d[0] for d in data] # データは2θ

    intensity: list[float] = [d[1] for d in data]
    
    neg: list[float] = [i for i in intensity if i<0]
    assert len(neg)==0 # 負のintensityをもつ壊れたデータがないことを確認
    
    neighbor_num: int = 50 # peak(極大値の中でも急激に増加するもの)判定で参照する近傍のデータ点数
    magnification: int = 4 # 周囲neighbor_num個の強度の最小値に比べて何倍大きければpeakと見なすかの閾値
    half: int = neighbor_num//2 # 中間点
    que: deque[float] = deque([])
    descending_intensity: list[tuple[float, int, float, float]] = []
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
    background_points: list[list[float]] = [[x,y] for _,_,x,y in sorted(descending_intensity[len(descending_intensity)//5*3:], key=lambda X:X[2])]
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
    background_x: list[float] = [two_theta[0]] + [x for x,y in background_points] + [two_theta[-1]]
    background_y: list[float] = [intensity[0]] + [y for x,y in background_points] + [intensity[-1]]
    
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
    # x = np.arange(10,90)
    # y = [interpolate_bg(i) for i in x]
    # plt.plot(x,y)
    # plt.scatter(background_x,background_y)
    # plt.show()
    if unbackground:
        intensity_unbackground: list[float] = [its-interpolate_bg(tht) for tht,its in zip(two_theta,intensity)]
    else:
        intensity_unbackground = intensity
    exp_tops: list[list[float]] = [[x,y] for _,_,x,y in sorted(descending_intensity[:display_num], key=lambda z:z[3], reverse=True)]

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
        parser: pymatgen.io.cif.CifParser = CifParser(cif_filename)
        structure: pymatgen.core.structure.Structure = parser.get_structures()[0]
        analyzer: pymatgen.analysis.diffraction.xrd.XRDCalculator = pymatgen.analysis.diffraction.xrd.XRDCalculator(wavelength='CuKa')
        diffraction_pattern: pymatgen.analysis.diffraction.core.DiffractionPattern = analyzer.get_pattern(structure)
        cal_tops: list[list[float]] = [[x,y] for _,_,x,y in sorted(zip(diffraction_pattern.d_hkls, diffraction_pattern.hkls, diffraction_pattern.x, diffraction_pattern.y), key=lambda z:z[3], reverse=True)[:display_num]]
        print(f"####### {material.name} start #########")
        for d_hkl, hkl, x, y in sorted(zip(diffraction_pattern.d_hkls, diffraction_pattern.hkls, diffraction_pattern.x, diffraction_pattern.y), key=lambda z:z[3], reverse=True)[:display_num]:
            print(f"{x:.3f}, {hkl}, {d_hkl:.3f}")
        print(f"####### {material.name} end #########")

        def LSM(x, y, linear=False): # x: list, y: list
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
        # 理論値を実験値に合わせて定数倍する．倍率は上位ピーク強度から最小二乗法
        a: float = LSM([y for x,y in cal_tops], [y for x,y in exp_tops])[1]

        # 理論計算
        theor_x = np.arange(0,90,0.001)
        theor_y = np.zeros_like(theor_x)
        for tx, ty in zip(diffraction_pattern.x, diffraction_pattern.y):
            theor_y[bisect_left(theor_x,tx)] = a * ty # 理論値を実験値に合わせて定数倍
        Gaussian = norm.pdf(np.arange(-1,1,0.001),0,0.05)
        Gaussian /= Gaussian[len(Gaussian)//2]
        theor_y = np.convolve(theor_y, Gaussian, mode="same")        
        ax.plot(theor_x, theor_y, linewidth=1.2, label=f"calc. {material.name}", color="red", zorder=0)
        max_value = max(max(intensity_unbackground), max(theor_y))
        for x in diffraction_pattern.x:
            ax.plot([x,x], [-max_value*0.14,-max_value*0.06], color="green", linewidth=1, zorder=1)
        ax.plot([x,x], [-max_value*0.14,-max_value*0.06], color="green", linewidth=1, label="Bragg peak", zorder=1)

        # 実験データ
        ax.plot(two_theta, intensity_unbackground, label="obs.", color="blue", marker="o", markersize=1.5, linewidth=0.5, zorder=2)

        ax.set_ylabel("intensity [a.u.]")
        ax.legend()
        ax.set_xticks(range(0,100,10))
        ax.set_xlim(0,90)
        ax.set_ylim(-max_value*0.2, max_value*1.1)
        ax.yaxis.set_ticklabels([]) # 目盛を削除
        if num != len(cif_filename_list)-1:
            ax.xaxis.set_ticklabels([]) # 目盛を削除
        else:
            ax.set_xlabel(r"$2\theta\, [{}^{\circ}]$")

    plt.show()
    if issave:
        if unbackground:
            fig.savefig(f"./pXray_unbackground_{'_'.join([material.name for material in materials_list])}.png", transparent=True)
        else:
            fig.savefig(f"./pXray_{'_'.join([material.name for material in materials_list])}.png", transparent=True)
    return fig, ax


def _compare_powder_Xray_experiment_with_calculation(experimental_data_filename: str, cif_filename: str, material: Crystal | None = None) -> tuple[plt.Figure, plt.Subplot]:
    """Compare experimental intensity data of powder X-ray diffraction with theoretical intensity distribution.

    Args:
        experimental_data_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".ras".
        cif_filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".cif".
        material (Crystal | None): `Crystal` instance of the measurement object.

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
                flag = False
            elif flag:
                data.append(list(map(float, line.strip().split())))

    N: int = len(data)
    two_theta: list[float] = [d[0] for d in data] # データは2θ
    intensity: list[float] = [d[1] for d in data]
    normalized_intensity: list[float] = [d[1]/max(intensity)*100 for d in data]
    neg: list[float] = [i for i in intensity if i<0]
    assert len(neg)==0 # 負のintensityをもつ壊れたデータがないことを確認
    
    neighbor_num: int = 20 # peak(極大値の中でも急激に増加するもの)判定で参照する近傍のデータ点数
    magnification: int = 4 # 周囲neighbor_num個の強度の最小値に比べて何倍大きければpeakと見なすかの閾値
    
    half: int = neighbor_num//2 # 中間点
    que: deque[float] = deque([])
    descending_intensity: list[tuple[float, int, float, float]] = []
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
    for d_hkl, hkl, x, y in sorted(zip(diffraction_pattern.d_hkls, diffraction_pattern.hkls, diffraction_pattern.x, diffraction_pattern.y), key=lambda z:z[3], reverse=True)[:10]: # type: ignore
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
    def fD(t: float):
        def integrand(x):
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

def demagnetizating_factor_ellipsoid(a: float, b: float, c: float) -> tuple[float, float, float]:
    """Calculating demagnetizating factor of ellipsoid 2a x 2b x 2c.

    Args:
        a (float): Length of an edge (arb. unit).
        b (float): Length of an edge (arb. unit).
        c (float): Length of an edge (arb. unit).
    
    Returns:
        (tuple[float]): Demagnetizating factor Nx, Ny, Nz.
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
# def demagnetizating_factor_rectangular_prism_Nf(a: float, b: float, c: float) -> float:
#     """Calculating demagnetizating factor of rectangular prism axbxc.

#     Thesis:
#         D.-X. Chen et al., IEEE Transactions on Magnetics 38, 4 (2002).

#     Args:
#         a (float): Length of an edge (arb. unit).
#         b (float): Length of an edge (arb. unit).
#         c (float): Length of an edge (arb. unit).
    
#     Returns:
#         (float): Demagnetizating factor.
#     """
#     def Ff(u: float, v: float) -> float:
#         return u * np.log(c**2*(8*u**2+4*v**2+c**2+4*u*np.sqrt(4*u**2+4*v**2+c**2)) / (4*v**2+c**2) / (8*u**2+c**2+4*u*np.sqrt(4*u**2+c**2)))

#     F1: float = np.sqrt(4*a**2+c**2) + np.sqrt(4*b**2+c**2) - np.sqrt(4*a**2+4*b**2+c**2) - c
#     Nf: float = 2/np.pi * np.arctan(4*a*b/(c*np.sqrt(4*a**2+4*b**2+c**2))) + (c/(2*np.pi*a*b))*(F1+Ff(a,b)+Ff(b,a))
#     return Nf

# def demagnetizating_factor_rectangular_prism_Nm(a: float, b: float, c: float) -> float:
#     """Calculating demagnetizating factor of rectangular prism axbxc.

#     Thesis:
#         D.-X. Chen et al., IEEE Transactions on Magnetics 38, 4 (2002).

#     Args:
#         a (float): Length of an edge (arb. unit).
#         b (float): Length of an edge (arb. unit).
#         c (float): Length of an edge (arb. unit).
    
#     Returns:
#         (float): Demagnetizating factor.
#     """
#     def Fm(u: float, v: float, w: float) -> float:
#         return u**2*v * np.log((u**2+w**2)*(u**2+2*v**2+2*v*np.sqrt(u**2+v**2)) / (u**2) / (u**2+2*v**2+w**2+2*v*np.sqrt(u**2+v**2+w**2)))

#     F2: float = a**3 + b**3 - 2*c**3 + (a**2+b**2-2*c**2)*np.sqrt(a**2+b**2+c**2)
#     F3: float = (2*c**2-a**2)*np.sqrt(a**2 + c**2) + (2*c**2-b**2)*np.sqrt(b**2 + c**2) - np.sqrt(b**2 + c**2)**3

#     Nm: float = 2/np.pi * np.arctan(a*b/(c*np.sqrt(a**2+b**2+c**2))) + 1/(3*np.pi*a*b*c)*(F2+F3) + 1/(2*np.pi*a*b*c) * (Fm(a,b,c)+Fm(b,a,c)-Fm(c,a,b)-Fm(c,b,a))
#     return Nm

def demagnetizating_factor_rectangular_prism(a: float, b: float, c: float) -> float:
    """Calculating demagnetizating factor of rectangular prism axbxc.

    Thesis:
        A. Aharoni et al., Journal of Applied Physics 83, 3432 (1998).

    Args:
        a (float): Length of an edge (arb. unit).
        b (float): Length of an edge (arb. unit).
        c (float): Length of an edge (arb. unit).
    
    Returns:
        (float): Demagnetizating factor.
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

