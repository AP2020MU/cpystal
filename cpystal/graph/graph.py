"""`cpystal.graph` is a module for making graphs.

Functions:
    `graph_moment_vs_temp`
    `graph_moment_vs_field`
    `graph_magnetization_vs_temp`
    `graph_magnetization_vs_field`
    `graph_Bohr_vs_field`
    `graph_Bohr_vs_temp`
    `graph_susceptibility_vs_temp`
    `graph_powder_Xray_intensity_vs_angle`
    `ax_transplant`
"""
from __future__ import annotations # class定義中に自己classを型ヒントとして使用するため

from collections import deque
from math import pi
from typing import Deque, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt # type: ignore
import numpy as np

from ..core import Crystal

def graph_moment_vs_temp(material: Crystal, Temp: List[float], Moment: List[float], field_val: float, SI: bool = False, per: Optional[str] = None) -> Tuple[plt.Figure, plt.Subplot]: # 磁場固定
    # 縦軸：磁気モーメント，横軸：温度 のグラフを作成
    # Moment: 磁気モーメント [emu]
    # Temp: 温度 [K]
    # field_val: 磁場 [Oe]
    moment_vs_temp: List[List[float]] = [[m,t] for m,t in zip(Moment,Temp)]
    X: List[float] = [t for m,t in moment_vs_temp]
    Y: List[float] = [m for m,t in moment_vs_temp]
    if SI:
        if per == "mol":
            if material.mol is None:
                raise TypeError
            Y = [m * 10**(-3) / material.mol for m in Y] # [Am^2/mol]
        elif per == "weight":
            if material.w is None:
                raise TypeError
            Y = [m / material.w for m in Y] # [Am^2/kg]
        else:
            Y = [m * 10**(-3) for m in Y] # [Am^2]
    else:
        if per == "mol":
            if material.mol is None:
                raise TypeError
            Y = [m / material.mol for m in Y] # [emu/mol]
        elif per == "weight":
            if material.w is None:
                raise TypeError
            Y = [m / material.w for m in Y] # [emu/g]
        else:
            Y = [m for m in Y] # [emu]
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["legend.framealpha"] = 0
    fig: plt.Figure = plt.figure(figsize=(8,7))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.plot(X, Y)
    ax.set_xlabel(r"Temperature (K)")
    if SI:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $(\mathrm{A\, m^{2}\, mol^{-1}})$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $(\mathrm{A\, m^{2}\, kg^{-1}})$")
        else:
            ax.set_ylabel(r"Magnetization $(\mathrm{A\, m^{2}})$")
    else:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $(\mathrm{emu\, mol^{-1}})$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $(\mathrm{emu\, g^{-1}})$")
        else:
            ax.set_ylabel(r"Magnetization $(\mathrm{emu})$")
    ax.set_title(f"{material.graphname} Magnetic Moment vs Temperature at {field_val} Oe")
    plt.show()
    return fig, ax


def graph_moment_vs_field(material: Crystal, Field: List[float], Moment: List[float], temp_val: float, SI: bool = False, per: Optional[str] = None) -> Tuple[plt.Figure, plt.Subplot]: # 温度固定
    # 縦軸：磁気モーメント，横軸：磁場 のグラフを作成
    # Moment: 磁気モーメント [emu]
    # Field: 磁場 [Oe]
    # temp_val: 温度 [K]
    moment_vs_field: List[List[float]] = [[m,f] for m,f in zip(Moment,Field)]
    X: List[float] = [f for m,f in moment_vs_field]
    Y: List[float] = [m for m,f in moment_vs_field]
    if SI:
        if per == "mol":
            if material.mol is None:
                raise TypeError
            Y = [m * 10**(-3) / material.mol for m in Y] # [Am^2/mol]
        elif per == "weight":
            if material.w is None:
                raise TypeError
            Y = [m / material.w for m in Y] # [Am^2/kg]
        else:
            Y = [m * 10**(-3) for m in Y] # [Am^2]
    else:
        if per == "mol":
            if material.mol is None:
                raise TypeError
            Y = [m / material.mol for m in Y] # [emu/mol]
        elif per == "weight":
            if material.w is None:
                raise TypeError
            Y = [m / material.w for m in Y] # [emu/g]
        else:
            Y = [m for m in Y] # [emu]
    
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["legend.framealpha"] = 0
    fig: plt.Figure = plt.figure(figsize=(8,7))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.plot(X, Y)
    ax.set_xlabel(r"Magnetic Field (Oe)")
    if SI:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $(\mathrm{A\, m^{2}\, mol^{-1}})$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $(\mathrm{A\, m^{2}\, kg^{-1}})$")
        else:
            ax.set_ylabel(r"Magnetization $(\mathrm{A\, m^{2}})$")
    else:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $(\mathrm{emu\, mol^{-1}})$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $(\mathrm{emu\, g^{-1}})$")
        else:
            ax.set_ylabel(r"Magnetization $(\mathrm{emu})$")
    ax.set_title(f"{material.graphname} Magnetic Moment vs Magnetic Field at {temp_val} K")
    plt.show()
    return fig, ax


def graph_magnetization_vs_temp(material: Crystal, Temp: List[float], Moment: List[float], field_val: float, SI: bool = False, per: Optional[str] = None) -> Tuple[plt.Figure, plt.Subplot]:
    # データはcgs固定．
    # SI: グラフ描画をSIにするかどうか
    # per: molあたり，重さあたりにするかどうか
    # 縦軸：磁化，横軸：温度 のグラフを作成
    magnetization_vs_temp: List[List[float]] = [[material.cal_magnetization(m=m,SI=SI,per=per),t] for m,t in zip(Moment,Temp)] # 磁場固定
    X: List[float] = [t for m,t in magnetization_vs_temp]
    Y: List[float] = [m for m,t in magnetization_vs_temp]
    
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["legend.framealpha"] = 0
    fig: plt.Figure = plt.figure(figsize=(8,7))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.plot(X, Y)
    ax.set_xlabel(r"Temperature (K)")
    if SI:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $(\mathrm{A\, m^{-1}\, mol^{-1}})$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $(\mathrm{A\, m^{-1}\, kg^{-1}})$")
        else:
            ax.set_ylabel(r"Magnetization $(\mathrm{A\, m^{-1}})$")
    else:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $(\mathrm{G\, mol^{-1}})$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $(\mathrm{G\, g^{-1}})$")
        else:
            ax.set_ylabel(r"Magnetization $(\mathrm{G})$")
    ax.set_title(f"{material.graphname} Magnetization vs Temperature at {field_val} Oe")
    plt.show()
    return fig, ax


def graph_magnetization_vs_field(material: Crystal, Field: List[float], Moment: List[float], temp_val: float, SI: bool = False, per: Optional[str] = None) -> Tuple[plt.Figure, plt.Subplot]: # データはcgs固定．グラフ描画をSIにするかどうか，1molあたりにするかどうか
    # 縦軸：磁化，横軸：磁場 のグラフを作成
    magnetization_vs_field: List[List[float]] = [[material.cal_magnetization(m=m,SI=SI,per=per),f] for m,f in zip(Moment,Field)] # 温度固定
    X: List[float] = [f for m,f in magnetization_vs_field]
    Y: List[float] = [m for m,f in magnetization_vs_field]
    
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["legend.framealpha"] = 0
    fig: plt.Figure = plt.figure(figsize=(8,7))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.plot(X, Y)
    ax.set_xlabel(r"Magnetic Field (Oe)")
    if SI:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $(\mathrm{A\, m^{-1}\, mol^{-1}})$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $(\mathrm{A\, m^{-1}\, kg^{-1}})$")
        else:
            ax.set_ylabel(r"Magnetization $(\mathrm{A\, m^{-1}})$")
    else:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $(\mathrm{G\, mol^{-1}})$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $(\mathrm{G\, g^{-1}})$")
        else:
            ax.set_ylabel(r"Magnetization $(\mathrm{G})$")
    ax.set_title(f"{material.graphname} Magnetization vs Magnetic Field at {temp_val} K")
    plt.show()
    return fig, ax


def graph_Bohr_vs_field(material: Crystal, Field: List[float], Moment: List[float], temp_val: float, per_formula_unit: bool = True) -> Tuple[plt.Figure, plt.Subplot]:
    Bohr_vs_field: List[List[float]]
    if per_formula_unit:
        # 縦軸：有効ボーア磁子数/式単位，横軸：磁場 のグラフを作成
        Bohr_vs_field = [[material.cal_Bohr_per_formula_unit(m=m),f] for m,f in zip(Moment,Field)] # 温度固定
    else:
        # 縦軸：有効ボーア磁子数/磁性イオン，横軸：磁場 のグラフを作成
        Bohr_vs_field = [[material.cal_Bohr_per_ion(m=m),f] for m,f in zip(Moment,Field)] # 温度固定
    X: List[float] = [f for b,f in Bohr_vs_field]
    Y: List[float] = [b for b,f in Bohr_vs_field]
    
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["legend.framealpha"] = 0
    fig: plt.Figure = plt.figure(figsize=(8,7))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.plot(X, Y)
    ax.set_xlabel(r"Magnetic Field (Oe)")
    if per_formula_unit:
        ax.set_ylabel(r"Magnetic Moment $(\mu_B/\mathrm{f.u.})$")
    else:
        ax.set_ylabel(r"Magnetic Moment $(\mu_B/\mathrm{ion})$")
    if material.date is not None:
        ax.set_title(f"{material.graphname}({material.date})\n Magnetic Moment vs Magnetic Field at {temp_val} K")
    else:
        ax.set_title(f"{material.graphname}\n Magnetic Moment vs Magnetic Field at {temp_val} K")
    plt.show()
    return fig, ax


def graph_Bohr_vs_temp(material: Crystal, Temp: List[float], Moment: List[float], field_val: float, per_formula_unit: bool = True) -> Tuple[plt.Figure, plt.Subplot]:
    Bohr_vs_temp: List[List[float]]
    if per_formula_unit:
        # 縦軸：有効ボーア磁子数/式単位，横軸：磁場 のグラフを作成
        Bohr_vs_temp = [[material.cal_Bohr_per_formula_unit(m=m),t] for m,t in zip(Moment,Temp)] # 温度固定
    else:
        # 縦軸：有効ボーア磁子数/磁性イオン，横軸：磁場 のグラフを作成
        Bohr_vs_temp = [[material.cal_Bohr_per_ion(m=m),t] for m,t in zip(Moment,Temp)] # 温度固定
    X: List[float] = [t for b,t in Bohr_vs_temp]
    Y: List[float] = [b for b,t in Bohr_vs_temp]
    
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["legend.framealpha"] = 0
    fig: plt.Figure = plt.figure(figsize=(8,7))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.plot(X, Y)
    ax.set_xlabel(r"Temperature (K)")
    if per_formula_unit:
        ax.set_ylabel(r"Magnetic Moment $(\mu_B/\mathrm{f.u.})$")
    else:
        ax.set_ylabel(r"Magnetic Moment $(\mu_B/\mathrm{ion})$")
    if material.date is not None:
        ax.set_title(f"{material.graphname}({material.date})\n Magnetic Moment vs Temperature at {field_val} Oe")
    else:
        ax.set_title(f"{material.graphname}\n Magnetic Moment vs Temperature at {field_val} Oe")
    plt.show()
    return fig, ax


def graph_susceptibility_vs_temp(material: Crystal, Temp: List[float], Moment: List[float], Field: float, SI: bool = False, per: Optional[str] = None) -> Tuple[plt.Figure, plt.Subplot]: # データはcgs固定．グラフ描画をSIにするかどうか，1molあたりにするかどうか
    # 縦軸：磁化率，横軸：温度 のグラフを作成
    # Moment: List[moment] moment: 磁気モーメント [emu]
    # Temp: List[temperature] temperature: 温度 [K]
    # Field: 磁場 [Oe]
    susceptibility_temp: List[List[float]] = [[material.cal_magnetization(m=m,SI=SI,per=per)/Field,t] for m,t in zip(Moment,Temp)] # 磁場固定
    X: List[float] = [t for s,t in susceptibility_temp]
    Y: List[float] = [s for s,t in susceptibility_temp]
    if SI:
        if per == "mol":
            if material.formula_weight is None or material.w is None:
                raise TypeError
            Y = [m * (material.formula_weight / material.w) * 4*pi*10**(-6) for m in Y] # [m^3/mol]
        elif per == "weight":
            material.cal_density()
            if material.density is None:
                raise TypeError
            Y = [m / material.density * 4*pi*10**(-3) for m in Y] # [m^3/kg]
        else:
            Y = [m * 4*pi for m in Y] # (無次元)
    else:
        if per == "mol":
            if material.formula_weight is None or material.w is None:
                raise TypeError
            Y = [m * (material.formula_weight / material.w) for m in Y] # [cm^3/mol]
        elif per == "weight":
            material.cal_density()
            if material.density is None:
                raise TypeError
            Y = [m / material.density for m in Y] # [cm^3/g]
        else:
            Y = [m for m in Y] # (無次元)
    
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["legend.framealpha"] = 0
    fig: plt.Figure = plt.figure(figsize=(8,7))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.plot(X, Y)
    ax.set_xlabel(r"Temperature (K)")
    if SI:
        if per == "mol":
            ax.set_ylabel(r"Susceptibility $(\mathrm{m^3\, mol^{-1}})$")
        elif per == "weight":
            ax.set_ylabel(r"Susceptibility $(\mathrm{m^3\, kg^{-1}})$")
        else:
            ax.set_ylabel(r"Susceptibility (dimensionless)")
    else:
        if per == "mol":
            ax.set_ylabel(r"Susceptibility $(\mathrm{cm^3\, mol^{-1}})$")
        elif per == "weight":
            ax.set_ylabel(r"Susceptibility $(\mathrm{cm^3\, g^{-1}})$")
        else:
            ax.set_ylabel(r"Susceptibility (dimensionless)")
    ax.set_title(f"{material.graphname} Susceptibility vs Temperature")
    plt.show()
    return fig, ax
    

def graph_powder_Xray_intensity_vs_angle(filename: str, display_num: int = 10, material: Optional[Crystal] = None) -> Tuple[plt.Figure, plt.Subplot]:
    with open(filename, encoding="shift_jis") as f:
        data: List[List[float]] = [list(map(float, s.strip().split())) for s in f.readlines()[3:]]
        N: int = len(data)
        two_theta: List[float] = [d[0] for d in data] # データは2θ
        theta: List[float] = [d[0]/2 for d in data] # データは2θ
        intensity: List[float] = [d[1] for d in data]
        neg: List[float] = [i for i in intensity if i<=0]
        assert len(neg)==0 # 負のintensityをもつ壊れたデータがないことを確認
        
        neighbor_num: int = 20 # peak(極大値の中でも急激に増加するもの)判定で参照する近傍のデータ点数
        
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
        for i, (_, p, theta_p, intensity_p) in enumerate(peak):
            if i == display_num:
                break
            d_hkl_over_n_alpha: float = Cu_K_alpha/np.sin(np.radians(theta_p))/2
            d_hkl_over_n_beta: float = Cu_K_beta/np.sin(np.radians(theta_p))/2
            print(f"θ = {theta_p}, 2θ = {2*theta_p}, intensity = {intensity_p}")
            print(f"    Kα: d_hkl/n = {d_hkl_over_n_alpha}")
            print(f"    Kβ: d_hkl/n = {d_hkl_over_n_beta}")
        
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["legend.framealpha"] = 0
    fig: plt.Figure = plt.figure(figsize=(8,7))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.plot(two_theta, intensity)
    ax.set_yscale('log')
    ax.set_xlabel(r"$2\theta\, ({}^{\circ})$")
    ax.set_ylabel(r"intensity (count)")
    if material is not None:
        ax.set_title(f"{material.graphname} powder X-ray diffraction")
    else:
        ax.set_title(f"powder X-ray diffraction")
    plt.show()
    return fig, ax


def ax_transplant(ax: plt.Subplot, fig_new: Optional[plt.Figure] = None, figsize: Optional[Tuple[float, float]] = None, ax_new: Optional[plt.Subplot] = None) -> Tuple[plt.Figure, plt.Subplot]:
    # 現状は最低限のpropertyしかないので必要な項目が増えたら追加する
    if fig_new is None:
        if figsize is None:
            fig_new = plt.figure()
        else:
            fig_new = plt.figure(figsize=figsize)
        ax_new = fig_new.add_subplot(111)
    else:
        if ax_new is None:
            ax_new = fig_new.add_subplot(111)

    ax_new.set_title(ax.title.get_text(), fontsize=ax.title.get_fontsize())
    ax_new.set_xlabel(ax.xaxis.label.get_text())
    ax_new.set_ylabel(ax.yaxis.label.get_text())
    ax_new.set_xlim(ax.get_xlim())
    ax_new.set_ylim(ax.get_ylim())

    # plot
    for line2d in ax.lines:
        ax_new.plot(line2d._xorig, line2d._yorig, label=line2d._label)

    # scatter
    for pathcollection in ax.collections:
        xy = list(pathcollection._offsets)
        x: List[float] = [i for i,j in xy]
        y: List[float] = [j for i,j in xy]
        ax_new.scatter(x, y, label=pathcollection._label)

    # text
    for t in ax.texts:
        ax_new.text(t._x, t._y, t._text)

    # legend
    dict_loc_real: Dict[int, str] =  {1:"upper right", 2:"upper left", 3:"lower left", 4:"lower right"}
    if ax._axes.legend_ is not None:
        if not ax._axes.legend_._loc_used_default:
            if ax._axes.legend_._bbox_to_anchor is not None:
                bbox_to_anchor = ax._axes.legend_._bbox_to_anchor._bbox._points[0]
            else:
                bbox_to_anchor = None
            ax_new.legend(bbox_to_anchor=bbox_to_anchor, 
                            loc=dict_loc_real[ax._axes.legend_._loc_real], 
                            borderaxespad=ax._axes.legend_.borderaxespad, 
                            fontsize=ax._axes.legend_._fontsize)
        else:
            ax_new.legend()
    return fig_new, ax_new


def main() -> None:
    pass
    return

if __name__ == "__main__":
    main()

