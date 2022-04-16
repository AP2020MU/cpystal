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
from mpl_toolkits.axisartist.axislines import SubplotZero
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

    ax.plot(X, Y, marker="o")
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

    ax.plot(X, Y, marker="o", color="blue")
    ax.set_xlabel(r"H (Oe)")
    if per_formula_unit:
        ax.set_ylabel(r"$M$ $(\mu_B/\mathrm{f.u.})$")
    else:
        ax.set_ylabel(r"$M$ $(\mu_B/\mathrm{ion})$")
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


def _graph_susceptibility_vs_temp(material: Crystal, Temp: List[float], Moment: List[float], Field: float, SI: bool = False, per: Optional[str] = None) -> Tuple[plt.Figure, plt.Subplot]: # データはcgs固定．グラフ描画をSIにするかどうか，1molあたりにするかどうか
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

def graph_susceptibility_vs_temp(material: Crystal, Temp: List[float], Moment: List[float], Field: float) -> Tuple[plt.Figure, plt.Subplot]: # データはcgs固定．グラフ描画をSIにするかどうか，1molあたりにするかどうか
    # 縦軸：磁化率，横軸：温度 のグラフを作成
    # Moment: List[moment] moment: 磁気モーメント [emu]
    # Temp: List[temperature] temperature: 温度 [K]
    # Field: 磁場 [Oe]
    if material.formula_weight is None or material.w is None:
        raise TypeError
    susceptibility_temp: List[List[float]] = [[m * (material.formula_weight / material.w) / Field, t] for m,t in zip(Moment,Temp)] # 磁場固定
    X: List[float] = [t for s,t in susceptibility_temp]
    Y1: List[float] = [s for s,t in susceptibility_temp] # [emu/mol.Oe]
    idx = sorted(range(len(X)),key=lambda x:X[x])
    X = [X[i] for i in idx]
    Y1 = [Y1[i] for i in idx]
    
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["legend.framealpha"] = 0
    fig: plt.Figure = plt.figure(figsize=(8,7))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    
    ax.plot(X, Y1, color="blue", marker="o", markersize=5, label=r"$\chi$")
    ax.set_xlabel(r"Temperature (K)")
    ax.set_ylabel(r"susceptibility $\chi$ $(\mathrm{emu/mol.Oe})$")
    # ax.annotate("", xy = (5, Y1[0]+abs(Y1[0])//7),
    #             size = 15, xytext = (15,Y1[0]+abs(Y1[0])//7),
    #             arrowprops = dict(arrowstyle = "<|-", color = "blue"))

    ax.legend(loc="center left", bbox_to_anchor=(0.65,0.5))
    plt.show()
    return fig, ax


def graph_susceptibility_vs_temp_CurieWeiss(material: Crystal, Temp: List[float], Moment: List[float], Field: float, Tc: float) -> Tuple[plt.Figure, plt.Subplot]: # データはcgs固定．グラフ描画をSIにするかどうか，1molあたりにするかどうか
    # 縦軸：磁化率，横軸：温度 のグラフを作成
    # Moment: List[moment] moment: 磁気モーメント [emu]
    # Temp: List[temperature] temperature: 温度 [K]
    # Field: 磁場 [Oe]
    if material.formula_weight is None or material.w is None:
        raise TypeError
    susceptibility_temp: List[List[float]] = [[m * (material.formula_weight / material.w) / Field, t] for m,t in zip(Moment,Temp)] # 磁場固定
    X: List[float] = [t for s,t in susceptibility_temp]
    Y1: List[float] = [s for s,t in susceptibility_temp] # [emu/mol.Oe]
    Y2 = [1/s for s in Y1] # # [(emu/mol.Oe)^-1]
    idx = sorted(range(len(X)),key=lambda x:X[x])
    X = [X[i] for i in idx]
    Y1 = [Y1[i] for i in idx]
    Y2 = [Y2[i] for i in idx]

    
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["legend.framealpha"] = 0
    fig: plt.Figure = plt.figure(figsize=(8,7))
    ax: plt.Subplot = fig.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    
    ax.plot(X, Y1, color="blue", marker="o", markersize=5, label=r"$\chi$")
    ax.set_xlabel(r"Temperature (K)")
    ax.set_ylabel(r"susceptibility $\chi$ $(\mathrm{emu/mol.Oe})$")
    # ax.annotate("", xy = (5, Y1[0]+abs(Y1[0])//7),
    #             size = 15, xytext = (15,Y1[0]+abs(Y1[0])//7),
    #             arrowprops = dict(arrowstyle = "<|-", color = "blue"))
    ax.set_xlim(0,310)

    ax2: plt.Subplot = ax.twinx()
    ax2.plot(X, Y2, color="black", marker="o", markersize=5, label=r"$\chi^{-1}$")
    ax2.set_ylabel(r"$\chi^{-1}$ $((\mathrm{emu/mol.Oe})^{-1})$")
    # ax2.annotate("", xy = (295, Y2[-1]+abs(Y2[-1])//7),
    #             size = 15, xytext = (285,Y2[-1]+abs(Y2[-1])//7),
    #             arrowprops = dict(arrowstyle = "-|>", color = "black"))
    ax2.set_xlim(0,310)

    Temp_high: np.ndarray = np.array([t for t in Temp if t>Tc])
    sus_inv: np.ndarray = Field / (np.array([m for t,m in zip(Temp,Moment) if t>Tc]) / material.mol)
    n: int = len(Temp_high)
    Temp_high_sum: float = np.sum(Temp_high)
    susinv_sum: float = np.sum(sus_inv)
    a: float = ((Temp_high@sus_inv - Temp_high_sum*susinv_sum/n) / (np.sum(Temp_high ** 2) - Temp_high_sum**2/n))
    b: float = (susinv_sum - a * Temp_high_sum)/n
    theta_Curie_Weiss: float = -b/a
    Curie_constant: float = 1/a
    ax.set_title(fr"$\Theta_{{CW}}={theta_Curie_Weiss:.3g}$ K, $C={Curie_constant:.3g}$ emu.K/mol.Oe")
    ax2.plot(Temp_high, a*Temp_high+b, label="Curie Weiss fit", color="red")
    ax.legend(loc="center left", bbox_to_anchor=(0.65,0.5))
    ax2.legend(loc="center left", bbox_to_anchor=(0.65,0.42))
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


def graph_furnace_temperature_profile(sequence: List[List[float]]) -> Tuple[plt.Figure, plt.Subplot]:
    """Graph the furnace temperature profile from the sequence.

    Args:
        sequence (List[List[float]]):
            List of [time_length (hour): float, target_temperature (Celsius degree): float].
            `sequence[0]` should be [0, {room_temperature}].

    Returns:
        (Tuple[plt.Figure, plt.Subplot]): `plt.Figure` object and `plt.axes._subplots.AxesZeroSubplot` object.
    """
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["legend.framealpha"] = 0
    fig: plt.Figure = plt.figure(figsize=(7,5))
    ax: plt.axes._subplots.AxesZeroSubplot = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    for direction in ["right", "top"]:
        ax.axis[direction].set_visible(False)    
    for direction in ["left",  "bottom"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].line.set_facecolor("black")

    Time: List[float] = [ti for ti,te in sequence]
    Temp: List[float] = [te for ti,te in sequence]
    Time_acc: List[float] = [0]
    Time_acc_log: List[float] = [0]
    for i in range(1,len(Time)):
        Time_acc.append(Time_acc[i-1]+Time[i])
        Time_acc_log.append(Time_acc_log[i-1]+np.log(1+Time[i]))
    t_end: float = max(Time_acc_log)
    room_temp: float = min(Temp)
    Temp_high: List[float] = [t for t in Temp if t > room_temp]
    min_temp: float
    if len(set(Temp_high)) != 1:
        min_temp = max(Temp_high) - (max(Temp_high)-min(Temp_high))*2.5
    else:
        min_temp = room_temp
    Temp = [t if t != room_temp else min_temp for t in Temp] # 見やすくするため，室温をmin_tempに変更
    ax.plot(Time_acc_log, Temp, color="black")
    ax.set_xlim(0, t_end*1.1)
    ax.set_ylim(min_temp, min_temp+(max(Temp)-min_temp)*1.1)
    for i, temp in enumerate(Temp):
        ax.plot([0,Time_acc_log[i]], [temp,temp], color='black', linewidth=0.8, linestyle=':')
        ax.plot([Time_acc_log[i],Time_acc_log[i]], [min_temp,temp], color='black', linewidth=0.8, linestyle=':')
    ax.xaxis.set_ticks(Time_acc_log)
    ax.yaxis.set_ticks([min_temp]+[i for i in Temp if i != min_temp])
    ax.xaxis.set_ticklabels(map(str, Time_acc))
    ax.yaxis.set_ticklabels(["R.T."]+list(map(str, [i for i in Temp if i != min_temp])))
    ax.set_xlabel("Time (hour)")
    ax.set_ylabel(u"Temperature (\u00B0C)")
    plt.show()
    return fig, ax


def graph_2zone_temperature_profile(sequence: List[List[float]]) -> Tuple[plt.Figure, plt.Subplot]:
    """Graph the 2-zones furnace temperature profile from the sequence.

    Args:
        sequence (List[List[float]]): 
            List of [time_length (hour): float, 
                    target_temperature_material (Celsius degree): float, 
                    target_temperature_growth (Celsius degree): float].
            `sequence[0]` should be [0, {room_temperature}, {room_temperature}].

    Returns:
        (Tuple[plt.Figure, plt.Subplot]): `plt.Figure` object and `plt.axes._subplots.AxesZeroSubplot` object.
    """
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["legend.framealpha"] = 0
    fig: plt.Figure = plt.figure(figsize=(7,5))
    ax: plt.axes._subplots.AxesZeroSubplot = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    for direction in ["right", "top"]:
        ax.axis[direction].set_visible(False)    
    for direction in ["left",  "bottom"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].line.set_facecolor("black")

    Time: List[float] = [ti for ti, te1, te2 in sequence]
    Temp_material: List[float] = [te1 for ti, te1, te2 in sequence]
    Temp_growth: List[float] = [te2 for ti, te1, te2 in sequence]
    Time_acc: List[float] = [0]
    Time_acc_log: List[float] = [0]
    for i in range(1,len(Time)):
        Time_acc.append(Time_acc[i-1]+Time[i])
        Time_acc_log.append(Time_acc_log[i-1]+np.log(1+Time[i]))
    t_end: float = max(Time_acc_log)
    room_temp: float = min(Temp_material)
    Temp_high: List[float] = [t for t in Temp_material+Temp_growth if t > room_temp]
    min_temp: float
    if len(set(Temp_high)) != 1:
        min_temp = max(Temp_high) - (max(Temp_high)-min(Temp_high))*2.5
    else:
        min_temp = room_temp
    Temp_material = [t if t != room_temp else min_temp for t in Temp_material] # 見やすくするため，室温をmin_tempに変更
    Temp_growth = [t if t != room_temp else min_temp for t in Temp_growth] # 見やすくするため，室温をmin_tempに変更
    max_temp: float = max(max(Temp_material),max(Temp_growth))
    ax.plot(Time_acc_log, Temp_material, color="red", label="Materials side")
    ax.plot(Time_acc_log, Temp_growth, color="blue", label="Growth side")
    ax.set_xlim(0, t_end*1.1)
    ax.set_ylim(min_temp, min_temp+(max_temp-min_temp)*1.1)
    for i, temp in enumerate(Temp_material):
        ax.plot([0,Time_acc_log[i]], [temp,temp], color='black', linewidth=0.8, linestyle=':')
        ax.plot([Time_acc_log[i],Time_acc_log[i]], [min_temp,temp], color='black', linewidth=0.8, linestyle=':')
    for i, temp in enumerate(Temp_growth):
        ax.plot([0,Time_acc_log[i]], [temp,temp], color='black', linewidth=0.8, linestyle=':')
        ax.plot([Time_acc_log[i],Time_acc_log[i]], [min_temp,temp], color='black', linewidth=0.8, linestyle=':')
    ax.xaxis.set_ticks(Time_acc_log)
    ax.yaxis.set_ticks([min_temp]+[i for i in Temp_material+Temp_growth if i != min_temp])
    ax.xaxis.set_ticklabels(map(str, Time_acc))
    ax.yaxis.set_ticklabels(["R.T."]+list(map(str, [i for i in Temp_material+Temp_growth if i != min_temp])))
    ax.set_xlabel("Time (hour)")
    ax.set_ylabel(u"Temperature (\u00B0C)")
    ax.legend(loc="lower center")
    plt.show()
    return fig, ax

def main() -> None:
    pass
    return

if __name__ == "__main__":
    main()

