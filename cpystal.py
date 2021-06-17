from __future__ import annotations # class定義中に自己classを型ヒントとして使用するため

from collections import defaultdict, deque
from math import pi, sqrt, cos, sin, radians
import re
from typing import Any, DefaultDict, Deque, Dict, Iterable, List, Optional, overload, Set, Tuple, Union

import matplotlib.pyplot as plt # type: ignore
import numpy as np
import pickle
import scipy.signal # type: ignore


# データ処理用
def str_to_float(s: str) -> Optional[float]:
    if s == '':
        return None
    else:
        return float(s)

# データ平滑化
def smoothing(data: List[float], mode: Any = None) -> List[float]:
    if mode == "s": # Savitzky-Golay filter
        deg: int = 2 # 多項式の次数
        window_length: int = len(data)//4*2+1 # 窓幅
        deriv: int = 0 # 微分階数
        return list(map(float, scipy.signal.savgol_filter(data, window_length=window_length, polyorder=deg, deriv=deriv)))
      
    else: # 単純移動平均
        window: int = 5 # 移動平均の範囲
        w: Any = np.ones(window)/window
        return list(map(float, np.convolve(data, w, mode='same')))

# 日本化学会原子量表(2021)より
# 原子量の定義：質量数12の炭素原子 12C の質量を正確に12としたときの相対質量
# 注意： ・原子量定義の諸々は元文献が詳しいので，慎重な取り扱いが必要な場面ではそちらを参照すること
#       ・安定各種が複数ある原子の原子量はその同位体存在比によって時間・空間変化する
#       ・以下のdictの13元素でコメントアウトされている区間[a,b]は地球上での変動範囲を表す
#           ・その13元素の原子量はすべて有効数字4桁の数値を用いた
#       ・(安定同位体が存在しない元素は基本的に None とし，コメントアウトで放射性同位体の質量数の一例を示した)
#           ・Dict[str, Optional[float]] にするとmypyのチェック回避が面倒なので削除した
#           ・ただし，安定同位体が存在しない元素のうちBi,Th,Pa,Uは地球上で固有の同位体比を示すため，原子量が定義されている
#       ・環境中の変動とは別に測定誤差も考慮する必要があるが，以下では記載していない
atomic_weight: Dict[str, float] = {
    "H": 1.008, # [1.00784, 1.00811]
    "He": 4.002602,
    "Li": 6.941, # [6.938, 6.997]
    "Be": 9.0121831,
    "B": 10.81, # [10.806, 10.821]
    "C": 12.01, # [12.0096, 12.0116]
    "N": 14.01, # [14.00643, 14.00728]
    "O": 16.00, # [15.99903, 15.99977]
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.31, # [24.304, 24.307]
    "Al": 26.9815384,
    "Si": 28.09, # [28.084, 28.086]
    "P": 30.973761998,
    "S": 32.07, # [32.059, 32.076]
    "Cl": 35.45, # [35.446, 35.457]
    "Ar": 39.95, # [39.792, 39.963]
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955908,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938043,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.921595,
    "Se": 78.971,
    "Br": 79.90, # [79.901, 79.907]
    "Kr": 83.798,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.90584,
    "Zr": 91.224,
    "Nb": 92.90637,
    "Mo": 95.95,
    #"Tc": None, # 99
    "Ru": 101.07,
    "Rh": 102.90549,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Cd": 112.414,
    "In": 114.818,
    "Sn": 118.710,
    "Sb": 121.760,
    "Te": 127.60,
    "I": 126.90447,
    "Xe": 131.293,
    "Cs": 132.90545196,
    "Ba": 137.327,
    "La": 138.90547,
    "Ce": 140.116,
    "Pr": 140.90766,
    "Nd": 144.242,
    #"Pm": None, # 145
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.925354,
    "Dy": 162.500,
    "Ho": 164.930328,
    "Er": 167.259,
    "Tm": 168.934218,
    "Yb": 173.045,
    "Lu": 174.9668,
    "Hf": 178.486,
    "Ta": 180.94788,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.966570,
    "Hg": 200.592,
    "Tl": 204.4, # [204.382, 204.385]
    "Pb": 207.2,
    "Bi": 208.98040,
    #"Po": None, # 210
    #"At": None, # 210
    #"Rn": None, # 222
    #"Fr": None, # 223
    #"Ra": None, # 226
    #"Ac": None, # 227
    "Th": 232.0377,
    "Pa": 231.03588,
    "U": 238.02891,
    #"Np": None, # 237
    #"Pu": None, # 239
    #"Am": None, # 243
    #"Cm": None, # 247
    #"Bk": None, # 247
    #"Cf": None, # 252
    #"Es": None, # 252
    #"Fm": None, # 257
    #"Md": None, # 258
    #"No": None, # 259
    #"Lr": None, # 262
    #"Rf": None, # 267
    #"Db": None, # 268
    #"Sg": None, # 271
    #"Bh": None, # 272
    #"Hs": None, # 277
    #"Mt": None, # 276
    #"Ds": None, # 281
    #"Rg": None, # 280
    #"Cn": None, # 285
    #"Nh": None, # 278
    #"Fl": None, # 289
    #"Mc": None, # 289
    #"Lv": None, # 293
    #"Ts": None, # 293
    #"Og": None, # 294
}


class Crystal: # 結晶の各物理量を計算
    def __init__(self, name: str, date: Optional[str] = None, auto_formula_weight: bool = True):
        self.name: str = name # 化合物名
        self.graphname: str = "$\mathrm{" + re.sub('([0-9]+)', '_{\\1}', name) + "}$" # グラフで表示する名前
        self.date: Optional[str] = date # 合成した日付(必要ならナンバリングもここに含める)
        
        self.NA: float = 6.02214076 * 10**(23) # アボガドロ定数:[/mol]
        
        # 格子定数
        self.a: Optional[float] = None      # 格子定数 [Å]
        self.b: Optional[float] = None      # 格子定数 [Å]
        self.c: Optional[float] = None      # 格子定数 [Å]
        self.alpha: Optional[float] = None  # 基本並進ベクトル間の角度 [°]
        self.beta: Optional[float] = None   # 基本並進ベクトル間の角度 [°]
        self.gamma: Optional[float] = None  # 基本並進ベクトル間の角度 [°]
        self.V: Optional[float] = None      # 単位胞の体積 [cm^3]
        self.num: Optional[int] = None      # 単位胞に含まれる化学式の数 (無次元)

        self.formula_weight: Optional[float] = None # モル質量(式量) [g/mol]
        self.w: Optional[float] = None              # 試料の質量 [g]
        self.num_magnetic_ion: Optional[int] = None # 化学式中の磁性イオンの数 (無次元)
        self.density: Optional[float] = None        # 密度 [g/cm^3]
        self.mol: Optional[float] = None            # 物質量 [mol]

        self.numbered_name: str = re.sub(r"([A-Z][a-z]?|\))(?=[^0-9a-z]+)", r"\g<1>1", name+"$")[:-1] # 元素数を明示したname ("$"は番兵)
        self.components: DefaultDict[str, int] = defaultdict(int)   # 化学式中に各元素がいくつあるか

        # 各クラス変数の単位
        # 内部では基本的にCGS単位系を用いる
        self.unit: Dict[str, str] = {
            "unit": "",
            "NA": "mol^-1", "name": "", "graphname": "", "date": "",
            "a": "Å", "b": "Å", "c": "Å", "alpha": "°", "beta": "°", "gamma": "°",
            "V": "cm^3", "num": "", "formula_weight": "g/mol", "w": "g", 
            "num_magnetic_ion": "", "density": "g/cm^3", "mol": "mol",
            "numbered_name": "", "components": ""
        }

        # 化学式を"形態素"ごとに分割したリスト
        divided_name: List[str] = re.split(r",+", re.sub(r"([A-Z][a-z]*|\d+|[()])", ",\\1,", self.numbered_name).strip(","))
        now: int = 1 # 倍率
        num_stack: List[int] = [1] # 後ろから見て，現在有効な数の積を格納するstack
        for s in reversed(divided_name): # 化学式を後ろからみる
            if s.isdigit(): # 数値
                now *= int(s)
                num_stack.append(int(s))
            elif s == ")":
                pass
            elif s == "(": # ()を付けるときは必ず直後に1以上の数字が来る
                now //= num_stack.pop()
            else:
                self.components[s] += now
                now //= num_stack.pop()
        if auto_formula_weight: # nameから自動で式量を計算
            formula_weight: float = 0.0 # 式量
            for element, n in self.components.items():
                if not element in atomic_weight:
                    raise KeyError
                formula_weight += atomic_weight[element] * n
            self.formula_weight = formula_weight


    def __str__(self) -> str:
        res: str = "\n"
        for k, v in self.__dict__.items():
            if v is None or k == "unit":
                continue
            if type(v) is float:
                res = res + f"{k} = {v:.5g} {self.unit[k]}\n"
            else:
                res = res + f"{k} = {v} {self.unit[k]}\n"
        return res

    def __add__(self, other: Crystal) -> Crystal:
        if type(other) is not Crystal:
            raise TypeError
        return Crystal(self.name + other.name)

    def __mul__(self, other: int) -> Crystal:
        if type(other) is not int:
            raise TypeError
        # self.numbered_name中の数字をすべてother倍する
        return Crystal(re.sub(r"[0-9]+", lambda x: str(other*int(x.group())), self.numbered_name))

    def set_lattice_constant(self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float, num: Optional[int] = None):
        # a,b,c: 格子定数 [Å]
        # alpha,beta,gamma: 基本並進ベクトル間の角度 [°]
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        ca = cos(radians(alpha))
        cb = cos(radians(beta))
        cc = cos(radians(gamma))
        # 単位胞の体積 [cm^3]
        self.V = a*b*c * sqrt(1 + 2*ca*cb*cc - ca**2 - cb**2 - cc**2) * 10**(-24)
        if num is not None:
            self.num = num # 単位胞に含まれる化学式の数 (無次元)
    
    def set_formula_weight(self, formula_weight: float):
        # コンストラクタでauto_formula_weight = Trueで自動設定可能
        # formula_weight: モル質量(式量) [g/mol]
        self.formula_weight = formula_weight

    def set_weight(self, w: float):
        # w: 試料の質量 [g]
        self.w = w
    
    def set_mol(self, mol: float):
        # mol: 試料の物質量 [mol]
        self.mol = mol

    def set_num_magnetic_ion(self, num_magnetic_ion: int):
        # num_magnetic_ion: 化学式中の磁性イオンの数 (無次元)
        self.num_magnetic_ion = num_magnetic_ion

    def cal_density(self) -> float:
        # formula_weight: モル質量(式量) [g/mol]
        # num: 単位胞の分子数 (無次元)
        # V: 単位胞の体積 [Å^3]
        # density: 密度 [g/cm^3]
        if self.formula_weight is None or self.num is None or self.V is None:
            raise TypeError
        self.density = self.formula_weight * self.num / self.NA / self.V
        return self.density

    def cal_mol(self) -> float:
        # formula_weight: モル質量(式量) [g/mol]
        # w: 試料の質量 [g]
        # mol: 試料の物質量 [mol]
        if self.formula_weight is None or self.w is None:
            raise TypeError
        self.mol = self.w / self.formula_weight
        return self.mol

    def cal_weight(self) -> float:
        # formula_weight: モル質量(式量) [g/mol]
        # mol: 試料の物質量 [mol]
        # w: 試料の質量 [g]
        if self.formula_weight is None or self.mol is None:
            raise TypeError
        self.w = self.formula_weight * self.mol
        return self.w

    def cal_magnetization(self, m: float, w: Optional[float] = None, SI: bool = False, per: Optional[str] = None) -> float:
        # m: 磁気モーメント [emu]
        # w: 試料の質量 [g]
        # density: 密度 [g/cm^3]
        # M: 磁化 [G] or [G/mol] or [A/m] or [A/(m mol)]
        if w is None:
            w = self.w
        if w is None or self.density is None or self.formula_weight is None:
            raise TypeError
        M: float = m * self.density / w # [G] = [emu/cm^3]
        if SI:
            M *= 10**3 # 1 G == 1000 A/m
            if per == "mol": # molあたり
                M *= self.formula_weight / w
            elif per == "weight": # kgあたり
                M *= 1/w * 1000
        else:
            if per == "mol": # molあたり
                M *= self.formula_weight / w
            elif per == "weight": # gあたり
                M *= 1/w
        return M

    def cal_Bohr_per_formula_unit(self, m: float, w: Optional[float] = None) -> float:
        muB: float = 9.274 * 10**(-21) # Bohr磁子 [emu]
        if w is None:
            w = self.w
        if w is None or self.formula_weight is None:
            raise TypeError
        # 式量あたりの有効Bohr磁子数 [μB/f.u.]
        mu: float = (m / muB) * (self.formula_weight / w / self.NA)
        return mu

    def cal_Bohr_per_ion(self, m: float, w: Optional[float] = None, num_magnetic_ion: Optional[int] = None) -> float:
        muB: float = 9.274 * 10**(-21) # Bohr磁子 [emu]
        if w is None:
            w = self.w
        if num_magnetic_ion is None:
            num_magnetic_ion = self.num_magnetic_ion
        if w is None or num_magnetic_ion is None or self.formula_weight is None:
            raise TypeError
        # 磁性イオンあたりの有効Bohr磁子数 [μB/ion]
        mu: float = (m / muB) * (self.formula_weight / w / self.NA / num_magnetic_ion)
        return mu

    def cal_ingredients(self) -> List[Tuple[str, float]]:
        # selfに含まれる各元素の重量をそれぞれ求める
        res: List[Tuple[str, float]] = []
        if self.formula_weight is None:
            raise TypeError
        for element, n in self.components.items():
            res.append((element, n*atomic_weight[element]/self.formula_weight))
        res = res[::-1]
        print(f"The ingredients of {self.name} ({self.w} g):")
        if self.w is None:
            print("\n".join([f"{element} = {ratio:.2%}" for element, ratio in res]))
        else:
            print("\n".join([f"{element} = {ratio*self.w:.4g} g ({ratio:.2%})" for element, ratio in res]))
        return res
    
    def save(self, overwrite: bool = False): # Crystalインスタンスのデータを保存
        filename: str = self.name
        if self.date is not None:
            filename = filename + self.date
        mode: str
        if overwrite:
            mode = 'wb' # 上書きあり
        else:
            mode = 'xb' # 上書きなし
        with open(filename+'.pickle', mode=mode) as f:
            pickle.dump(self, f)

    def load(self, filename: str): # Crystalインスタンスのデータをロード
        with open(filename+'.pickle', mode='rb') as f:
            pre: Crystal = pickle.load(f)
            self.__dict__ = pre.__dict__



def make_moment_vs_temp(material: Crystal, Moment: List[float], Temp: List[float], field_val: float, SI: bool = False, per: Optional[str] = None): # 磁場固定
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
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111)
    ax.plot(X, Y)
    ax.set_xlabel(r"Temperature [K]")
    if SI:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $[\mathrm{A\, m^{2}\, mol^{-1}}]$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $[\mathrm{A\, m^{2}\, kg^{-1}}]$")
        else:
            ax.set_ylabel(r"Magnetization $[\mathrm{A\, m^{2}}]$")
    else:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $[\mathrm{emu\, mol^{-1}}]$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $[\mathrm{emu\, g^{-1}}]$")
        else:
            ax.set_ylabel(r"Magnetization $[\mathrm{emu}]$")
    ax.set_title(f"{material.graphname} Magnetic Moment vs Temperature at {field_val} Oe")
    plt.show()


def make_moment_vs_field(material: Crystal, Moment: List[float], Field: List[float], temp_val: float, SI: bool = False, per: Optional[str] = None): # 温度固定
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
    
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111)
    ax.plot(X, Y)
    ax.set_xlabel(r"Magnetic Field [Oe]")
    if SI:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $[\mathrm{A\, m^{2}\, mol^{-1}}]$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $[\mathrm{A\, m^{2}\, kg^{-1}}]$")
        else:
            ax.set_ylabel(r"Magnetization $[\mathrm{A\, m^{2}}]$")
    else:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $[\mathrm{emu\, mol^{-1}}]$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $[\mathrm{emu\, g^{-1}}]$")
        else:
            ax.set_ylabel(r"Magnetization $[\mathrm{emu}]$")
    ax.set_title(f"{material.graphname} Magnetic Moment vs Magnetic Field at {temp_val} K")
    plt.show()


def make_magnetization_vs_temp(material: Crystal, Moment: List[float], Temp: List[float], field_val: float, SI: bool = False, per: Optional[str] = None):
    # データはcgs固定．
    # SI: グラフ描画をSIにするかどうか
    # per: molあたり，重さあたりにするかどうか
    # 縦軸：磁化，横軸：温度 のグラフを作成
    magnetization_vs_temp: List[List[float]] = [[material.cal_magnetization(m=m,SI=SI,per=per),t] for m,t in zip(Moment,Temp)] # 磁場固定
    X: List[float] = [t for m,t in magnetization_vs_temp]
    Y: List[float] = [m for m,t in magnetization_vs_temp]
    
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111)
    ax.plot(X, Y)
    ax.set_xlabel(r"Temperature [K]")
    if SI:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $[\mathrm{A\, m^{-1}\, mol^{-1}}]$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $[\mathrm{A\, m^{-1}\, kg^{-1}}]$")
        else:
            ax.set_ylabel(r"Magnetization $[\mathrm{A\, m^{-1}}]$")
    else:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $[\mathrm{G\, mol^{-1}}]$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $[\mathrm{G\, g^{-1}}]$")
        else:
            ax.set_ylabel(r"Magnetization $[\mathrm{G}]$")
    ax.set_title(f"{material.graphname} Magnetization vs Temperature at {field_val} Oe")
    plt.show()


def make_magnetization_vs_field(material: Crystal, Moment: List[float], Field: List[float], temp_val: float, SI: bool = False, per: Optional[str] = None): # データはcgs固定．グラフ描画をSIにするかどうか，1molあたりにするかどうか
    # 縦軸：磁化，横軸：磁場 のグラフを作成
    magnetization_vs_field: List[List[float]] = [[material.cal_magnetization(m=m,SI=SI,per=per),f] for m,f in zip(Moment,Field)] # 温度固定
    X: List[float] = [f for m,f in magnetization_vs_field]
    Y: List[float] = [m for m,f in magnetization_vs_field]
    
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111)
    ax.plot(X, Y)
    ax.set_xlabel(r"Magnetic Field [Oe]")
    if SI:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $[\mathrm{A\, m^{-1}\, mol^{-1}}]$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $[\mathrm{A\, m^{-1}\, kg^{-1}}]$")
        else:
            ax.set_ylabel(r"Magnetization $[\mathrm{A\, m^{-1}}]$")
    else:
        if per == "mol":
            ax.set_ylabel(r"Magnetization $[\mathrm{G\, mol^{-1}}]$")
        elif per == "weight":
            ax.set_ylabel(r"Magnetization $[\mathrm{G\, g^{-1}}]$")
        else:
            ax.set_ylabel(r"Magnetization $[\mathrm{G}]$")
    ax.set_title(f"{material.graphname} Magnetization vs Magnetic Field at {temp_val} K")
    plt.show()


def make_Bohr_vs_field(material: Crystal, Moment: List[float], Field: List[float], temp_val: float, per_formula_unit: bool = True):
    Bohr_vs_field: List[List[float]]
    if per_formula_unit:
        # 縦軸：有効ボーア磁子数/式量，横軸：磁場 のグラフを作成
        Bohr_vs_field = [[material.cal_Bohr_per_formula_unit(m=m),f] for m,f in zip(Moment,Field)] # 温度固定
    else:
        # 縦軸：有効ボーア磁子数/磁性イオン，横軸：磁場 のグラフを作成
        Bohr_vs_field = [[material.cal_Bohr_per_ion(m=m),f] for m,f in zip(Moment,Field)] # 温度固定
    X: List[float] = [f for b,f in Bohr_vs_field]
    Y: List[float] = [b for b,f in Bohr_vs_field]
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111)
    ax.plot(X, Y)
    ax.set_xlabel(r"Magnetic Field [Oe]")
    ax.set_ylabel(r"Magnetic Moment per ion $[\mu_B/ion]$")
    ax.set_title(f"{material.graphname} Magnetic Moment vs Magnetic Field at {temp_val} K")
    plt.show()


def make_susceptibility_vs_temp(material: Crystal, Moment: List[float], Temp: List[float], Field: float, SI: bool = False, per: Optional[str] = None): # データはcgs固定．グラフ描画をSIにするかどうか，1molあたりにするかどうか
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
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111)
    ax.plot(X, Y)
    ax.set_xlabel(r"Temperature [K]")
    if SI:
        if per == "mol":
            ax.set_ylabel(r"Susceptibility $[\mathrm{m^3\, mol^{-1}}]$")
        elif per == "weight":
            ax.set_ylabel(r"Susceptibility $[\mathrm{m^3\, kg^{-1}}]$")
        else:
            ax.set_ylabel(r"Susceptibility [dimensionless]")
    else:
        if per == "mol":
            ax.set_ylabel(r"Susceptibility $[\mathrm{cm^3\, mol^{-1}}]$")
        elif per == "weight":
            ax.set_ylabel(r"Susceptibility $[\mathrm{cm^3\, g^{-1}}]$")
        else:
            ax.set_ylabel(r"Susceptibility [dimensionless]")
    ax.set_title(f"{material.graphname} Susceptibility vs Temperature")
    plt.show()
    

def make_powder_Xray_intensity_vs_angle(filename: str, display_num: int = 10, material: Optional[Crystal] = None):
    with open(filename, encoding="shift_jis") as f:
        data: List[List[float]] = [list(map(float, s.strip().split())) for s in f.readlines()[3:]]
        N: int = len(data)
        two_theta: List[float] = [d[0] for d in data] # データは2θ
        theta: List[float] = [d[0]/2 for d in data] # データは2θ
        intensity: List[float] = [d[1] for d in data]
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
        for i, (_, p, theta_p, intensity_p) in enumerate(peak):
            if i == display_num:
                break
            d_hkl_over_n_alpha: float = Cu_K_alpha/np.sin(np.radians(theta_p))/2
            d_hkl_over_n_beta: float = Cu_K_beta/np.sin(np.radians(theta_p))/2
            print(f"θ = {theta_p}, 2θ = {2*theta_p}, intensity = {intensity_p}")
            print(f"    Kα: d_hkl/n = {d_hkl_over_n_alpha}")
            print(f"    Kβ: d_hkl/n = {d_hkl_over_n_beta}")
        
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        ax.plot(two_theta, intensity)
        ax.set_yscale('log')
        ax.set_xlabel(r"$2\theta\, [{}^{\circ}]$")
        ax.set_ylabel("intensity [count]")
        if material is not None:
            ax.set_title(f"{material.graphname} powder X-ray diffraction")
        else:
            ax.set_title(f"powder X-ray diffraction")
        plt.show()





def main():

    HoMn2Ge2: Crystal = Crystal(name="HoMn2Ge2")
    SnIn: Crystal = Crystal(name="SnIn")

    Ho: Crystal = Crystal("Ho")
    Ho.set_weight(0.0934)
    m: float = Ho.cal_mol() # Hoの物質量
    HoMn2Ge2.set_mol(m)
    HoMn2Ge2.cal_weight()
    SnIn.set_mol(m*20)
    SnIn.cal_weight()

    In = Crystal("In")
    In.set_weight(1.3087)
    In.cal_mol()
    SnIn.set_mol(In.mol)
    SnIn.cal_weight()

    print(SnIn.w + HoMn2Ge2.w)

    SnIn.cal_ingredients()
    HoMn2Ge2.cal_ingredients()


    """
    data_filename: str = "./210602NiPS3vertical.dc.dat"
    with open(data_filename, encoding="shift_jis", mode="r") as current_file:
        label: List[str] = []
        data: List[List[Any]] = []
        flag: int = 0
        for l in current_file.readlines():
            if flag == 0 and l == "[Data]\n":
                flag = 1
                continue
            if flag == 1:
                label = l.strip().split(",")
                flag = 2
            elif flag == 2:
                data.append(list(map(str_to_float,l.strip().split(","))))

        N: int = len(data)

        dict_label: Dict[str, int] = {v:k for k,v in enumerate(label)}
        Temp: List[float] =          [data[i][dict_label["Temperature (K)"]] for i in range(N)]
        Field: List[float] =         [data[i][dict_label["Field (Oe)"]] for i in range(N)]
        Time: List[float] =          [data[i][dict_label["Time"]] for i in range(N)]
        LongMoment: List[float] =    [data[i][dict_label["Long Moment (emu)"]] for i in range(N)]
        RegFit: List[float] =        [data[i][dict_label["Long Reg Fit"]] for i in range(N)]
        
        # 目視でデータの切れ目を確認
        for i in range(N):
            pass
            #print(i, Field[i], Temp[i], LongMoment[i], RegFit[i])

        idx: int = 146 # データの切れ目(手入力)
        field_fixed_moment: List[float] = LongMoment[:idx]
        field_fixed_temp: List[float] = Temp[:idx]

        temp_fixed_moment: List[float] = LongMoment[idx:]
        temp_fixed_field: List[float] = Field[idx:]

    # 試料のデータを元にCrystalインスタンスを生成
    NiPS3: Crystal = Crystal(name="NiPS3", graphname=r"$\mathrm{{NiPS_3}}$", date="20210602")
    NiPS3.set_lattice_constant(a=5.840, b=10.141, c=6.651, alpha=90, beta=107.13, gamma=90, num=4)
    NiPS3.set_formula_weight(185.86) # [g/mol]
    NiPS3.set_weight(0.0016) # [g]
    NiPS3.cal_density()
    NiPS3.cal_mol()
    NiPS3.set_num_magnetic_ion(1)
    NiPS3.save(overwrite=True)
    """
    """
    # 過去のデータから復元するときの例
    NiPS3 = Crystal(name="NiPS3")
    NiPS3.load("NiPS320210602")
    print(NiPS3)
    """


    """
    make_moment_vs_temp(NiPS3, field_fixed_moment, field_fixed_temp, field_val=1000)
    make_moment_vs_field(NiPS3, temp_fixed_moment, temp_fixed_field, temp_val=10)
    make_magnetization_vs_field(NiPS3, temp_fixed_moment, temp_fixed_field, temp_val=10, SI=True, per=True)
    make_magnetization_vs_temp(NiPS3, field_fixed_moment, field_fixed_temp, field_val=1000, SI=True, per=True)
    make_Bohr_vs_field(NiPS3, temp_fixed_moment, temp_fixed_field, temp_val=10)
    make_susceptibility_vs_temp(NiPS3, field_fixed_moment, field_fixed_temp, Field=1000, SI=True, per=True)
    """

    """
    Ni = Crystal(name="Ni",graphname=r"$\mathrm{{Ni}}$")
    Ni.set_lattice_constant(a=3.524, b=3.524, c=3.524, alpha=90, beta=90, gamma=90, num=4)
    Ni.set_formula_weight(58.6934) # [g/mol]
    Ni.set_weight(2) # [g]
    Ni.cal_density()
    Ni.set_num_magnetic_ion(1)
    print(Ni.cal_magnetization(109.12, SI=True, per=True))
    """


if __name__ == "__main__":
    main()

