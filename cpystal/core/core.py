"""`cpystal`: for dealing with crystals and experimental data of physical property.

`cpystal` is designed to handle experimental data on crystals.
It places particular emphasis on calculating and storing data on the properties of actual synthesized samples, 
and on graphing these data. In this respect, it is distinct from `pymatgen`, a similar crystal and material analysis module.
Of course, pymatgen is a very useful python module, so we use it as an adjunct in `cpystal`.
"""
from __future__ import annotations

from collections import defaultdict
from math import sqrt, cos, radians
import re
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np
import pickle
import scipy.signal # type: ignore


# データ処理用
def str_to_float(s: str) -> Optional[float]:
    """Convert str number to float object.

    Args:
        s (str): str-type number.
    
    Returns:
        (Optional[int]): None if s is empty-string, float otherwise
    """
    if s == '':
        return None
    else:
        return float(s)

# データ平滑化
def smoothing(data: List[float], mode: Any = None) -> List[float]:
    """Data smoothing function.

    Args:
        data (list[float]): 1-dimension data.
        mode (Any): To choose smoothing algorithm. Savitzky-Golay filter if mode == "s", Simple Moving Average otherwise.

    Returns:
        (list[float]): smoothed data.
    """
    if mode == "s": # Savitzky-Golay filter
        deg: int = 2 # 多項式の次数
        window_length: int = len(data)//4*2+1 # 窓幅
        deriv: int = 0 # 微分階数
        return list(map(float, scipy.signal.savgol_filter(data, window_length=window_length, polyorder=deg, deriv=deriv)))
      
    else: # 単純移動平均
        window: int = 9 # 移動平均の範囲
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


class Semimutable_dict(Dict[Any, Any]):
    """Semi-mutable dictionary inherited from `dict`

    The only difference from `dict` is that using `[]` is not allowed, but using `update_force` method is allowed to replace the value.
    """
    def __init__(self, *args: Any) -> None:
        super().__init__(args)
        self.__updatable: bool = False

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self and not self.__updatable:
            raise TypeError(f"elements of '{self.__class__.__name__}' cannot be changed by '[]' operator; use 'update_force' method")
        super().__setitem__(key, value)
        self.__updatable = False

    def update_force(self, key: Any, value: Any) -> None:
        """Instance method for replacing the value.

        Args:
            key (Any): Immutable object.
            value (Any): New value of `key`.
        """
        self.__updatable = True
        self[key] = value
        

class Crystal: # 結晶の各物理量を計算
    """A `Crystal` instance corresponds to a certain sample of a crystal.

    Note:
        Any attribute of a `Crystal` instance can be freely assigned from the time it is created until the end of the program (i.e. mutable).
        However, once it is saved as a pickle file, it becomes impossible to change the attributes of the instance created by loading from the pickle file (i.e. immutable).
        This feature was implemented to prevent the unintentional destruction of the sample data which is saved in a pickle file.

    Attributes:
        name (str): The chemical formula of the crystal.
        graphname (str): TeX-formed `name`.
        date (Optional[str]): The date the sample was synthesized. If there is a numbering system, it will be included here.
        NA (float): Avogadro constant.
        a (float): Lattice constant.
        b (float): Lattice constant.
        c (float): Lattice constant.
        alpha (float): Lattice constant.
        beta (float): Lattice constant.
        gamma (float): Lattice constant.
        V (float): Volume of a unit cell.
        fu_per_unit_cell (float): Number of formula unit in a unit cell.
        formula_weight (float): Weight of formula unit per mol.
        w (float): The weight of the sample.
        num_magnetic_ion (float): Number of magnetic ions in a formula unit.
        density (float): Density of the crystal.
        mol (float): The amount of substance of the sample.
        numbered_name (float): Changed `name` that the elemental numbers in the chemical formula are clearly indicated by adding '1'.
        components (Defaultdict[str, float]): Number of each element in a formula unit.
        unit (dict[str, str]): The unit of each attribute.
        graphs (Semimutable_dict[str, Any]): Semimutable dictionary of experimental data plotted in `matplotlib.axes._subplots.AxesSubplot` object.

    """
    __slots__ = ("name", "graphname", "date", "NA", 
                "a", "b", "c", "alpha", "beta", "gamma", "V", "fu_per_unit_cell",
                "formula_weight", "w", "num_magnetic_ion", "density", "mol",
                "numbered_name", "components", "unit", "graphs", "_Crystal__updatable",)

    def __init__(self, name: str, date: Optional[str] = None, auto_formula_weight: bool = True) -> None:
        """All attributes are initialized in `__init__` method.

        Args:
            name (str): The chemical formula of the crystal.
            date (Optional[str]): The date the sample was synthesized. If there is a numbering system, it will be included here.
            auto_formula_weight (bool): If this argument is `True`, `formula_weight` is calculated automatically from `name`.
        """
        self.__updatable: bool = True
        self.name: str = name # 化合物名
        self.graphname: str = "$\mathrm{" + re.sub(r"(\d+\.*\d*)", "_{\\1}", name) + "}$" # グラフで表示する名前
        self.date: Optional[str] = date # 合成した日付(必要ならナンバリングもここに含める)
        
        self.NA: float = 6.02214076 * 10**(23) # アボガドロ定数:[/mol]
        
        # 格子定数
        self.a: Optional[float] = None              # 格子定数 [Å]
        self.b: Optional[float] = None              # 格子定数 [Å]
        self.c: Optional[float] = None              # 格子定数 [Å]
        self.alpha: Optional[float] = None          # 基本並進ベクトル間の角度 [°]
        self.beta: Optional[float] = None           # 基本並進ベクトル間の角度 [°]
        self.gamma: Optional[float] = None          # 基本並進ベクトル間の角度 [°]
        self.V: Optional[float] = None              # 単位胞の体積 [cm^3]
        self.fu_per_unit_cell: Optional[int] = None # 単位胞に含まれる式単位の数 (無次元)

        self.formula_weight: Optional[float] = None # モル質量(式単位あたり) [g/mol]
        self.w: Optional[float] = None              # 試料の質量 [g]
        self.num_magnetic_ion: Optional[int] = None # 式単位中の磁性イオンの数 (無次元)
        self.density: Optional[float] = None        # 密度 [g/cm^3]
        self.mol: Optional[float] = None            # 物質量 [mol]

        self.numbered_name: str = re.sub(r"([A-Z][a-z]?|\))(?=[^0-9a-z]+)", r"\g<1>1", name+"$")[:-1] # '1'を追加して元素数を明示したname ("$"は番兵)
        self.components: DefaultDict[str, float] = defaultdict(float)   # 式単位中に各元素がいくつあるか

        # 各クラス変数の単位
        # 内部では基本的にCGS単位系を用いる
        self.unit: Dict[str, str] = {
            "unit": "",
            "NA": "mol^-1", "name": "", "graphname": "", "date": "",
            "a": "Å", "b": "Å", "c": "Å", "alpha": "°", "beta": "°", "gamma": "°",
            "V": "cm^3", "fu_per_unit_cell": "", "formula_weight": "g/mol", "w": "g", 
            "num_magnetic_ion": "", "density": "g/cm^3", "mol": "mol",
            "numbered_name": "", "components": ""
        }

        self.graphs: Semimutable_dict = Semimutable_dict()

        # 化学式を"形態素"ごとに分割したリスト
        divided_name: List[str] = re.split(r",+", re.sub(r"([A-Z][a-z]*|(\d|\.)+|[()])", ",\\1,", self.numbered_name).strip(","))
        now: float = 1.0 # 倍率
        num_stack: List[float] = [1.0] # 後ろから見て，現在有効な数の積を格納するstack
        for s in reversed(divided_name): # 化学式を後ろからみる
            if re.match(r"\d+\.*\d*", s): # 数値
                now *= float(s)
                num_stack.append(float(s))
            elif s == ")":
                pass
            elif s == "(": # ()を付けるときは必ず直後に1以上の数字が来る
                now /= num_stack.pop()
            else:
                self.components[s] += now
                now /= num_stack.pop()
        if auto_formula_weight: # nameから自動でモル質量を計算
            formula_weight: float = 0.0 # モル質量(式単位あたり)
            for element, n in self.components.items():
                if not element in atomic_weight:
                    raise KeyError
                formula_weight += atomic_weight[element] * n
            self.formula_weight = formula_weight


    def __str__(self) -> str:
        res: str = "\n"
        for k in self.__slots__:
            if not hasattr(self, k): # 後方互換性
                continue
            v: Any = getattr(self, k)
            if v is None or k == "unit" or k == "_Crystal__updatable":
                continue
            if not k in self.unit:
                res = res + f"{k} = {v}\n"
            elif type(v) is float:
                res = res + f"{k} = {v:.5g} {self.unit[k]}\n"
            else:
                res = res + f"{k} = {v} {self.unit[k]}\n"
        return res

    def __add__(self, other: Crystal) -> Crystal:
        if type(other) is not Crystal:
            raise TypeError(f"unsupported operand type(s) for +:{self.__class__.__name__} and {type(other).__name__}")
        return self.__class__(self.name + other.name)

    def __mul__(self, other: Union[int, float]) -> Crystal:
        if type(other) is not int:
            raise TypeError(f"unsupported operand type(s) for +:{self.__class__.__name__} and {type(other).__name__}")
        # 化学式をother倍する
        divided_name: List[str] = re.split(r",+", re.sub(r"([A-Z][a-z]*|(\d|\.)+|[()])", ",\\1,", self.numbered_name).strip(","))
        parentheses_depth: int = 0 # かっこの中にある数字は飛ばす
        for i, s in enumerate(divided_name):
            if s == "(":
                parentheses_depth += 1
            elif s == ")":
                parentheses_depth -= 1
            else:
                if parentheses_depth == 0 and re.match(r"\d+\.*\d*", s):
                    divided_name[i] = f"{float(s) * other:.4g}"
        return self.__class__("".join(divided_name))

    def __setattr__(self, name: str, value: Any) -> None:
        if name == f"_{self.__class__.__name__}__updatable":
            object.__setattr__(self, name, value)
        elif name in self.__slots__:
            if hasattr(self, name):
                if self.__updatable:
                    object.__setattr__(self, name, value)
                else:
                    raise TypeError(f"'{self.__class__.__name__}' object made by '{self.__class__.__name__}.load' is immutable")
            else:
                object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getstate__(self) -> Dict[Any, Any]:
        state: Dict[Any, Any] = {key: getattr(self, key) for key in self.__slots__}
        return state

    def __setstate__(self, state: Dict[Any, Any]) -> None:
        for key in self.__slots__:
            if key in state:
                object.__setattr__(self, key, state[key])
            else:
                # 後方互換性のため(後のバージョンで削除予定)
                if key == "fu_per_unit_cell" and "num" in state:
                    object.__setattr__(self, "fu_per_unit_cell", state["num"])
                if key == "_Crystal__updatable":
                    object.__setattr__(self, "_Crystal__updatable", False)

    def is_updatable(self) -> bool:
        """Return updatability of the instance.

        Returns:
            (bool): updatability of the instance.
        """
        return self.__updatable

    def set_lattice_constant(self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float, fu_per_unit_cell: Optional[int] = None) -> None:
        """Setting lattice constants of the crystal.

        Args:
            a (float): Lattice constant.
            b (float): Lattice constant.
            c (float): Lattice constant.
            alpha (float): Lattice constant.
            beta (float): Lattice constant.
            gamma (float): Lattice constant.
            fu_per_unit_cell (Optional[int]): Number of formula unit in a unit cell.
        """
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
        if fu_per_unit_cell is not None:
            self.fu_per_unit_cell = fu_per_unit_cell # 単位胞に含まれる式単位の数 (無次元)
    
    def set_formula_weight(self, formula_weight: float) -> None:
        """Setting formula weight (per formula unit) of the crystal.

        Args:
            formula_weight (float): Weight of formula unit per mol (unit: [g/mol]).
        """
        # コンストラクタでauto_formula_weight = Trueで自動設定可能
        # formula_weight: モル質量(式単位あたり) [g/mol]
        self.formula_weight = formula_weight

    def set_weight(self, w: float) -> None:
        """Setting the weight of the sample.

        Args:
            w (float): The weight of the sample (unit: [g]).
        """
        # w: 試料の質量 [g]
        self.w = w
    
    def set_mol(self, mol: float) -> None:
        """Setting the amount of substance of the sample.

        Args:
            mol (float): The amount of substance of the sample (unit: [mol]).
        """
        # mol: 試料の物質量 [mol]
        self.mol = mol

    def set_num_magnetic_ion(self, num_magnetic_ion: int) -> None:
        """Setting the number of magnetic ions in a formula unit.

        Args:
            num_magnetic_ion (float): Number of magnetic ions in a formula unit.
        """
        # num_magnetic_ion: 式単位中の磁性イオンの数 (無次元)
        self.num_magnetic_ion = num_magnetic_ion

    def cal_density(self) -> float:
        """Calculating the density of the crystal.

        Returns:
            (float): The density of the crystal (unit: [g/cm^3]).
        """
        # formula_weight: モル質量(式単位あたり) [g/mol]
        # fu_per_unit_cell: 単位胞の分子数 (無次元)
        # V: 単位胞の体積 [cm^3]
        # density: 密度 [g/cm^3]
        if self.formula_weight is None or self.fu_per_unit_cell is None or self.V is None:
            raise TypeError(f"one or more of the attributes are 'NoneType': 'formula_weight', 'fu_per_unit_cell', 'V'")
        self.density = self.formula_weight * self.fu_per_unit_cell / self.NA / self.V
        return self.density

    def cal_mol(self) -> float:
        """Calculating the amount of substance of the sample from the weight of the sample.

        Returns:
            (float): The amount of substance of the sample (unit: [mol]).
        """
        # formula_weight: モル質量(式単位あたり) [g/mol]
        # w: 試料の質量 [g]
        # mol: 試料の物質量 [mol]
        if self.formula_weight is None or self.w is None:
            raise TypeError(f"one or more of the attributes are 'NoneType': 'formula_weight', 'w'")
        self.mol = self.w / self.formula_weight
        return self.mol

    def cal_weight(self) -> float:
        """Calculating the weight of the sample from the amount of substance of the sample.

        Returns:
            (float): The weight of the sample (unit: [g]).
        """
        # formula_weight: モル質量(式単位あたり) [g/mol]
        # mol: 試料の物質量 [mol]
        # w: 試料の質量 [g]
        if self.formula_weight is None or self.mol is None:
            raise TypeError(f"one or more of the attributes are 'NoneType': 'formula_weight', 'mol'")
        self.w = self.formula_weight * self.mol
        return self.w

    def cal_magnetization(self, m: float, w: Optional[float] = None, SI: bool = False, per: Optional[str] = None) -> float:
        """Calculating magnetization from measured value of magnetic moment.

        Args:
            m (float): Magnetic moment (unit: [emu]).
            w (Optional[float]): The weight of the sample (unit: [g]).
            SI (bool): If True, magnetization is calculated in SI (MKSA) system.
            per (Optional[str]): If per == "mol", magnetization per mol is calculated. If per == "weight", magnetization per weight is calculated. 
        
        Returns:
            (float): Magnetization (unit: [G], [G/mol], [G/g], [A/m], [A/m/mol] or [A/m/kg]).
        """
        # m: 磁気モーメント [emu]
        # w: 試料の質量 [g]
        # density: 密度 [g/cm^3]
        # M: 磁化 [G] or [G/mol] or [A/m] or [A/(m mol)]
        if w is None:
            w = self.w
        if w is None or self.density is None or self.formula_weight is None:
            raise TypeError(f"one or more of the attributes are 'NoneType': 'formula_weight', 'w', 'density'")
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
        """Calculating magnetization in units of Bohr magneton per formula unit.

        Args:
            m (float): Magnetic moment (unit: [emu]).
            w (Optional[float]): The weight of the sample (unit: [g]).
        
        Returns:
            (float): Magnetization in units of Bohr magneton per formula unit.
        """
        muB: float = 9.274 * 10**(-21) # Bohr磁子 [emu]
        if w is None:
            w = self.w
        if w is None or self.formula_weight is None:
            raise TypeError(f"one or more of the attributes are 'NoneType': 'formula_weight', 'w'")
        # 式単位あたりの有効Bohr磁子数 [μB/f.u.]
        mu: float = (m / muB) / (w / self.formula_weight * self.NA)
        return mu

    def cal_Bohr_per_ion(self, m: float, w: Optional[float] = None, num_magnetic_ion: Optional[int] = None) -> float:
        """Calculating magnetization in units of Bohr magneton per magnetic ion.

        Args:
            m (float): Magnetic moment (unit: [emu]).
            w (Optional[float]): The weight of the sample (unit: [g]).
            num_magnetic_ion (Optional[float]): Number of magnetic ions in a formula unit.
        
        Returns:
            (float): Magnetization in units of Bohr magneton per magnetic ion.
        """
        muB: float = 9.274 * 10**(-21) # Bohr磁子 [emu]
        if w is None:
            w = self.w
        if num_magnetic_ion is None:
            num_magnetic_ion = self.num_magnetic_ion
        if w is None or num_magnetic_ion is None or self.formula_weight is None:
            raise TypeError(f"one or more of the attributes are 'NoneType': 'formula_weight', 'w', 'num_magnetic_ion'")
        # 磁性イオンあたりの有効Bohr磁子数 [μB/ion]
        mu: float = (m / muB) / (w / self.formula_weight * self.NA) / num_magnetic_ion
        return mu

    def cal_ingredients(self) -> List[Tuple[str, float]]:
        """Calculating the weight of each element in the sample.

        Returns:
            (list[tuple[str, float]]): List of tuple (element name, element weight ratio to the total).
        """
        # selfに含まれる各元素の重量をそれぞれ求める
        res: List[Tuple[str, float]] = []
        if self.formula_weight is None:
            raise TypeError(f"the attribute 'formula_weight' is 'NoneType'")
        for element, n in self.components.items():
            res.append((element, n*atomic_weight[element]/self.formula_weight))
        res = res[::-1]
        print(f"The ingredients of {self.name} ({self.w} g):")
        if self.w is None:
            print("\n".join([f"{element} = {ratio:.2%}" for element, ratio in res]))
        else:
            print("\n".join([f"{element} = {ratio*self.w:.4g} g ({ratio:.2%})" for element, ratio in res]))
        return res

    def save(self, filename: str, overwrite: bool = False) -> None: # Crystalインスタンスのデータを保存
        """Saving the `Crystal` instance data as a pickle file.

        Note:
            Once a `Crystal` instance is saved as a pickle file, the instance created by the class method `Crystal.load` from the pickle file will be an immutable object.

        Args:
            filename (str): Output file name (if necessary, add file path to the head). The suffix of `filename` must be ".pickle".
            overwrite (bool): If True, a file with the same name is overwritten.
        """
        if not filename.endswith(".pickle"):
            raise FileNotFoundError("suffix of 'filename' must be '.pickle'")
        self.__updatable = False # saveしたらimmutableオブジェクトになる
        mode: str
        if overwrite:
            mode = 'wb' # 上書きあり
        else:
            mode = 'xb' # 上書きなし
        with open(filename, mode=mode) as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str) -> Crystal:
        """Static method to load a `Crystal` instance from a pickle file.

        Note:
            Once a `Crystal` instance is saved as a pickle file, the instance created by the class method `Crystal.load` from the pickle file will be an immutable object.

        Args:
            filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` must be ".pickle".

        Returns:
            (Crystal): `Crystal` instance saved in the pickle file `filename`.
        """
        with open(filename, mode='rb') as f:
            res: Crystal = pickle.load(f)
        return res

    @classmethod
    def from_cif(cls, cif_filename: str) -> Crystal:
        """Class method to make a `Crystal` instance from a cif file.
        
        Args:
            cif_filename (str): Input file name (if necessary, add file path to the head).

        Returns:
            (Crystal): `Crystal` instance made from the cif file `cif_filename`.
        """
        with open(cif_filename) as f:
            lines: List[str] = f.readlines()
        for line in lines:
            if line.startswith("_chemical_formula_structural"):
                name: str = re.sub(r".+\'(.+)\'", "\\1", line).replace(" ", "").rstrip()
        res: Crystal = cls(name)
        a: float
        b: float
        c: float
        alpha: float
        beta: float
        gamma: float
        fu_per_unit_cell: int
        for line in lines:
            if line.startswith("_cell_length_a "):
                a = float(re.sub(r"\(.*\)", "", line).replace("_cell_length_a ", "").rstrip())
            if line.startswith("_cell_length_b "):
                b = float(re.sub(r"\(.*\)", "", line).replace("_cell_length_b ", "").rstrip())
            if line.startswith("_cell_length_c "):
                c = float(re.sub(r"\(.*\)", "", line).replace("_cell_length_c ", "").rstrip())
            if line.startswith("_cell_angle_alpha "):
                alpha = float(re.sub(r"\(.*\)", "", line).replace("_cell_angle_alpha ", "").rstrip())
            if line.startswith("_cell_angle_beta "):
                beta = float(re.sub(r"\(.*\)", "", line).replace("_cell_angle_beta ", "").rstrip())
            if line.startswith("_cell_angle_gamma "):
                gamma = float(re.sub(r"\(.*\)", "", line).replace("_cell_angle_gamma ", "").rstrip())
            if line.startswith("_cell_formula_units_Z "):
                fu_per_unit_cell = int(line.replace("_cell_formula_units_Z ", "").rstrip())
        res.set_lattice_constant(a, b, c, alpha, beta, gamma, fu_per_unit_cell)
        return res

# 型エイリアス
LF = List[float]
LLF = List[List[float]] 
class PPMSResistivity:
    """This is a class for acquiring experimental data of Physical Properties Measurement System (PPMS) from '.dat' files.

    Attributes:
        filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` may be ".dat".
        material (Optional[Crystal]): `Crystal` instance of the measurement object.
        Temp (List[float]): Temperature (K) data.
        Field (List[float]): Magnetic field (Oe) data.
        Time (List[float]): Time stamp (sec) data.
        B1Resistivity (List[float]): Bridge 1 Resistivity (Ohm) data.
        B2Resistivity (List[float]): Bridge 2 Resistivity (Ohm) data.
        B1R_sd (List[float]): Standard deviation of Bridge 1 Resistivity (Ohm) data.
        B2R_sd (List[float]): Standard deviation of Bridge 2 Resistivity (Ohm) data.
        B1Current (List[float]): Bridge 1 Current (μA) data.
        B2Current (List[float]): Bridge 2 Current (μA) data.

        (optional: when used `PPMS_Resistivity.set_S_l`)
        Sxx (float): Area of the sample perpendicular to the current (μm^2).
        Syx (float): Area of the sample parallel to the current (μm^2).
        lxx (float): Length of the sample parallel to the current (μm).
        lyx (float): Length of the sample perpendicular to the current (μm).
    """
    def __near_abs(self, x: float, k: float) -> float: # xに最も近いkの整数倍数
        if k == 0:
            return 0.0
        a: int = int(x/k)
        return min([(a-1)*k, a*k, (a+1)*k], key=lambda y:abs(x-y))

    def _LSM(self, x: LF, y: LF, linear: bool = False) -> Tuple[LF, float, float]: # 最小二乗法
        X: Any = np.array(x)
        Y: Any = np.array(y)
        if linear: # 線形関数近似
            a = X@Y / (X ** 2).sum()
            return list(a*X), a, 0
        else: # 1次関数近似
            n = len(X)
            xs = np.sum(X)
            ys = np.sum(Y)
            a = ((X@Y - xs*ys/n) / (np.sum(X ** 2) - xs**2/n))
            b = (ys - a * xs)/n
            return list(a*X + b), a, b
    
    def __init__(self, filename: str, material: Optional[Crystal] = None):
        """Initializer of `PPMS_Resistivity`.

        Args:
            filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` may be ".dat".
            material (Optional[Crystal]): `Crystal` instance of the measurement object.
        """
        self.filename: str = filename
        self.material: Optional[Crystal] = material

        with open(filename, encoding="shift_jis", mode="r") as current_file:
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
        self.Temp: LF =          [data[i][dict_label["Temperature (K)"]] for i in range(N)]
        self.Field: LF =         [data[i][dict_label["Magnetic Field (Oe)"]] for i in range(N)]
        self.Time: LF =          [data[i][dict_label["Time Stamp (sec)"]] for i in range(N)]
        self.B1Resistivity: LF = [data[i][dict_label["Bridge 1 Resistivity (Ohm)"]] for i in range(N)]
        self.B2Resistivity: LF = [data[i][dict_label["Bridge 2 Resistivity (Ohm)"]] for i in range(N)]
        self.B1R_sd: LF =        [data[i][dict_label["Bridge 1 Std. Dev. (Ohm)"]] for i in range(N)]
        self.B2R_sd: LF =        [data[i][dict_label["Bridge 2 Std. Dev. (Ohm)"]] for i in range(N)]
        self.B1Current: LF =     [data[i][dict_label["Bridge 1 Excitation (uA)"]] for i in range(N)]
        self.B2Current: LF =     [data[i][dict_label["Bridge 2 Excitation (uA)"]] for i in range(N)]


    def set_S_l(self, Sxx: float, lxx: float, Syx: float, lyx: float) -> None: # S:[μm^2], l:[μm]
        self.Sxx: float = Sxx
        self.Syx: float = Syx
        self.lxx: float = lxx
        self.lyx: float = lyx
    
    def symmetrize(self, delta_H: float, up_data: LLF, down_data: LLF) -> Tuple[LF, LF, LF, LF, LF]:
        """Symmetrization and antisymmetrization are performed based on the data of the field-increasing and field-decreasing processes.

        Args:
            delta_H (float): Difference of the magnetic field between each step.
            up_data (List[List[float]]): List of [field (float), Rxx (float), Rxx_sd (float), Ryx (float), Ryx_sd (float)] which represents field-increasing data.
            down_data (List[List[float]]): List of [field (float), Rxx (float), Rxx_sd (float), Ryx (float), Ryx_sd (float)] which represents field-decreasing data.

        Returns:
            (Tuple[List[float], List[float], List[float]], List[float], List[float]): 
                The first element of return value is 'effective_field' that is the list of field value whose reverse field value exists in data.
                The second and third element of return value is the list of Rxx and Rxx_sd.
                The fourth and fifth element of return value is the list of Ryx and Ryx_sd.
        """
        
        # (up/down)_data := List[List[field: float, Rxx: float, Rxx_sd: float, Ryx: float, Ryx_sd: float]]
        # 磁場を1往復させたときのデータから，Rxx・Ryxをそれぞれ対称化・反対称化

        up_idx:   Dict[float, int] = {self.__near_abs(h, delta_H):i for i, (h, *_) in enumerate(up_data)}
        down_idx: Dict[float, int] = {self.__near_abs(-h, delta_H):i for i, (h, *_) in enumerate(down_data)}
        
        effective_field: LF = []
        Rxx: LF = []
        Ryx: LF = []

        Rxx_sd: LF = []
        Ryx_sd: LF = []

        for h in sorted(set(down_idx.keys()) & set(up_idx.keys())):
            i: int = up_idx[h]
            j: int = down_idx[h]
            effective_field.append(h)
            _, Rxx_i, Rxx_sd_i, Ryx_i, Ryx_sd_i = up_data[i]
            _, Rxx_j, Rxx_sd_j, Ryx_j, Ryx_sd_j = down_data[j]
            # 対称化・反対称化
            Rxx.append( (Rxx_i+Rxx_j)/2 ) # [Ω]
            Ryx.append( (Ryx_i-Ryx_j)/2 ) # [Ω]
            # 標準偏差の伝播則
            Rxx_sd.append( (Rxx_sd_i**2+Rxx_sd_j**2)**0.5 / 2 ) # [Ω]
            Ryx_sd.append( (Ryx_sd_i**2+Ryx_sd_j**2)**0.5 / 2 ) # [Ω]
        return effective_field, Rxx, Rxx_sd, Ryx, Ryx_sd
                    

class MPMS:
    """This is a class for acquiring experimental data of Magnetic Property Measurement System (MPMS) from '.dat' files.
    
    Attributes:
        filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` may be ".dat".
        material (Optional[Crystal]): `Crystal` instance of the measurement object.
        Temp (List[float]): Temperature (K) data.
        Field (List[float]): Magnetic field (Oe) data.
        Time (List[float]): Time stamp (sec) data.
        LongMoment (List[float]): Longitudinal moment (emu) data.
        Regfit (List[float]): Regression fit of longitudinal moment data.
    """
    def __init__(self, filename: str, material: Optional[Crystal] = None):
        """Initializer of `MPMS`.

        Args:
            filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` may be ".dat".
            material (Optional[Crystal]): `Crystal` instance of the measurement object.
        """
        self.filename: str = filename
        self.material: Optional[Crystal] = material

        with open(filename, encoding="shift_jis", mode="r") as current_file:
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

        self.Temp: List[float] =          [data[i][dict_label["Temperature (K)"]] for i in range(N)]
        self.Field: List[float] =         [data[i][dict_label["Field (Oe)"]] for i in range(N)]
        self.Time: List[float] =          [data[i][dict_label["Time"]] for i in range(N)]
        self.LongMoment: List[float] =    [data[i][dict_label["Long Moment (emu)"]] for i in range(N)]
        self.RegFit: List[float] =        [data[i][dict_label["Long Reg Fit"]] for i in range(N)]


def ingredient_flake_dp(A: List[int], W: int) -> None: # A: 適当に整数化したフレークの重さ, W: 目標重量
    """Choose optimal flakes whose total weight meets the target value.

    Note:
        The result will be output to stdout.

    Args:
        A (List[int]): List of weight of flakes, properly integerized.
        W (int): Target weight value.
    """
    N: int = len(A)
    K: int = W+20 # 余裕を持って求めておく
    dp: List[List[int]] = [[0]*K for i in range(N+1)]
    dp[0][0] = 1
    for i in range(1,N+1):
        for j in range(K):
            if dp[i-1][j] and A[i-1]+j<K:
                dp[i][A[i-1]+j] = 1
            if dp[i-1][j]:
                dp[i][j] = 1

    #print([i for i in range(K) if dp[N][i]])
    for k in range(-10,11): # 目標値のまわり±10を見る
        now: int = W+k
        ans: List[int] = []
        if dp[N][now]:
            for i in range(N)[::-1]:
                if now-A[i]>=0 and dp[i][now-A[i]]:
                    now -= A[i]
                    ans.append(A[i])
        print(W+k, ans)
    return



def main() -> None:
    pass
    return

if __name__ == "__main__":
    main()

