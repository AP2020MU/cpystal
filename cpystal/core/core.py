"""`cpystal`: for dealing with crystals and experimental data of physical property.

`cpystal` is designed to handle experimental data on crystals.
It places particular emphasis on calculating and storing data on the properties of actual synthesized samples, 
and on graphing these data. In this respect, it is distinct from `pymatgen`, a similar crystal and material analysis module.
Of course, pymatgen is a very useful python module, so we use it as an adjunct in `cpystal`.
"""
from __future__ import annotations

from collections import defaultdict
from math import pi, sqrt, cos, radians
import re
from typing import Any, Dict, List, TypeVar

import numpy as np
import numpy.typing as npt
import scipy.signal # type: ignore


# データ処理用
def str_to_float(s: str) -> float | None:
    """Convert str number to float object.

    Args:
        s (str): str-type number.
    
    Returns:
        (int | None): None if s is empty-string, float otherwise
    """
    if s == '':
        return None
    else:
        return float(s)

# データ平滑化
def smoothing(data: list[float], mode: Any = None) -> list[float]:
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
#           ・Dict[str, float | None] にするとmypyのチェック回避が面倒なので削除した
#           ・ただし，安定同位体が存在しない元素のうちBi,Th,Pa,Uは地球上で固有の同位体比を示すため，原子量が定義されている
#       ・環境中の変動とは別に測定誤差も考慮する必要があるが，以下では記載していない
atomic_weight: dict[str, float] = {
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


class SemimutableDict(Dict[Any, Any]):
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
        

Crystalchild = TypeVar("Crystalchild", bound="Crystal")

class Crystal: # 結晶の各物理量を計算
    """A `Crystal` instance corresponds to a certain sample of a crystal.

    Note:
        Any attribute of a `Crystal` instance can be freely assigned from the time it is created until the end of the program (i.e. mutable).
        However, once it is saved as a pickle file, it becomes impossible to change the attributes of the instance created by loading from the pickle file (i.e. immutable).
        This feature was implemented to prevent the unintentional destruction of the sample data which is saved in a pickle file.

    Attributes:
        name (str): The chemical formula of the crystal.
        graphname (str): TeX-formed `name`.
        date (str | None): The date the sample was synthesized. If there is a numbering system, it will be included here.
        spacegroup_name (str): Space group name in International (Hermann-Mauguin) notation of the crytal.
        NA (float): Avogadro constant 6.02214076 * 10**(23) [mol^-1].
        muB (float): Bohr magneton 9.27401 * 10**(-21) [emu].
        kB (float): Boltzmann constant 1.380649 * 10**(-23) [J/K].
        a (float): Lattice constant [Å].
        b (float): Lattice constant [Å].
        c (float): Lattice constant [Å].
        alpha (float): Lattice constant [°].
        beta (float): Lattice constant [°].
        gamma (float): Lattice constant [°].
        V (float): Volume of a unit cell [cm^3].
        Z (float): Number of formula unit in a unit cell.
        formula_weight (float): Weight of formula unit per mol [g/mol].
        w (float): The weight of the sample [g].
        num_magnetic_ion (float): Number of magnetic ions in a formula unit.
        density (float): Density of the crystal [g/cm^3].
        mol (float): The amount of substance of the sample [mol].
        numbered_name (float): Changed `name` that the elemental numbers in the chemical formula are clearly indicated by adding '1'.
        components (Defaultdict[str, float]): Number of each element in a formula unit.
        unit (dict[str, str]): The unit of each attribute.
        graphs (SemimutableDict[str, Any]): Semimutable dictionary of experimental data plotted as `matplotlib.axes._subplots.AxesSubplot` object.

    """
    attributes: tuple[str, ...] = ("name", "graphname", "date", "spacegroup_name", "NA", "muB", "kB",
                "a", "b", "c", "alpha", "beta", "gamma", "V", "Z",
                "formula_weight", "w", "num_magnetic_ion", "density", "mol",
                "numbered_name", "components", "unit", "graphs",)

    def __init__(self, name: str, date: str | None = None, auto_formula_weight: bool = True) -> None:
        """All attributes are initialized in `__init__` method.

        Args:
            name (str): The chemical formula of the crystal.
            date (str | None): The date the sample was synthesized. If there is a numbering system, it will be included here.
            auto_formula_weight (bool): If this argument is `True`, `formula_weight` is calculated automatically from `name`.
        """

        self.name: str = name # 化合物名
        self.graphname: str = "$\mathrm{" + re.sub(r"(\d+\.*\d*)", "_{\\1}", name) + "}$" # グラフで表示する名前
        self.date: str | None = date # 合成した日付(必要ならナンバリングもここに含める)
        self.spacegroup_name: str  # 空間群名(国際表記)
        
        self.NA: float = 6.02214076e+23 # Avogadro定数 [mol^-1]
        self.muB: float = 9.27401e-21 # Bohr磁子 [emu]
        self.kB: float = 1.380649e-23 # Boltzmann定数 [J/K]
        
        # 格子定数
        self.a: float              # 格子定数 [Å]
        self.b: float              # 格子定数 [Å]
        self.c: float              # 格子定数 [Å]
        self.alpha: float          # 基本並進ベクトル間の角度 [°]
        self.beta: float           # 基本並進ベクトル間の角度 [°]
        self.gamma: float          # 基本並進ベクトル間の角度 [°]
        self.V: float              # 単位胞の体積 [cm^3]
        self.Z: int                # 単位胞に含まれる式単位の数 (無次元)

        self.formula_weight: float # モル質量(式単位あたり) [g/mol]
        self.w: float              # 試料の質量 [g]
        self.num_magnetic_ion: int # 式単位中の磁性イオンの数 (無次元)
        self.density: float        # 密度 [g/cm^3]
        self.mol: float            # 物質量 [mol]

        self.numbered_name: str = re.sub(r"([A-Z][a-z]?|\))(?=[^0-9a-z]+)", r"\g<1>1", name+"$")[:-1] # '1'を追加して元素数を明示したname ("$"は番兵)
        self.components: defaultdict[str, float] = defaultdict(float)   # 式単位中に各元素がいくつあるか

        # 各インスタンス変数の単位
        self.unit: dict[str, str] = {
            "NA": "mol^-1", "muB": "emu", "kB": "J/K",
            "a": "Å", "b": "Å", "c": "Å", "alpha": "°", "beta": "°", "gamma": "°",
            "V": "cm^3", "Z": "", "formula_weight": "g/mol", "w": "g", 
            "num_magnetic_ion": "", "density": "g/cm^3", "mol": "mol",
        }

        self.graphs: SemimutableDict = SemimutableDict()

        # 化学式を"形態素"ごとに分割したリスト
        divided_name: list[str] = re.split(r",+", re.sub(r"([A-Z][a-z]*|(\d|\.)+|[()])", ",\\1,", self.numbered_name).strip(","))
        now: float = 1.0 # 倍率
        num_stack: list[float] = [1.0] # 後ろから見て，現在有効な数の積を格納するstack
        for s in reversed(divided_name): # 化学式を後ろからみる
            if re.match(r"\d+\.*\d*", s): # 小数表示を許した数値
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
            try:
                formula_weight: float = 0.0 # モル質量(式単位あたり)
                for element, n in self.components.items():
                    if not element in atomic_weight:
                        raise KeyError
                    formula_weight += atomic_weight[element] * n
                self.formula_weight = formula_weight
            except KeyError:
                print(f"'name' includes atom(s) whose atomic weight is undefined. please set 'formula_weight' manually")

    def __str__(self: Crystalchild) -> str:
        res: str = "\n"
        for k in self.__dict__:
            v: Any = getattr(self, k)
            if v is None or k == "unit":
                continue
            if not k in self.unit:
                res = res + f"{k} = {v}\n"
            elif type(v) is float:
                res = res + f"{k} = {v:.5g} {self.unit[k]}\n"
            else:
                res = res + f"{k} = {v} {self.unit[k]}\n"
        return res

    def __add__(self: Crystalchild, other: Crystalchild) -> Crystalchild:
        if isinstance(other, Crystal):
            raise TypeError(f"unsupported operand type(s) for +:{self.__class__.__name__} and {type(other).__name__}")
        return self.__class__(self.name + other.name)

    def __mul__(self: Crystalchild, other: int | float) -> Crystalchild:
        if type(other) is not int:
            raise TypeError(f"unsupported operand type(s) for *:{self.__class__.__name__} and {type(other).__name__}")
        # 化学式をother倍する
        divided_name: list[str] = re.split(r",+", re.sub(r"([A-Z][a-z]*|(\d|\.)+|[()])", ",\\1,", self.numbered_name).strip(","))
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

    @property
    def num_atom_per_formula_unit(self) -> int:
        """Return the number of atom in a formula unit.
        """
        return sum(map(int, self.components.values()))

    @property
    def fu_per_unit_cell(self) -> int:
        """Return the number of formula unit in a unit cell.
        """
        return self.Z

    def set_lattice_constant(self: Crystalchild, a: float, b: float, c: float, alpha: float, beta: float, gamma: float, Z: int | None = None) -> None:
        """Setting lattice constants of the crystal.

        Args:
            a (float): Lattice constant.
            b (float): Lattice constant.
            c (float): Lattice constant.
            alpha (float): Lattice constant.
            beta (float): Lattice constant.
            gamma (float): Lattice constant.
            Z (int | None): Number of formula unit in a unit cell.
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
        if Z is not None:
            self.Z = Z # 単位胞に含まれる式単位の数 (無次元)
    
    def set_spacegroup_name(self: Crystalchild, spacegroup_name: str) -> None:
        """Setting space group name of the crystal.

        Args:
            spacegroup_name (str): Space group name in International (Hermann-Mauguin) notation.
        """
        self.spacegroup_name = spacegroup_name

    def set_formula_weight(self: Crystalchild, formula_weight: float) -> None:
        """Setting formula weight (per formula unit) of the crystal.

        Args:
            formula_weight (float): Weight of formula unit per mol (unit: [g/mol]).
        """
        # コンストラクタでauto_formula_weight = Trueで自動設定可能
        # formula_weight: モル質量(式単位あたり) [g/mol]
        self.formula_weight = formula_weight

    def set_weight(self: Crystalchild, w: float) -> None:
        """Setting the weight of the sample.

        Args:
            w (float): The weight of the sample (unit: [g]).
        """
        # w: 試料の質量 [g]
        self.w = w
    
    def set_mol(self: Crystalchild, mol: float) -> None:
        """Setting the amount of substance of the sample.

        Args:
            mol (float): The amount of substance of the sample (unit: [mol]).
        """
        # mol: 試料の物質量 [mol]
        self.mol = mol

    def set_num_magnetic_ion(self: Crystalchild, num_magnetic_ion: int) -> None:
        """Setting the number of magnetic ions in a formula unit.

        Args:
            num_magnetic_ion (float): Number of magnetic ions in a formula unit.
        """
        # num_magnetic_ion: 式単位中の磁性イオンの数 (無次元)
        self.num_magnetic_ion = num_magnetic_ion

    def cal_density(self: Crystalchild) -> float:
        """Calculating the density of the crystal.

        Returns:
            (float): The density of the crystal (unit: [g/cm^3]).
        """
        # formula_weight: モル質量(式単位あたり) [g/mol]
        # Z: 単位胞中の式単位数 (無次元)
        # V: 単位胞の体積 [cm^3]
        # density: 密度 [g/cm^3]
        if self.formula_weight is None or self.Z is None or self.V is None:
            raise TypeError(f"one or more of the attributes are 'NoneType': 'formula_weight', 'Z', 'V'")
        self.density = self.formula_weight * self.Z / self.NA / self.V
        return self.density

    def cal_mol(self: Crystalchild) -> float:
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

    def cal_weight(self: Crystalchild) -> float:
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

    def cal_magnetization(self: Crystalchild, m: float, w: float | None = None, SI: bool = False, per: str | None = None) -> float:
        """Calculating magnetization from measured value of magnetic moment.

        Args:
            m (float): Magnetic moment (unit: [emu]).
            w (float | None): The weight of the sample (unit: [g]).
            SI (bool): If True, magnetization is calculated in SI (MKSA) system.
            per (str | None): If per == "mol", magnetization per mol is calculated. If per == "weight", magnetization per weight is calculated. 
        
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

    def cal_Bohr_per_formula_unit(self: Crystalchild, m: float, w: float | None = None) -> float:
        """Calculating magnetization in units of Bohr magneton per formula unit.

        Args:
            m (float): Magnetic moment (unit: [emu]).
            w (float | None): The weight of the sample (unit: [g]).
        
        Returns:
            (float): Magnetization in units of Bohr magneton per formula unit.
        """
        if w is None:
            w = self.w
        if w is None or self.formula_weight is None:
            raise TypeError(f"one or more of the attributes are 'NoneType': 'formula_weight', 'w'")
        # 式単位あたりの有効Bohr磁子数 [μB/f.u.]
        mu: float = (m / self.muB) / (w / self.formula_weight * self.NA)
        return mu

    def cal_Bohr_per_ion(self: Crystalchild, m: float, w: float | None = None, num_magnetic_ion: int | None = None) -> float:
        """Calculating magnetization in units of Bohr magneton per magnetic ion.

        Args:
            m (float): Magnetic moment (unit: [emu]).
            w (float | None): The weight of the sample (unit: [g]).
            num_magnetic_ion (float | None): Number of magnetic ions in a formula unit.
        
        Returns:
            (float): Magnetization in units of Bohr magneton per magnetic ion.
        """
        if w is None:
            w = self.w
        if num_magnetic_ion is None:
            num_magnetic_ion = self.num_magnetic_ion
        if w is None or num_magnetic_ion is None or self.formula_weight is None:
            raise TypeError(f"one or more of the attributes are 'NoneType': 'formula_weight', 'w', 'num_magnetic_ion'")
        # 磁性イオンあたりの有効Bohr磁子数 [μB/ion]
        mu: float = (m / self.muB) / (w / self.formula_weight * self.NA) / num_magnetic_ion
        return mu

    def cal_ingredients(self: Crystalchild) -> list[tuple[str, float]]:
        """Calculating the weight of each element in the sample.

        Returns:
            (list[tuple[str, float]]): list of tuple (element name, element weight ratio to the total).
        """
        # selfに含まれる各元素の重量をそれぞれ求める
        res: list[tuple[str, float]] = []
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

    def cal_Bpfu_to_emu(self, mu: float) -> float:
        """Calculating 'emu'-unit magnetization from the value of 'muB/f.u.'-unit magnetization.

        Args:
            mu (float): 'muB/f.u.'-unit magnetization.
        
        Returns:
            (float): 'emu'-unit magnetization
        """
        return mu * self.muB * self.w / self.formula_weight * self.NA

    def cal_effective_moment(self, Curie_constant: float) -> float:
        """Calculating effective moment from the value of Curie constant.

        Note:
            C = Nμ^2/3kB
            μ^2 = g^2 J(J+1) μB^2
            where N is the number of magnetic ion per mol,
                g is Lande factor,
                J is total angular momentum quantum number.
            (emu*Oe = erg = 10^(-7) Joule)

        Args:
            Curie_constant (float): Curie constant (unit: [emu.K/mol.Oe]).

        Returns:
            (float): Effective moment μ/μB = g√J(J+1).
        """
        return (Curie_constant / (self.num_magnetic_ion * self.NA / 3 / self.kB) * 10**7) ** 0.5 / self.muB

    def cal_phonon_velocity(self, TD: float) -> float:
        """Calculating phonon velocity from Debye temperature.

        Note:
            k_B T_D = \hbar \omega_D = \hbar v k_D,
            where Debye wave number k_D satifies 4/3 \pi k_D^3 = (2\pi)^3 N / V,
            where N is the number of atom in the sample and V is the volume of the sample.

        Args:
            TD (float): Debye temperature (unit: [K]).
        
        Returns:
            (float): Phonon velocity (unit: [cm/s]).
        """
        P: PhysicalConstant = PhysicalConstant()
        return P.kB * TD / P.hbar * (6*P.pi**2 * self.num_atom_per_formula_unit * self.Z / self.V)**(-1/3) # cm/s

    @classmethod
    def from_cif(cls, cif_filename: str) -> Crystal:
        """Class method to make a `Crystal` instance from a cif file.
        
        Args:
            cif_filename (str): Input file name (if necessary, add file path to the head).

        Returns:
            (Crystalchild): `Crystal` instance made from the cif file `cif_filename`.
        """
        with open(cif_filename) as f:
            lines: list[str] = f.readlines()
        for line in lines:
            if line.startswith("_chemical_formula_structural"):
                name: str = re.sub(r".+\'(.+)\'", "\\1", re.sub(r"_chemical_formula_structural", "", line)).replace(" ", "").rstrip()
        res: Crystal = cls(name)
        a: float
        b: float
        c: float
        alpha: float
        beta: float
        gamma: float
        Z: int
        spacegroup_name: str
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
                Z = int(line.replace("_cell_formula_units_Z ", "").rstrip())
            if line.startswith("_space_group_name_H-M_alt"):
                spacegroup_name = line.replace("_space_group_name_H-M_alt ", "").rstrip()
        res.set_lattice_constant(a, b, c, alpha, beta, gamma, Z)
        res.spacegroup_name = spacegroup_name
        return res

# 型エイリアス
LF = List[float]
LLF = List[List[float]]
class PPMSResistivity:
    """This is a class for acquiring experimental data of Physical Properties Measurement System (PPMS) from '.dat' files.

    Attributes:
        filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` may be ".dat".
        material (Crystal | None): `Crystal` instance of the measurement object.
        Temp (list[float]): Temperature (K) data.
        Field (list[float]): Magnetic field (Oe) data.
        Time (list[float]): Time stamp (sec) data.
        B1Resistivity (list[float]): Bridge 1 Resistivity (Ohm) data.
        B2Resistivity (list[float]): Bridge 2 Resistivity (Ohm) data.
        B1R_sd (list[float]): Standard deviation of Bridge 1 Resistivity (Ohm) data.
        B2R_sd (list[float]): Standard deviation of Bridge 2 Resistivity (Ohm) data.
        B1Current (list[float]): Bridge 1 Current (μA) data.
        B2Current (list[float]): Bridge 2 Current (μA) data.

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

    def _LSM(self, x: LF, y: LF, linear: bool = False) -> tuple[LF, float, float]: # 最小二乗法
        X: Any = np.array(x)
        Y: Any = np.array(y)
        if linear: # 線形関数近似
            a = X@Y / (X ** 2).sum()
            return list(a*X), a, 0
        else: # 1次関数近似
            n = len(X)
            xs = np.sum(X)
            Moment_sum = np.sum(Y)
            a = ((X@Y - xs*Moment_sum/n) / (np.sum(X ** 2) - xs**2/n))
            b = (Moment_sum - a * xs)/n
            return list(a*X + b), a, b
    
    def __init__(self, filename: str, material: Crystal):
        """Initializer of `PPMS_Resistivity`.

        Args:
            filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` may be ".dat".
            material (Crystal | None): `Crystal` instance of the measurement object.
        """
        self.filename: str = filename
        self.material: Crystal | None = material

        with open(filename, encoding="shift_jis", mode="r") as current_file:
            label: list[str] = []
            data: list[list[Any]] = []
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

        dict_label: dict[str, int] = {v:k for k,v in enumerate(label)}
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
    
    def symmetrize(self, delta_H: float, up_data: LLF, down_data: LLF) -> tuple[LF, LF, LF, LF, LF]:
        """Symmetrization and antisymmetrization are performed based on the data of the field-increasing and field-decreasing processes.

        Args:
            delta_H (float): Difference of the magnetic field between each step.
            up_data (list[list[float]]): list of [field (float), Rxx (float), Rxx_sd (float), Ryx (float), Ryx_sd (float)] which represents field-increasing data.
            down_data (list[list[float]]): list of [field (float), Rxx (float), Rxx_sd (float), Ryx (float), Ryx_sd (float)] which represents field-decreasing data.

        Returns:
            (tuple[list[float], list[float], list[float]], list[float], list[float]): 
                The first element of return value is 'effective_field' that is the list of field value whose reverse field value exists in data.
                The second and third element of return value is the list of Rxx and Rxx_sd.
                The fourth and fifth element of return value is the list of Ryx and Ryx_sd.
        """
        
        # (up/down)_data := list[list[field: float, Rxx: float, Rxx_sd: float, Ryx: float, Ryx_sd: float]]
        # 磁場を1往復させたときのデータから，Rxx・Ryxをそれぞれ対称化・反対称化

        up_idx:   dict[float, int] = {self.__near_abs(h, delta_H):i for i, (h, *_) in enumerate(up_data)}
        down_idx: dict[float, int] = {self.__near_abs(-h, delta_H):i for i, (h, *_) in enumerate(down_data)}
        
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
        material (Crystal | None): `Crystal` instance of the measurement object.
        N (int): Length of data.
        Temp (list[float]): Temperature (K) data.
        Field (list[float]): Magnetic field (Oe) data.
        Time (list[float]): Time stamp (sec) data.
        LongMoment (list[float]): Longitudinal moment (emu) data.
        Regfit (list[float]): Regression fit of longitudinal moment data.
    """
    def __init__(self, filename: str, material: Crystal):
        """Initializer of `MPMS`.

        Args:
            filename (str): Input file name (if necessary, add file path to the head). The suffix of `filename` may be ".dat".
            material (Crystal): `Crystal` instance of the measurement object.
        """
        self.filename: str = filename
        self.material: Crystal = material

        with open(filename, encoding="shift_jis", mode="r") as current_file:
            label: list[str] = []
            data: list[list[Any]] = []
            flag: int = 0
            for l in current_file.readlines():
                if flag == 0 and l.startswith("[Data]"):
                    flag = 1
                    continue
                if flag == 1:
                    label = re.split(r",\t*", l.strip())
                    flag = 2
                elif flag == 2:
                    data.append(list(map(str_to_float, re.split(r",\t*", l.strip()))))

        N: int = len(data)

        dict_label: dict[str, int] = {v:k for k,v in enumerate(label)}
        self.N: int = N
        self.Temp: list[float] =          [data[i][dict_label["Temperature (K)"]] for i in range(N)]
        self.Field: list[float] =         [data[i][dict_label["Field (Oe)"]] for i in range(N)]
        self.Time: list[float] =          [data[i][dict_label["Time"]] for i in range(N)]
        self.LongMoment: list[float] =    [data[i][dict_label["Long Moment (emu)"]] for i in range(N)]
        self.RegFit: list[float] =        [data[i][dict_label["Long Reg Fit"]] for i in range(N)]

    def __str__(self) -> str:
        res: list[str] = []
        res.append("----------------------------")
        res.append("idx, Temp, Field, LongMoment")
        for i in range(self.N):
            res.append(f"{i}, {self.Temp[i]}, {self.Field[i]}, {self.LongMoment[i]}")
        res.append("----------------------------")
        return "\n".join(res)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return [key, self.Temp[key], self.Field[key], self.LongMoment[key]]
        elif isinstance(key, slice):
            start: int = 0
            stop: int = self.N
            step: int = 1
            if key.start is not None:
                start = key.start
            if key.stop is not None:
                stop = key.stop
            if key.step is not None:
                step = key.step
            return list(zip(range(start,stop,step), self.Temp[key], self.Field[key], self.LongMoment[key]))
        else:
            raise KeyError("key must be 'int' or 'slice'")

    def cal_Curie_Weiss_temp(self, temp: list[float], moment: list[float], field: float) -> tuple[float, float]:
        """Calculating Curie-Weiss temperature and Curie constant.

        Note:
            χ = M/H = C / (T-θc), where θc is Curie-Weiss temperature and C is Curie constant

        Args:
            temp (list[float]): list of temperature (K). Elements must be above Curie (or Neel) temperature.
            moment (list[float]): list of magnetic moment (emu).

        Returns:
            (tuple[float, float]): Curie-Weiss temperature (K) and Curie constant (emu.K/mol.Oe).
        """
        if self.material is None:
            raise TypeError
        Temp: npt.NDArray[np.float64] = np.array(temp)
        sus_inv: npt.NDArray[np.float64] = np.array(field / (np.array(moment) / self.material.mol))
        n: int = len(Temp)
        Temp_sum: float = np.sum(Temp)
        susinv_sum: float = np.sum(sus_inv)
        a: float = ((Temp@sus_inv - Temp_sum*susinv_sum/n) / (np.sum(Temp ** 2) - Temp_sum**2/n))
        b: float = (susinv_sum - a * Temp_sum)/n
        theta_Curie_Weiss: float = -(b/a)
        Curie_constant: float = 1/a
        return theta_Curie_Weiss, Curie_constant

    def Bpfu(self) -> list[float]:
        """Moment list in 'μB/f.u.'.
        """
        return [self.material.cal_Bohr_per_formula_unit(m) for m in self.LongMoment]

    def Field_T(self) -> list[float]:
        """Field list in 'Testa'.
        """
        return [f/10000 for f in self.Field]

    def Susceptibility(self, H: float, magnification: float = 1.0) -> list[float]:
        """χ=M/H list in 'emu/mol.Oe'.

        Args:
            H (float): Magnetic field (Oe).
            magnification (float): Amplification magnification. Defaults to 1.0.
        """
        return [magnification * m * (self.material.formula_weight / self.material.w) / H for m in self.LongMoment]
    
    def inv_Susceptibility(self, H: float, magnification: float = 1.0) -> list[float]:
        """1/χ=H/M list in '(emu/mol.Oe)^{-1}'.

        Args:
            H (float): Magnetic field (Oe).
            magnification (float): Amplification magnification. Defaults to 1.0.
        """
        return [magnification * 1/x for x in self.Susceptibility(H)]


Energychild = TypeVar('Energychild', bound='Energy')

class Energy:
    """from energy to other physical quantity or vice versa.

    Note:
        All units stand for SI.

    Args:
        quantity (float): Quantity part of a physical quantity.
        unit (str): Unit part of a physical quantity.
    
    Example:
        1 meV -> 11.6 K
        1 THz -> 4.14 meV
    """
    def __init__(self, quantity: float, unit: str) -> None:
        self.quantity: float = quantity
        self.unit: str = unit
        self.c: float = 299792458 # m/s
        self.kB: float = 1.380649e-23 # J/K
        self.e: float = 1.602176634e-19 # C
        self.h: float = 6.62607015e-34 # Js
        self.NA: float = 6.02214076e23 # mol^-1
        self.hbar: float = self.h/2/pi # Js
        self.me: float = 9.1093837015e-31 # kg
        self.muB: float = self.hbar * self.e / self.me / 2 # J/T

        # 数値の単位は [unit/meV]
        # eV, hν, hck, kBT, muBH
        self.mapping: dict[str, float] = {
            "meV": 1.,
            "THz":  (1e-3*self.e) / (self.h*1e12),
            "cm^-1": (1e-3*self.e) / (1e2*self.h*self.c),
            "A^-1": (1e-3*self.e) / (1e10*self.h*self.c),
            "K":    (1e-3*self.e) / self.kB,
            "T":    (1e-3*self.e) / (self.muB),
            "J":    (1e-3*self.e)
        }

    def __str__(self) -> str:
        return f"{self.quantity} {self.unit}"
    
    def __repr__(self) -> str:
        return f"{self.quantity} {self.unit}"

    def __add__(self: Energychild, other: Any) -> Energychild:
        new_quantity: float = self.quantity + other.quantity * self.mapping[self.unit] / self.mapping[other.unit]
        return self.__class__(new_quantity, self.unit)
    
    def __iadd__(self: Energychild, other: Any) -> Energychild:
        self.quantity += other.quantity * self.mapping[self.unit] / self.mapping[other.unit]
        return self
    
    def __mul__(self: Energychild, other: Any) -> Energychild:
        return self.__class__(self.quantity * other, self.unit)
    
    def __rmul__(self: Energychild, other: Any) -> Energychild:
        return self.__class__(self.quantity * other, self.unit)
    
    def __imul__(self: Energychild, other: Any) -> Energychild:
        self.quantity *= other
        return self

    def to(self: Energychild, new_unit: str) -> Energychild:
        """Translate the physical quantity as other units.

        Args:
            new_unit (str): New unit.

        Example:
            1 meV -> 11.6 K
            1 THz -> 4.14 meV
        """
        if new_unit in self.mapping:
            new_quantity: float = self.quantity * self.mapping[new_unit] / self.mapping[self.unit]
            return self.__class__(new_quantity, new_unit)
        else:
            raise ValueError("the argument 'new_unit' is invalid.")


def ingredient_flake_dp(A: list[int], W: int) -> None: # A: 適当に整数化したフレークの重さ, W: 目標重量
    """Choose optimal flakes whose total weight meets the target value.

    Note:
        The result will be output to stdout.

    Args:
        A (list[int]): list of weight of flakes, properly integerized.
        W (int): Target weight value.
    """
    N: int = len(A)
    K: int = W+20 # 余裕を持って求めておく
    dp: list[list[int]] = [[0]*K for i in range(N+1)]
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
        ans: list[int] = []
        if dp[N][now]:
            for i in range(N)[::-1]:
                if now-A[i]>=0 and dp[i][now-A[i]]:
                    now -= A[i]
                    ans.append(A[i])
        print(W+k, ans)
    return


class PhysicalConstant:
    """Physical constants.

    Attributes:
        pi: 3.14159265358979
        exp: 2.718281828459045
        NA: 6.02214076e+23 mol^-1
        kB: 1.380649e-23 J/K
        c: 299792458 m/s
        h: 6.62607015e-34 Js
        hbar: 1.0545718176461565e-34 Js
        e: 1.602176634e-19 C
        Patm: 101325 Pa
        Celsius: 273.15 K
        gn: 9.80665 ms^-2
        mu0: 1.2566370614359173e-06 NA^-2
        epsilon0: 8.854187817620389e-12 CV^-1m^-1
        me: 9.1093837015e-31 kg
        muB: 9.2740100783e-24 J/T
        muB_emu: 9.2740100783e-21 emu
        sigma: 5.670374419184428e-08 Wm-2K-4
        G: 6.6743e-11 Nm^2kg^-2
        alpha: 0.007297352565305217 
        phi0: 2.067833848461929e-15 Wb
        Rydberg: 10973731.556123963 m^-1
        a0: 5.291772111941794e-11 m
        R: 8.31446261815324 JK^-1mol^-1
    """
    def __init__(self) -> None:
        # 数学定数
        self._pi: float = float(np.pi)
        self._exp: float = float(np.e)

        # 定義値
        self._NA: float = 6.02214076 * 10**(23) # Avogadro定数 [mol^-1]
        self._kB: float = 1.380649 * 10**(-23) # Boltzmann定数 [J/K]
        self._c: float = 299792458 # 光速 [m/s]
        self._h: float = 6.62607015 * 10**(-34) # Planck定数 [Js]
        self._hbar: float = self._h / (2*self._pi) # Dirac定数 [Js]
        self._e: float = 1.602176634 * 10**(-19) # 電気素量 [C]
        self._Kcd: float = 683 # 発光効率 [lm/W]
        self._Patm: float = 101325 # 標準大気圧 [Pa]
        self._Celsius: float = 273.15 # Celsius温度ゼロ点 [K]
        self._gn: float = 9.80665 # 標準重力加速度 [ms^-2]

        self._mu0: float = 4 * self._pi * 10**(-7) # 真空の透磁率(旧定義) [NA^-2]
        self._epsilon0: float = 1 / (self._mu0 * self._c**2) # 真空の誘電率 [F/m]
        self._me: float = 9.1093837015 * 10**(-31) # 電子質量 [kg]
        self._muB: float = self._e * self._hbar / (2 * self._me) # Bohr磁子 [J/T]
        self._muB_emu: float = 9.2740100783 * 10**(-21) # Bohr磁子 [emu]
        self._sigma: float = self._pi**2 / 60 * self._kB**4 / self._hbar**3 / self._c**2 # Stefan-Boltzmann定数 [Wm^-2K^-4]
        self._G: float = 6.67430 * 10**(-11) # 万有引力定数 [Nm^2kg^-2]
        self._alpha: float = self._e**2 / (4 * self._pi * self._epsilon0 * self._hbar * self._c) # 微細構造定数 [dimensionless]
        self._phi0: float = self._h / (2 * self._e) # 磁束量子 [Wb]
        self._Rydberg: float = self._alpha**2 * self._me * self._c / (2 * self._h) # Rydberg定数 [m^-1]
        self._a0: float = self._alpha / (4 * self._pi * self._Rydberg) # Bohr半径 [m]
        self._R: float = self._NA * self._kB # 気体定数 [JK^-1mol^-1]
        self._Lorenz: float = self._pi**2 / 3 * (self._kB/self._e)**2 # Lorenz定数 [WΩ/K^2]

        self._unit: dict[str, str] = {
            "NA": "mol^-1",
            "kB": "J/K",
            "c": "m/s",
            "h": "Js",
            "hbar": "Js",
            "e": "C",
            "Patm": "Pa",
            "Celsius": "K",
            "gn": "ms^-2",

            "mu0": "NA^-2",
            "epsilon0": "CV^-1m^-1",
            "me": "kg",
            "muB": "J/T",
            "muB_emu": "emu",
            "sigma": "Wm^-2K^-4",
            "G": "Nm^2kg^-2",
            "alpha": "",
            "phi0": "Wb",
            "Rydberg": "m^-1",
            "a0": "m",
            "R": "JK^-1mol^-1",
            "Lorenz": "WOhm/K^2",
            
        }
    
    @property
    def pi(self) -> float: return self._pi

    @property
    def exp(self) -> float: return self._exp

    @property
    def NA(self) -> float: return self._NA

    @property
    def kB(self) -> float: return self._kB

    @property
    def c(self) -> float: return self._c

    @property
    def h(self) -> float: return self._h

    @property
    def hbar(self) -> float: return self._hbar

    @property
    def e(self) -> float: return self._e

    @property
    def Kcd(self) -> float: return self._Kcd

    @property
    def Patm(self) -> float: return self._Patm

    @property
    def Celsius(self) -> float: return self._Celsius

    @property
    def gn(self) -> float: return self._gn

    @property
    def mu0(self) -> float: return self._mu0

    @property
    def epsilon0(self) -> float: return self._epsilon0

    @property
    def me(self) -> float: return self._me

    @property
    def muB(self) -> float: return self._muB

    @property
    def muB_emu(self) -> float: return self._muB_emu

    @property
    def sigma(self) -> float: return self._sigma

    @property
    def G(self) -> float: return self._G

    @property
    def alpha(self) -> float: return self._alpha

    @property
    def phi0(self) -> float: return self._phi0

    @property
    def Rydberg(self) -> float: return self._Rydberg

    @property
    def a0(self) -> float: return self._a0

    @property
    def R(self) -> float: return self._R

    @property
    def unit(self) -> dict[str, str]: return self._unit

    def __str__(self) -> str:
        res: list[str] = []
        for k, v in self.__dict__.items():
            if k[1:] in self._unit:
                res.append(f"{k[1:]}: {v} {self._unit[k[1:]]}")
        return "\n".join(res)
        

def main() -> None:
    pass
    return

if __name__ == "__main__":
    main()

