"""`cpystal.analysis.spacegroup` is a module for analyzing space group of crystal.

Classes:
    `REF`
    `MatrixREF`
    `SymmetryOperation`
    `PhysicalPropertyTensorAnalyzer`

Functions:
    `spacegroup_to_pointgroup`
"""
from __future__ import annotations # class定義中に自己classを型ヒントとして使用するため

from typing import Any, DefaultDict, Dict, Iterable, List, Optional, overload, Set, Tuple, Union
from fractions import Fraction
from functools import reduce
from math import gcd
from collections import defaultdict
import re

#from ..core import Crystal


class REF: # 有理数体Qに√pを添加した単純拡大体Q[√p]のclass REF(Rational Extension Field)
    """Rational Extension Field.

    This class represents a algebraic simple Rational Extension Field with a root of a polynomial: x^2-p, 
    where p is an integer, and negative values are allowed. 
    Depending on the value of p, it may not be an extension field (e.g. p=4), 
    but it can be formally defined even in such cases.

    Mathematically, an instance of this class is equivalent to an element of the set described as follows:
        Q(√p)={a+b√p|a,b∈Q}
    where Q is the field of rational numbers.

    Attributes:
        p (int): Square of the generator of simple extension Q(√p).
        a (Fraction): Rational number. (definition: see above)
        b (Fraction): Rational number. (definition: see above)
    """
    def __init__(self, p: int, a: Fraction = Fraction(), b: Fraction = Fraction()) -> None:
        self.p: int = p # Q[√p]の生成元の2乗
        self.a: Fraction = a
        self.b: Fraction = b
    
    def __str__(self) -> str:
        return f"({str(self.a)}, {str(self.b)})"
    
    def __repr__(self) -> str:
        return f"({str(self.a)}, {str(self.b)})"

    def __neg__(self) -> REF:
        return self.__class__(self.p, -self.a, -self.b)
    
    def __eq__(self, other: Any) -> bool:
        if type(other) is REF and self.p == other.p and self.a == other.a and self.b == other.b:
            return True
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: REF) -> bool:
        if type(other) is REF:
            return self.to_float() < other.to_float()
        else:
            raise TypeError(f"'<' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __le__(self, other: REF) -> bool:
        if type(other) is REF:
            return self.to_float() <= other.to_float()
        else:
            raise TypeError(f"'<=' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __gt__(self, other: REF) -> bool:
        if type(other) is REF:
            return self.to_float() > other.to_float()
        else:
            raise TypeError(f"'>' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")

    def __ge__(self, other: REF) -> bool:
        if type(other) is REF:
            return self.to_float() >= other.to_float()
        else:
            raise TypeError(f"'>=' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")

    def __add__(self, other: Any) -> REF:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a+other, self.b)
        elif type(other) is REF:
            if self.p == other.p:
                return self.__class__(self.p, self.a+other.a, self.b+other.b)
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")

    def __radd__(self, other: Any) -> REF:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a+other, self.b)
        elif type(other) is REF:
            if self.p == other.p:
                return self.__class__(self.p, self.a+other.a, self.b+other.b)
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __iadd__(self, other: Any) -> REF:
        if type(other) is int or type(other) is Fraction:
            self.a += other
            return self
        elif type(other) is REF:
            if self.p == other.p:
                self.a += other.a
                self.b += other.b
                return self
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __sub__(self, other: Any) -> REF:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a-other, self.b)
        elif type(other) is REF:
            if self.p == other.p:
                return self.__class__(self.p, self.a-other.a, self.b-other.b)
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __rsub__(self, other: Any) -> REF:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a-other, self.b)
        elif type(other) is REF:
            if self.p == other.p:
                return self.__class__(self.p, self.a-other.a, self.b-other.b)
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __isub__(self, other: Any) -> REF:
        if type(other) is int or type(other) is Fraction:
            self.a -= other
            return self
        elif type(other) is REF:
            if self.p == other.p:
                self.a -= other.a
                self.b -= other.b
                return self
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")

    def __mul__(self, other: Any) -> REF:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a*other, self.b*other)
        elif type(other) is REF:
            if self.p == other.p:
                p: int = self.p
                a: Fraction = self.a
                b: Fraction = self.b
                c: Fraction = other.a
                d: Fraction = other.b
                return self.__class__(self.p, a*c + p*b*d, a*d + b*c)
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")

    def __rmul__(self, other: Any) -> REF:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a*other, self.b*other)
        elif type(other) is REF:
            if self.p == other.p:
                p: int = self.p
                a: Fraction = self.a
                b: Fraction = self.b
                c: Fraction = other.a
                d: Fraction = other.b
                return self.__class__(self.p, a*c + p*b*d, a*d + b*c)
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")

    def __imul__(self, other: Any) -> REF:
        if type(other) is int or type(other) is Fraction:
            self.a *= other
            self.b *= other
            return self
        elif type(other) is REF:
            if self.p == other.p:
                p: int = self.p
                a: Fraction = self.a
                b: Fraction = self.b
                c: Fraction = other.a
                d: Fraction = other.b
                self.a = a*c + p*b*d
                self.b = a*d + b*c
                return self
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __truediv__(self, other: Any) -> REF:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a/other, self.b/other)
        elif type(other) is REF:
            if self.p == other.p:
                p: int = self.p
                a: Fraction = self.a
                b: Fraction = self.b
                c: Fraction = other.a
                d: Fraction = other.b
                aa: Fraction = (a*c - p*b*d) / (c**2 - p * d**2)
                bb: Fraction = (-a*d + b*c) / (c**2 - p * d**2)
                return self.__class__(p, aa, bb)
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(self).__name__}' and '{type(other).__name__}'")

    def __rtruediv__(self, other: Any) -> REF:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a/other, self.b/other)
        elif type(other) is REF:
            if self.p == other.p:
                p: int = self.p
                a: Fraction = self.a
                b: Fraction = self.b
                c: Fraction = other.a
                d: Fraction = other.b
                aa: Fraction = (a*c - p*b*d) / (c**2 - p * d**2)
                bb: Fraction = (-a*d + b*c) / (c**2 - p * d**2)
                return self.__class__(p, aa, bb)
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(self).__name__}' and '{type(other).__name__}'")

    def __itruediv__(self, other: Any) -> REF:
        if type(other) is int or type(other) is Fraction:
            self.a /= other
            self.b /= other
            return self
        elif type(other) is REF:
            if self.p == other.p:
                p: int = self.p
                a: Fraction = self.a
                b: Fraction = self.b
                c: Fraction = other.a
                d: Fraction = other.b
                self.a = (a*c - p*b*d) / (c**2 - p * d**2)
                self.b = (-a*d + b*c) / (c**2 - p * d**2)
                return self
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(self).__name__}' and '{type(other).__name__}'")

    def __copy__(self) -> REF:
        return self
    
    def copy(self) -> REF:
        return self

    def __deepcopy__(self) -> REF:
        return self.__class__(self.p, self.a, self.b)

    def deepcopy(self) -> REF:
        return self.__class__(self.p, self.a, self.b)
    
    def zero(self) -> REF: # 加法単位元 乗法零元
        return self.__class__(self.p, Fraction(), Fraction())
    
    def inv(self) -> REF: # 乗法逆元
        p: int = self.p
        a: Fraction = self.a
        b: Fraction = self.b
        return self.__class__(p, a/(a**2 - p*b**2), -b/(a**2 - p*b**2))

    def swap(self) -> REF: # 有理部と無理部を入れ替えたREFを生成
        p: int = self.p
        a: Fraction = self.a
        b: Fraction = self.b
        return self.__class__(p, b, a)

    def conjugate(self) -> REF: # 一般化した共役(p=-1のとき複素共役に一致)
        p: int = self.p
        a: Fraction = self.a
        b: Fraction = self.b
        return self.__class__(p, a, -b)

    def to_float(self) -> float:
        p: int = self.p
        if p < 0:
            raise ValueError(f"negative `REF.p`: {self.p}; to get the value of this REF instance, use 'to_complex'")
        a: Fraction = self.a
        b: Fraction = self.b
        return a.numerator/a.denominator + b.numerator/b.denominator*(p**0.5) # float

    def to_complex(self) -> complex:
        p: int = self.p
        a: Fraction = self.a
        b: Fraction = self.b
        return complex(a.numerator/a.denominator + b.numerator/b.denominator*(p**0.5)) # complex

# 型エイリアス
Matrix = List[List[Any]]

class MatrixREF:
    """Matrix of `REF` instance.

    Attributes:
        p (int): Square of the generator of simple extension Q(√p).
        mat (List[List[REF]]): 2-dimension matrix of `REF` instance.
        shape (Tuple[int, int]): Shape of `MatrixREF.mat`. First element is the number of row, second is the number of column.
    """
    def __init__(self, p: int, mat: Optional[Matrix] = None): # p:int, mat:List[List[int/Fraction]]
        self.p: int = p
        self.mat: List[List[REF]] = []
        self.shape: Tuple[int, int]
        if mat is None:
            self.shape = (0, 0)         
        else:
            self.shape = (len(mat), len(mat[0]))
            for row in mat:
                now: List[REF] = []
                for r in row:
                    if type(r) is int or type(r) is Fraction:
                        now.append(REF(p, Fraction(r)))
                    elif type(r) is REF:
                        if self.p == r.p:
                            now.append(r.deepcopy())
                        else:
                            raise TypeError(f"REF generator is not same")
                    else:
                        raise TypeError(f"type of matrix components must be int/Fraction/REF")
                self.mat.append(now)

    def __str__(self) -> str:
        res: str = ',\n '.join([str(m) for m in self.mat])
        return f"[{res}]"

    def __repr__(self) -> str:
        res: str = ',\n '.join([str(m) for m in self.mat])
        return f"[{res}]"
    
    def __len__(self) -> int:
        return len(self.mat)
    
    def __eq__(self, other: Any) -> List[List[bool]]: # type: ignore
        m: int
        n: int
        m, n = self.shape
        smat: List[List[REF]] 
        omat: List[List[REF]]
        res: List[List[bool]]
        if type(other) is int or type(other) is Fraction:
            other_: REF = REF(self.p, other)
            smat = self.mat
            res = [[smat[i][j]==other_ for j in range(n)] for i in range(m)]
            return res # List[List[bool]]

        elif type(other) is MatrixREF:
            if self.p == other.p:
                smat = self.mat
                omat = other.mat
                res = [[smat[i][j]==omat[i][j] for j in range(n)] for i in range(m)]
                return res # List[List[bool]]
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"incompatible type(s) for : '{type(self).__name__}' and '{type(other).__name__}'")
        
    def __ne__(self, other: Any) -> bool:
        if type(self).__name__ != type(other).__name__:
            return True
        if self.shape != other.shape:
            return True
        m: int
        n: int
        m, n = self.shape
        for i in range(m):
            for j in range(n):
                if self.mat[i][j] != other.mat[i][j]:
                    return True
        return False

    def __neg__(self) -> MatrixREF:
        m: int
        n: int
        m, n = self.shape
        res: List[List[REF]] = [[REF(self.p)]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                res[i][j] = -self.mat[i][j]
        ret: MatrixREF = self.__class__(self.p, mat=None)
        ret.mat = res
        ret.shape = (m, n)
        return ret

    def __matmul__(self, other: MatrixREF) -> MatrixREF:
        if type(other) is MatrixREF:
            if self.p == other.p:
                if self.shape[1] == other.shape[0]:
                    # (l,m)*(m,n) -> (l,n)
                    l: int = self.shape[0]
                    m: int = self.shape[1]
                    n: int = other.shape[1]
                    smat: List[List[REF]] = self.mat
                    omat: List[List[REF]] = other.mat
                    res: List[List[REF]] = [[REF(self.p)]*n for _ in range(l)]
                    for i in range(l):
                        for k in range(n):
                            val: REF = REF(self.p)
                            for j in range(m):
                                val += smat[i][j]*omat[j][k]
                            res[i][k] = val
                    ret: MatrixREF = self.__class__(self.p, mat=None)
                    ret.mat = res
                    ret.shape = (l, n)
                    return ret # ret: MatrixREF
                else:
                    raise TypeError(f"matrix shape is inappropriate")
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for @: '{type(self).__name__}' and '{type(other).__name__}'")
    
    @overload
    def __getitem__(self, key: int) -> List[REF]:
        ...
    @overload
    def __getitem__(self, key: slice) -> Iterable[List[REF]]:
        ...
    def __getitem__(self, key: Any) -> Any:
        return self.mat[key]
    
    @overload
    def __setitem__(self, key: int, value: List[REF]) -> None:
        ...
    @overload
    def __setitem__(self, key: slice, value: Iterable[List[REF]]) -> None:
        ...
    def __setitem__(self, key, value):
        if type(value) is int or type(value) is Fraction:
            self.mat[key] = REF(self.p, value)
        elif type(value) is REF:
            if self.p == value.p:
                self.mat[key] = value
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"invalid type for an element of MatrixREF: {type(value)}")

    def __copy__(self) -> MatrixREF:
        return self
    
    def copy(self) -> MatrixREF:
        return self

    def __deepcopy__(self) -> MatrixREF:
        ret: MatrixREF = self.__class__(self.p)
        ret.mat = [[self.mat[i][j].deepcopy() for j in range(self.shape[1])] for i in range(self.shape[0])]
        ret.shape = self.shape
        return ret

    def deepcopy(self) -> MatrixREF:
        ret: MatrixREF = self.__class__(self.p)
        ret.mat = [[self.mat[i][j].deepcopy() for j in range(self.shape[1])] for i in range(self.shape[0])]
        ret.shape = self.shape
        return ret

    def identity(self, shape: Optional[Tuple[int, int]] = None) -> MatrixREF:
        # m×n単位行列
        m: int
        n: int
        if shape is None:
            m, n = self.shape
        else:
            m, n = shape
        p: int = self.p
        res: List[List[REF]] = [[REF(p)]*n for _ in range(m)]
        for i in range(min(m,n)):
            res[i][i] = REF(p, Fraction(1,1), Fraction())
        ret: MatrixREF = self.__class__(p)
        ret.shape = (m,n)
        ret.mat = res
        return ret # ret: MatrixREF
    
    def sum(self, axis: Optional[int] = None) -> Union[REF, List[REF]]:
        m: int
        n: int
        m, n = self.shape
        smat = self.mat
        res: Union[REF, List[REF]]
        if axis is None:
            res = REF(self.p)
            for i in range(m):
                for j in range(n):
                    res += smat[i][j]
            return res # REF
        else:
            if axis == 0 or axis == -2:
                res = [REF(self.p)]*n
                for i in range(m):
                    for j in range(n):
                        res[j] += smat[i][j]
                return res # List[REF]
            elif axis == 1 or axis == -1:
                res = [REF(self.p)]*m
                for i in range(m):
                    for j in range(n):
                        res[i] += smat[i][j]
                return res # List[REF]
            else:
                raise KeyError(f"axis {axis} is out of bounds for MatrixREF of dimension {len(self.shape)}")


class _UnionFind:
    def __init__(self, n: int): # O(n)
        self.parent: List[int] = [-1]*n
        self.n: int = n
    def root(self, x: int) -> int: # 要素xの根の番号を返す O(α(n))
        if self.parent[x] < 0:
            return x
        else:
            self.parent[x] = self.root(self.parent[x])
            return self.parent[x]
    def size(self, x: int) -> int: # 要素xの所属するグループの要素数を調べる O(α(n))
        return -self.parent[self.root(x)]
    def merge(self, x: int, y: int) -> bool: # xとyを結合する O(α(n))
        x = self.root(x)
        y = self.root(y)
        if x == y:
            return False 
        if self.parent[x] > self.parent[y]: # 大きい方(x)に小さい方(y)をぶら下げる
            x, y = y, x 
        self.parent[x] += self.parent[y]
        self.parent[y] = x
        return True 
    def issame(self, x: int, y: int) -> bool: # xとyが同じグループにあるならTrue O(α(n))
        return self.root(x) == self.root(y)
    def family(self, x:int) -> List[int]: # xが属する連結成分を返す O(n)
        return [i for i in range(self.n) if self.issame(i,x)]
    def maximum(self) -> List[int]: # 最大連結成分を返す O(n)
        return self.family(self.parent.index(min(self.parent)))
    def all_root(self) -> List[int]: # 全ての根の番号を返す O(n)
        return [i for i in range(self.n) if self.parent[i] < 0]
    def decompose(self) -> List[List[int]]: # 連結成分分解を返す O(nα(n))
        return [self.family(i) for i in self.all_root()]


class SymmetryOperation(MatrixREF):
    """Symmetry opration represented as matrix.

    This class is inherited from `MatrixREF`.

    Attributes:
        p (int): Square of the generator of simple extension Q(√p).
        mat (List[List[REF]]): 2-dimension matrix of `REF` instance.
        shape (Tuple[int, int]): Shape of `MatrixREF.mat`. First element is the number of row, second is the number of column.
        mirrority (bool): True if the symmetry opration changes right-handed system to left-handed system or vice versa.
    """
    def __init__(self, p: int, mat: Matrix, mirrority: bool = False):
        super().__init__(p, mat=mat)
        self.mirrority: bool = mirrority


def spacegroup_to_pointgroup(name: str) -> str:
    """Convert space group name to point group name.

    1. Capital alphabets and spaces are removed.
    2. Subscripts (represented by "_" with numbers) are removed.
    3. All "1" except a component of monoclinic symbols ("1" or "-1") are removed.
    4. "a","b","c","d", and "n" are converted to "m"
    5. Exception handling.

    Args:
        name (str): Space group name in International (Hermann-Mauguin) notation.
    
    Returns:
        (str): Point group name in International (Hermann-Mauguin) notation converted from the space group name.
    
    Examples:
        >>> point_group_name = spacegroup_to_pointgroup("Fd-3m") # Diamond
        >>> print(point_group_name)
            m-3m
    """
    name = re.sub(r"[A-Z]|\name", "", name)
    name = re.sub(r"_\d|_\{\d\}", "", name)
    name = re.sub(r"([^-])1", "\\1", name)
    name = re.sub(r"[a-dn]", "m", name)
    if name == "-4m2":
        name = "-42m"
    return name

Relation = List[Tuple[REF, int]]
Relations = List[List[Tuple[REF, int]]]
Relation_ternary = List[Tuple[REF, Tuple[int, ...]]]
Relations_ternary = List[List[Tuple[REF, Tuple[int, ...]]]]

class PhysicalPropertyTensorAnalyzer:
    """Analyze non-zero elements of physical property tensors based on the symmetry of crystallographic point group.

    All symmetry operations of crystallographic point groups can be represented as a 3×3 matrix on a simple rationnal extension field: 'M_{3×3}(Q[√3])' in an appropriate orthogonal basis.
    Therefore, it is possible to determine which elements are equivalent or zero by straightforward exact calculation.

    Attributes:
        point_group_name (str): Target point group name written in Schönflies notation.
        unitary_matrice (List[MatrixREF]): List of the symmetry operations of the crystallographic point group represented as a matrix in an appropriate orthogonal basis.

    Todo:
        To implement the converter from spacegroup name to point group name.
    """
    # 適切に定めた直交座標系を基底にとったときの対称操作の行列表示
    # 基本並進ベクトルたちが直交系をなすならそれを使う
    C1 = SymmetryOperation(3, [[1,0,0],
                                [0,1,0],
                                [0,0,1]])

    C2_100 = SymmetryOperation(3, [[1,0,0],
                                    [0,-1,0],
                                    [0,0,-1]])
    C2_010 = SymmetryOperation(3, [[-1,0,0],
                                    [0,1,0],
                                    [0,0,-1]])
    C2_001 = SymmetryOperation(3, [[-1,0,0],
                                    [0,-1,0],
                                    [0,0,1]])

    C2_n101 = SymmetryOperation(3, [[0,0,-1],
                                   [0,-1,0],
                                   [-1,0,0]])
    C2_1n10 = SymmetryOperation(3, [[0,-1,0],
                                    [-1,0,0],
                                    [0,0,-1]])
    C2_01n1 = SymmetryOperation(3, [[-1,0,0],
                                    [0,0,-1],
                                    [0,-1,0]])

    C2_110 = SymmetryOperation(3, [[0,1,0],
                                    [1,0,0],
                                    [0,0,-1]])

    C3_111 = SymmetryOperation(3, [[0,0,1],
                                    [1,0,0],
                                    [0,1,0]])
    C3_1n1n1 = SymmetryOperation(3, [[0,0,-1],
                                    [-1,0,0],
                                    [0,1,0]])
    C3_n11n1 = SymmetryOperation(3, [[0,0,1],
                                    [-1,0,0],
                                    [0,-1,0]])
    C3_n1n11 = SymmetryOperation(3, [[0,1,0],
                                    [0,0,-1],
                                    [-1,0,0]])

    C3_001 = SymmetryOperation(3, [[0,-1,0],
                                    [1,-1,0],
                                    [0,0,1]])
    C3_001 = SymmetryOperation(3,  [[REF(3,Fraction(-1,2)),REF(3,b=Fraction(1,2)),0],
                                    [REF(3,b=Fraction(-1,2)),REF(3,Fraction(-1,2)),0],
                                    [0,0,1]])

    C4_001 = SymmetryOperation(3,  [[0,-1,0],
                                    [1,0,0],
                                    [0,0,1]])
    C4_010 = SymmetryOperation(3, [[0,0,1],
                                    [0,1,0],
                                    [-1,0,0]])
    C4_100 = SymmetryOperation(3, [[1,0,0],
                                  [0,0,-1],
                                  [0,1,0]])


    m_100 = SymmetryOperation(3, [[-1,0,0],
                                    [0,1,0],
                                    [0,0,1]], mirrority=True)
    m_010 = SymmetryOperation(3, [[1,0,0],
                                    [0,-1,0],
                                    [0,0,1]], mirrority=True)
    m_001 = SymmetryOperation(3, [[1,0,0],
                                    [0,1,0],
                                    [0,0,-1]], mirrority=True)

    m_110 = SymmetryOperation(3, [[0,-1,0],
                                    [-1,0,0],
                                    [0,0,1]], mirrority=True)
    m_1n10 = SymmetryOperation(3, [[0,1,0],
                                    [1,0,0],
                                    [0,0,1]], mirrority=True)
    m_n101 = SymmetryOperation(3, [[0,0,1],
                                    [0,1,0],
                                    [1,0,0]], mirrority=True)
    m_01n1 = SymmetryOperation(3, [[1,0,0],
                                    [0,0,1],
                                    [0,1,0]], mirrority=True)

    inversion = SymmetryOperation(3, [[-1,0,0],
                                        [0,-1,0],
                                        [0,0,-1]], mirrority=True)

    # 国際表記 -> Schönflies表記
    international_to_schoenflies_notation: Dict[str, str] = {
        # 立方晶(cubic system)
        "m-3m": "Oh",
        "-43m": "Td",
        "432": "O",
        "m-3":"Th",
        "23": "T",
        # 正方晶(tetragonal system)
        "4/mmm": "D4h",
        "-42m": "D2d",
        "4mm" :"C4v",
        "422" :"D4",
        "4/m": "C4h",
        "-4": "S4",
        "4": "C4",
        # 直方晶(orthorhombic system)
        "mmm": "D2h",
        "mm2": "C2v",
        "222": "D2",
        # 六方晶(hexagonal system)
        "6/mmm": "D6h",
        "-62m": "D3h",
        "6mm": "C6v",
        "622": "D6",
        "6/m": "C6h",
        "-6": "C3h",
        "6": "C6",
        # 三方晶(trigonal system)
        "-3m": "D3d",
        "3m": "C3v",
        "32": "D3",
        "-3": "S6",
        "3": "C3",
        # 単斜晶(monoclinic system)
        "2/m": "C2h",
        "m": "S1",
        "2": "C2",
        # 三斜晶(triclinic system)
        "-1": "S2",
        "1": "C1"
    }


    # 参照元 https://www.cryst.ehu.es/cryst/get_point_genpos.html (三方晶やC3_001の行列は改変)
    # 各結晶点群の生成元(適切に定めた直交座標系を基底にとっていることに注意)
    # 三方晶では
    # (default):   [111]方向をz軸，c軸とz軸とy軸がx=0上になるようにとる
    # _rombohedral: [111]方向をxyzでの(1,1,1)方向，c軸とz軸と[111]がx=y上になるようにとる
    PointGroup_generators: Dict[str, List[SymmetryOperation]] = {
        # 立方晶
        "Oh": [C2_001, C2_010, C3_111, C2_110, inversion],
        "Td": [C2_001, C2_010, C3_111, m_1n10],
        "O":  [C2_001, C2_010, C3_111, C2_110],
        "Th": [C2_001, C2_010, C3_111, inversion],
        "T":  [C2_001, C2_010, C3_111],
        # 正方晶
        "D4h":[C2_001, C4_001, C2_010, inversion],
        "D2d":[C2_001, -C4_001, C2_010], # type: ignore # problem: type hint in inherited class
        "C4v":[C2_001, C4_001, m_010],
        "D4": [C2_001, C4_001, C2_010],
        "C4h":[C2_001, C4_001, inversion],
        "S4": [C2_001, -C4_001], # type: ignore # problem: type hint in inherited class
        "C4": [C2_001, C4_001],
        # 直方晶
        "D2h":[C2_001, C2_010, inversion],
        "C2v":[C2_001, m_010],
        "D2": [C2_001, C2_010],
        # 六方晶
        "D6h":[C3_001, C2_001, C2_110, inversion],
        "D3h":[C3_001, m_001, m_110],
        "C6v":[C3_001, C2_001, m_110],
        "D6": [C3_001, C2_001, C2_110],
        "C6h":[C3_001, C2_001, inversion],
        "C3h":[C3_001, m_001],
        "C6": [C3_001, C2_001],
        # 三方晶
        "D3d":[C3_001, C2_100, inversion],
        "C3v":[C3_001, m_100],
        "D3": [C3_001, C2_100],
        "S6": [C3_001, inversion],
        "C3": [C3_001],
        #"D3d_rhombohedral": [C3_111, C2_1n10, inversion],
        #"C3v_rhombohedral": [C3_111, m_1n10],
        #"D3_rhombohedral":  [C3_111, C2_1n10],
        #"S6_rhombohedral":  [C3_111, inversion],
        #"C3_rhombohedral":  [C3_111],
        # 単斜晶
        "C2h":[C2_001, inversion],
        "S1": [m_001],
        "C2": [C2_001],
        # 三斜晶
        "S2": [inversion],
        "C1": [C1],
    }

    def __init__(self, point_group_name: str) -> None:
        self.point_group_name: str = point_group_name
        pgname_in_schoenflies_notation: str = self.international_to_schoenflies_notation[self.point_group_name]
        self.unitary_matrice: List[SymmetryOperation] = self.PointGroup_generators[pgname_in_schoenflies_notation]

    @classmethod
    def _Gauss_Elimination_REF(cls, A: MatrixREF) -> MatrixREF: # 掃き出し法による行簡約化
        B: MatrixREF = A.deepcopy()
        m: int
        n: int
        m, n = B.shape # m×n行列
        zero: REF = REF(B.p)
        for k in range(n):
            if k == m:
                break
            flag: bool = True
            for i in range(k,m):
                if B[i][k] != zero:
                    flag = False
                    break
            if flag: # k列は簡約済み
                continue
            if i != k: # i行とk行の入れ替え
                for j in range(k,n):
                    B[i][j], B[k][j] = B[k][j], B[i][j]
            w: REF = B[k][k].inv()
            for j in range(k,n): # k行全体をB[k][k]で割って先頭を1にする
                B[k][j] *= w
            for i in range(m):
                if i == k:
                    continue
                v: REF = B[i][k].deepcopy()
                for j in range(k,n): # i行からk行*B[i][k]を引く
                    B[i][j] -= v*B[k][j]
        return B

    @classmethod
    def _ternary(cls, n: int, fill: int) -> Tuple[int, ...]: # fill: 桁数
        res: List[int] = [] # res[i] := nの3進展開の"下から"i桁目
        r: int
        while n:
            n, r = divmod(n,3)
            res.append(r)
        return tuple(res + [0]*max(0,fill-len(res))) # Tuple[int, ...]

    @classmethod
    def _tensor_elem_relations(cls, N: int, R: SymmetryOperation, axiality: bool) -> Relations:
        # N階のテンソルに直交変換Rを施したときの，要素間の関係式(つまりテンソルに関わる物理量は全て共変と仮定)
        # relations[i]: \sum_{j} relations[i][j][0]*A_{relations[i][j][1]} = 0 という関係式を表す
        # N=2: A_{ij}  = R_{il}R{jm} A_{lm}
        # N=3: A_{ijk} = R_{il}R{jm}R_{kn} A_{lmn} 
        # などを元に計算
        M: int = 3**N
        zero: REF = REF(3) # 有理拡大体の零元
        relations: Relations = []
        for ijk in range(M):
            now: Relation = []
            val_ijk: REF = REF(3, Fraction(1,1))
            for lmn in range(M): # 添字について和をとる
                val: REF = REF(3, Fraction(1,1))
                for a, b in zip(cls._ternary(ijk,N), cls._ternary(lmn,N)):
                    val *= R[a][b]
                if val != zero: # 非ゼロの項のみ記録
                    if ijk == lmn:
                        val_ijk -= val
                        continue
                    if axiality and R.mirrority:
                        # 軸性テンソルに対する右手系-左手系変換(鏡映・反転)では符号反転
                        now.append((val, lmn))
                    else:
                        # 極性テンソルに対する任意の変換，軸性テンソルに対する回転
                        now.append((-val, lmn))
            if val_ijk != zero:
                now.append((val_ijk, ijk))
            if now:
                relations.append(now)
        return relations # List[List[Tuple[REF,int]]]

    @classmethod
    def _symmetry(cls, N: int, expr: str) -> Relations: # 対称性から同値になるテンソル要素たちの関係式を生成
        # "ijk = jki = kij" のように，添字で物理的な制約に基づくテンソルの対称性を導入する
        # 複数の条件がある場合はカンマで区切る
        M: int = 3**N
        expressions: List[List[str]] = [s.split("=") for s in re.sub(r"[\u3000 \t]", "", expr).split(",")]
        expressions_data : List[Tuple[List[str], str, DefaultDict[str,List[int]], str, DefaultDict[str,List[int]]]] = []
        relations: Relations = []
        for expression in expressions:
            for i in range(1,len(expression)):
                if len(expression[0]) != N or len(expression[i]) != N:
                    raise ValueError(f"expressions must be all the same length: {N}")
                characters: List[str] = list(set(list(expression[0]+expression[i])))
                d0: DefaultDict[str, List[int]] = defaultdict(list) # 添字表現=expressionにおいて，文字cが出現するindexのリスト
                d1: DefaultDict[str, List[int]] = defaultdict(list)
                for j in range(N):
                    d0[expression[0][j]].append(j)
                    d1[expression[i][j]].append(j)
                expressions_data.append((characters, expression[0], d0, expression[i], d1)) # この関係式で出現する文字全体の集合も持っておく
        for ijk in range(M):
            for lmn in range(ijk,M):
                if ijk == lmn:
                    continue
                ijk_ternary: Tuple[int, ...] = cls._ternary(ijk, N)
                lmn_ternary: Tuple[int, ...] = cls._ternary(lmn, N)
                for characters, s0, d0, s1, d1 in expressions_data:
                    flag: bool = True
                    for c in characters: # 表現の文字ごとに見る
                        numbers: Set[int] = set()
                        for i in d0[c]:
                            numbers.add(ijk_ternary[i])
                        for l in d1[c]:
                            numbers.add(lmn_ternary[l])
                        if len(numbers) != 1: # 同じ文字に対応する添字の数字は1種類でないといけない
                            flag = False
                            break
                    if flag: # 全ての文字について条件を満たすならそれらのテンソル要素は同値
                        relations.append([(REF(3, Fraction(1,1)), ijk), (REF(3, Fraction(-1,1)), lmn)])
                        break
        return relations # List[List[Tuple[REF, int]]]

    @classmethod
    def _relation_to_ternary(cls, relation: Relation, N: int) -> Relation_ternary: # relationの添字を3進数に変換(printでの出力用)
        return [(val, cls._ternary(ijk,N)) for val,ijk in relation]

    @classmethod
    def _delete_invalid(cls, relations: Relations, nonzero: Set[int]) -> Tuple[Relations, Set[int], bool]: # 無効な式を削除，0の要素を除外
        renewed: Relations = []
        flag: bool = False
        for relation in relations:
            if len(relation) == 1:
                flag = True
                ijk: int = relation[0][1]
                if ijk in nonzero:
                    nonzero.discard(ijk) # 0の要素を除外
                continue # 無効な式を削除
            else:
                now: Relation = []
                for val, ijk in relation:
                    if not ijk in nonzero: # 式中の0の要素を削除
                        flag = True
                        continue
                    now.append((val,ijk))
                if now:
                    renewed.append(now)
        return renewed, nonzero, flag # Tuple[List[List[Tuple[REF,int]]], Set[int], bool]

    @classmethod
    def _untangle_relations(cls, M: int, relations: Relations, nonzero: Set[int]) -> Tuple[Relations, Set[int], bool]: # 係数行列簡約化で関係式を簡約化
        flag: bool = False
        zero: REF = REF(3)
        U: _UnionFind = _UnionFind(M)
        renewed: Relations = []
        for relation in relations:
            for i in range(len(relation)):
                for j in range(i,len(relation)):
                    if i == j:
                        continue
                    if relation[i][1] in nonzero and relation[j][1] in nonzero:
                        U.merge(relation[i][1], relation[j][1])
                    
        for family in U.decompose(): # 関係式で結びついているテンソル要素たちごとに処理
            if len(family) == 1:
                continue
            family_inv: Dict[int, int] = {a:i for i, a in enumerate(family)}
            A: List[List[REF]] = [] # 係数行列
            for relation in relations:
                if U.issame(relation[0][1], family[0]):
                    a = [zero]*len(family)
                    for val, ijk in relation:
                        if ijk in nonzero:
                            a[family_inv[ijk]] = val
                    A.append(a)

            AA: MatrixREF = cls._Gauss_Elimination_REF(MatrixREF(3, A)) # Gaussの消去法(掃き出し法)で行簡約
            # 簡約化して行の非ゼロ要素が1つだけならそれに対応するテンソル要素の値は0
            # それ以外の行は前よりも簡単になった関係式になっている
            zero_elements: List[int] = []
            m: int
            n: int
            m, n = AA.shape
            for i in range(m):
                idxs: List[int] = []
                for j in range(n):
                    if AA[i][j] != zero:
                        idxs.append(j)
                if len(idxs) == 1: # 行の非ゼロ要素が1つだけなのでこれに対応するテンソル要素はゼロ
                    zero_elements.append(family[idxs[0]])
                else: # 新しい関係式
                    renewed.append([(AA[i][j], family[j]) for j in idxs])
    
            for z in zero_elements:
                if z in nonzero:
                    flag = True
                    nonzero.discard(z)
        return renewed, nonzero, flag # Tuple[List[List[Tuple[REF,int]]], Set[int], bool]

    @classmethod
    def _remove_duplicate(cls, relations: Relations, nonzero: Set[int]) -> Tuple[Relations, Set[int], bool]: # 複数の等価な式を一本化
        flag: bool = False
        renewed: Relations = []
        for idx1 in range(len(relations)):
            d1: DefaultDict[int, REF] = defaultdict(lambda: REF(3))
            for val, ijk in relations[idx1]:
                d1[ijk] += val
            flag2: bool = True # relations[idx1]と等価なものが存在するならFalse
            for idx2 in range(idx1,len(relations)):
                if idx1 == idx2:
                    continue
                d2: DefaultDict[int, REF] = defaultdict(lambda: REF(3)) # (関係式)=0において，関係式の各テンソル要素の係数を計算
                d2_inv: DefaultDict[int, REF] = defaultdict(lambda: REF(3)) # 両辺に-1をかけた関係式
                for val, ijk in relations[idx2]:
                    d2[ijk] += val
                    d2_inv[ijk] -= val

                if d1 == d2 or d1 == d2_inv:
                    flag2 = False
                    break
            if flag2: # 他に等価な式がないならOK
                renewed.append(relations[idx1])
            else:
                flag = True
        return renewed, nonzero, flag # Tuple[List[List[Tuple[REF,int]]], Set[int], bool]

    @classmethod
    def _simplify_coefficient(cls, R: List[REF]) -> List[REF]: # O(len(A))
        # Rの要素たちの比を保ったままREFの有理部と無理部の係数を整数に簡約化
        def lcm(a: int, b: int) -> int:
            return a * b // gcd(a, b) # int 
        L: List[int] = []
        G: List[int] = []
        z: Fraction = Fraction()
        flag: bool = False # 有理部の係数が全て0ならFalse
        for r in R:
            a: Fraction
            b: Fraction
            a, b = r.a, r.b
            if a != z: # 有理部が非ゼロなら分母と分子をそれぞれ追加
                flag = True
                L.append(a.denominator)
                G.append(a.numerator)
            if b != z: # 無理部が非ゼロなら分母と分子をそれぞれ追加
                L.append(b.denominator)
                G.append(b.numerator)
        l: int = reduce(lcm, L)
        g: int = reduce(gcd, G)
        f: Fraction = Fraction(l, g)
        res: List[REF]
        if flag:
            res = [r*f for r in R]
        else: # 有理部と無理部の入れ替え
            res = [r.swap()*f for r in R]
        return res # List[REF]

    @classmethod
    def _simplify_relations_value(cls, relations: Relations) -> Relations:
        # relation: List[Tuple[REF,int]] (in relations)の比を保ったままREFの有理部と無理部の係数を整数に簡約化
        res: Relations = []
        for relation in relations:
            if len(relation) == 0:
                continue
            vals: List[REF] = []
            ijks: List[int] = []
            for val, ijk in relation:
                vals.append(val)
                ijks.append(ijk)
            res.append(list(zip(cls._simplify_coefficient(vals), ijks))) # REFの有理部と無理部の簡約化
        return res # List[List[Tuple[REF, int]]]

    @classmethod
    def _extract_independent(cls, N: int, relations: Relations, nonzero: Set[int]) -> Relations: # 独立な成分と従属成分に分離
        if len(relations) == 0:
            return relations
        indep: Set[int] = set()
        dep: Set[int] = set()
        relations = sorted([sorted(relation, key=lambda x:cls._ternary(x[1], N)) for relation in relations], key=lambda x:x[0][1])
        for relation in relations:
            family: List[int] = [ijk for _, ijk in relation]
            undetermined: int = 0
            for i, ijk in enumerate(family):
                if (not ijk in indep) and (not ijk in dep):
                    undetermined += 1
            if undetermined == 0:
                continue
            for i, ijk in enumerate(family):
                if ijk in indep or ijk in dep:
                    continue
                if undetermined == 1: # 関係式で最後に残ったものは従属成分
                    dep.add(ijk)
                    break
                indep.add(ijk) # 独立成分として扱う
                undetermined -= 1
        
        # 行列の列を入れ替えて，左側にdepを，右側にindepを寄せる
        # 掃き出せばdepをindepたちだけで表現できる
        active: List[int] = sorted(list(nonzero), key=lambda x:cls._ternary(x, N))
        l: List[int] = []
        r: List[int] = []
        for ijk in active:
            if ijk in dep:
                l.append(ijk)
            elif ijk in indep:
                r.append(ijk)
            else:
                indep.add(ijk)
                r.append(ijk)
        active = l + r
        active_inv: Dict[int, int] = {a:i for i, a in enumerate(active)}
        
        A: List[List[REF]] = [] # 係数行列
        for relation in relations:
            a = [REF(3)]*len(active)
            for val, ijk in relation:
                a[active_inv[ijk]] = val
            A.append(a)
        AA: MatrixREF = cls._Gauss_Elimination_REF(MatrixREF(3, A)) # 掃き出し法で行簡約
        
        dep_represented_by_indep: Relations = [] # 従属成分たちのみを対象とした関係式(基本的に引数のrelationsと等価)
        for i in range(len(AA)):
            idxs: List[int] = []
            for j in range(len(active)):
                if AA[i][j] != REF(3):
                    idxs.append(j)
            dep_represented_by_indep.append([(AA[i][j], active[j]) for j in idxs])
        
        def represent(N: int, relation: Relation) -> str: # 独立成分のみで従属成分を表示
            r: REF
            ijk0: int
            r, ijk0 = relation[0]
            res: List[str] = []
            for val, ijk in relation[1:]:
                v: REF = -val/r
                if v.b == 0:
                    if v.a == 1:
                        res.append(str(cls._ternary(ijk, N)))
                    elif v.a == -1:
                        res.append(f"-{cls._ternary(ijk, N)}")
                    else:
                        res.append(str(v.a)+str(cls._ternary(ijk, N)))
                elif v.a == 0:
                    if v.b.denominator == 1:
                        if v.b == 1:
                            res.append(f"√{v.p}"+str(cls._ternary(ijk, N)))
                        else:
                            res.append(f"{v.b}√{v.p}"+str(cls._ternary(ijk, N)))
                    else:
                        if v.b.numerator == 1:
                            res.append(f"√{v.p}/{v.b.denominator}"+str(cls._ternary(ijk, N)))
                        else:
                            res.append(f"{v.b.numerator}√{v.p}/{v.b.denominator}"+str(cls._ternary(ijk, N)))
                else:    
                    res.append(f"({v.a}+{v.b}√{v.p})"+str(cls._ternary(ijk, N)))
            return f"{cls._ternary(ijk0, N)} = " + " + ".join(res)

        print(f"number of independent elements: {len(indep)}")
        print(f"--independent elements--")
        print(*sorted([cls._ternary(ijk, N) for ijk in indep]), sep="\n")
        print(f"------------------------")
        print()
        print(f"--dependent elements represented by indp.--")
        print(*[represent(N, relation) for relation in dep_represented_by_indep], sep="\n")
        print(f"-------------------------------------------")

        return relations # List[List[Tuple[REF,int]]]

    def get_elements_info(self, rank: int, axiality: bool, expr: Optional[str] = None) -> List[Tuple[int, ...]]: # N階の極性テンソルがR∈self.unitary_matriceの対称操作で不変となるときの非ゼロになりうる要素の添字を計算
        """Determine which elements are equivalent or zero by straightforward exact calculation based on Neumann's principle.

        Note:
            All analysis results are output to stdout.

        Args:
            rank (int): Rank of target physical property tensor.
            axiality (bool): True if the tensor is an axial tensor.
            expr (Optional[str]): String representing a relation between elements that is already known.

        Returns: 
            (List[Tuple[int, ...]]): Indice(0-indexed) of non-zero elements of the tensor.

        Examples:
            >>> point_group_name = spacegroup_to_pointgroup("Fd-3m") # Diamond
            >>> assert point_group_name == "m-3m"
            >>> PPTA = PhysicalPropertyTensorAnalyzer(point_group_name)
            >>> PPTA.get_elements_info(rank=4, axiality=False, expr="ijkl=ijlk=jikl=klij") # elastic modulus tensor (4-tensor): 4階の弾性率テンソル
            >>> PPTA.get_elements_info(rank=3, axiality=False, expr="ijk=ikj") # Optical Parametric Oscillator: 光パラメトリック発振

        """
        M: int = 3**rank
        nonzero: Set[int] = set(range(M))
        relations: Relations = []
        if expr is not None: # 添字で直接表現したテンソル要素間の対称性を元に関係式を構築
            relations.extend(self._symmetry(rank, expr))
        for R in self.unitary_matrice: # (結晶点群に属する)直交変換を元にテンソル要素間の関係式を構築
            relations.extend(self._tensor_elem_relations(rank, R, axiality))
        
        flag: bool = True
        while flag:
            flag1: bool = True
            flag2: bool = True
            flag3: bool = True
            while flag1:
                relations, nonzero, flag1 = self._delete_invalid(relations,nonzero) # 無効な式を削除
            relations, nonzero, flag2 = self._untangle_relations(M,relations,nonzero) # 行簡約化で関係式を簡単化
            relations, nonzero, flag3 = self._remove_duplicate(relations,nonzero) # 重複した等価な式たちを一本化
            flag = flag1 or flag2 or flag3

            relations = self._simplify_relations_value(relations) # 関係式の係数を簡単な比に変換

        # not necessary
        #print(f"-----relations-----")
        #print(*sorted([sorted(self._relation_to_ternary(relation, rank), key=lambda x:x[1]) for relation in relations]), sep="\n")
        #print(f"-------------------")

        # テンソル要素のうち非ゼロの要素の添字を0-indexedで出力
        print()
        print(f"number of nonzero elements: {len(nonzero)}")
        res: List[Tuple[int, ...]] = sorted([self._ternary(ijk, rank) for ijk in nonzero])
        print(f"nonzero elements: {res}")
        print()
        
        # 独立成分と従属成分を分離
        self._extract_independent(rank, relations, nonzero)
        return res # List[Tuple[int, ...]]


def main() -> None:
    pass

if __name__ == "__main__":
    main()

