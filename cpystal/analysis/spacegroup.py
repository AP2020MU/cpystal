"""`cpystal.analysis.spacegroup` is a module for analyzing space group of crystal.

Classes:
    `REF`
    `MatrixREF`
    `SymmetryOperation`
    `PhysicalPropertyTensorAnalyzer`

Functions:
    `spacegroup_to_pointgroup`
"""
from __future__ import annotations

from collections import defaultdict, deque
from fractions import Fraction
from functools import reduce
from itertools import combinations
from math import gcd
from typing import Any, Iterable, List, overload, Tuple, TypeVar
import re

import numpy as np
import numpy.typing as npt

#from ..core import Crystal

REFchild = TypeVar("REFchild", bound="REF")
MatrixREFchild = TypeVar("MatrixREFchild", bound="MatrixREF")
SymmetryOperationchild = TypeVar("SymmetryOperationchild", bound="SymmetryOperation")

class REF: # 有理数体Qに√pを添加した単純拡大体Q(√p)のclass REF(Rational Extension Field)
    """Rational Extension Field.

    This class represents an algebraic simple Rational Extension Field with a root of a polynomial: x^2-p, 
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
    def __init__(self: REFchild, p: int, a: Fraction = Fraction(), b: Fraction = Fraction()) -> None:
        self.p: int = p # Q[√p]の生成元の2乗
        self.a: Fraction = a
        self.b: Fraction = b
    
    def __str__(self: REFchild) -> str:
        if self.a == Fraction() and self.b == Fraction():
            return "0"
        elif self.a == Fraction():
            return f"{self.b}√{self.p}"
        elif self.b == Fraction():
            return f"{self.a}"
        else:
            return f"{str(self.a)}+{str(self.b)}√{self.p}"
        # return f"({str(self.a)}, {str(self.b)})"
    
    def __repr__(self: REFchild) -> str:
        return f"({str(self.a)}, {str(self.b)})"

    def __neg__(self: REFchild) -> REFchild:
        return self.__class__(self.p, -self.a, -self.b)
    
    def __eq__(self: REFchild, other: Any) -> bool:
        if isinstance(other, REF) and self.p == other.p and self.a == other.a and self.b == other.b:
            return True
        else:
            return False

    def __ne__(self: REFchild, other: Any) -> bool:
        return not self.__eq__(other)

    def __lt__(self: REFchild, other: REFchild) -> bool:
        if isinstance(other, REF):
            return self.to_float() < other.to_float()
        else:
            raise TypeError(f"'<' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __le__(self: REFchild, other: REFchild) -> bool:
        if isinstance(other, REF):
            return self.to_float() <= other.to_float()
        else:
            raise TypeError(f"'<=' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __gt__(self: REFchild, other: REFchild) -> bool:
        if isinstance(other, REF):
            return self.to_float() > other.to_float()
        else:
            raise TypeError(f"'>' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")

    def __ge__(self: REFchild, other: REFchild) -> bool:
        if isinstance(other, REF):
            return self.to_float() >= other.to_float()
        else:
            raise TypeError(f"'>=' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")

    def __add__(self: REFchild, other: Any) -> REFchild:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a+other, self.b)
        elif isinstance(other, REF):
            if self.p == other.p:
                return self.__class__(self.p, self.a+other.a, self.b+other.b)
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")

    def __radd__(self: REFchild, other: Any) -> REFchild:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a+other, self.b)
        elif isinstance(other, REF):
            if self.p == other.p:
                return self.__class__(self.p, self.a+other.a, self.b+other.b)
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __iadd__(self: REFchild, other: Any) -> REFchild:
        if type(other) is int or type(other) is Fraction:
            self.a += other
            return self
        elif isinstance(other, REF):
            if self.p == other.p:
                self.a += other.a
                self.b += other.b
                return self
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __sub__(self: REFchild, other: Any) -> REFchild:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a-other, self.b)
        elif isinstance(other, REF):
            if self.p == other.p:
                return self.__class__(self.p, self.a-other.a, self.b-other.b)
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __rsub__(self: REFchild, other: Any) -> REFchild:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a-other, self.b)
        elif isinstance(other, REF):
            if self.p == other.p:
                return self.__class__(self.p, self.a-other.a, self.b-other.b)
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __isub__(self: REFchild, other: Any) -> REFchild:
        if type(other) is int or type(other) is Fraction:
            self.a -= other
            return self
        elif isinstance(other, REF):
            if self.p == other.p:
                self.a -= other.a
                self.b -= other.b
                return self
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")

    def __mul__(self: REFchild, other: Any) -> REFchild:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a*other, self.b*other)
        elif isinstance(other, REF):
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

    def __rmul__(self: REFchild, other: Any) -> REFchild:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a*other, self.b*other)
        elif isinstance(other, REF):
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

    def __imul__(self: REFchild, other: Any) -> REFchild:
        if type(other) is int or type(other) is Fraction:
            self.a *= other
            self.b *= other
            return self
        elif isinstance(other, REF):
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
    
    def __truediv__(self: REFchild, other: Any) -> REFchild:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a/other, self.b/other)
        elif isinstance(other, REF):
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

    def __rtruediv__(self: REFchild, other: Any) -> REFchild:
        if type(other) is int or type(other) is Fraction:
            return self.__class__(self.p, self.a/other, self.b/other)
        elif isinstance(other, REF):
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

    def __itruediv__(self: REFchild, other: Any) -> REFchild:
        if type(other) is int or type(other) is Fraction:
            self.a /= other
            self.b /= other
            return self
        elif isinstance(other, REF):
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

    def __copy__(self: REFchild) -> REFchild:
        return self
    
    def copy(self: REFchild) -> REFchild:
        return self

    def __deepcopy__(self: REFchild) -> REFchild:
        return self.__class__(self.p, self.a, self.b)

    def deepcopy(self: REFchild) -> REFchild:
        return self.__class__(self.p, self.a, self.b)
    
    def zero(self: REFchild) -> REFchild: # 加法単位元 乗法零元
        return self.__class__(self.p, Fraction(), Fraction())
    
    def inv(self: REFchild) -> REFchild: # 乗法逆元
        p: int = self.p
        a: Fraction = self.a
        b: Fraction = self.b
        return self.__class__(p, a/(a**2 - p*b**2), -b/(a**2 - p*b**2))

    def swap(self: REFchild) -> REFchild: # 有理部と無理部を入れ替えたREFを生成
        p: int = self.p
        a: Fraction = self.a
        b: Fraction = self.b
        return self.__class__(p, b, a)

    def conjugate(self: REFchild) -> REFchild: # 一般化した共役(p=-1のとき複素共役に一致)
        p: int = self.p
        a: Fraction = self.a
        b: Fraction = self.b
        return self.__class__(p, a, -b)

    def to_float(self: REFchild) -> float:
        p: int = self.p
        if p < 0:
            raise ValueError(f"negative `REF.p`: {self.p}; to get the value of this REF instance, use 'to_complex'")
        a: Fraction = self.a
        b: Fraction = self.b
        return a.numerator/a.denominator + b.numerator/b.denominator*(p**0.5) # float

    def to_complex(self: REFchild) -> complex:
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
        mat (list[list[REF]]): 2-dimension matrix of `REF` instance.
        shape (tuple[int, int]): Shape of `MatrixREF.mat`. First element is the number of row, second is the number of column.
    """
    def __init__(self: MatrixREFchild, p: int, mat: Matrix | None = None): # p:int, mat:list[list[int/Fraction]]
        self.p: int = p
        self.mat: list[list[REF]] = []
        self.shape: tuple[int, int]
        if mat is None:
            self.shape = (0, 0)         
        else:
            self.shape = (len(mat), len(mat[0]))
            for row in mat:
                now: list[REF] = []
                for r in row:
                    if type(r) is int or type(r) is Fraction:
                        now.append(REF(p, Fraction(r)))
                    elif isinstance(r, REF):
                        if self.p == r.p:
                            now.append(r.deepcopy())
                        else:
                            raise TypeError(f"REF generator is not same")
                    else:
                        raise TypeError(f"type of matrix components must be int/Fraction/REF")
                self.mat.append(now)

    def __str__(self: MatrixREFchild) -> str:
        res: str = ',\n '.join(['['+', '.join([str(elem) for elem in row])+']' for row in self.mat])
        return f"[{res}]"

    def __repr__(self: MatrixREFchild) -> str:
        res: str = ',\n '.join(['['+', '.join([str(elem) for elem in row])+']' for row in self.mat])
        return f"[{res}]"
    
    def __len__(self: MatrixREFchild) -> int:
        return len(self.mat)
    
    def __eq__(self: MatrixREFchild, other: int | Fraction | MatrixREFchild) -> list[list[bool]]: # type: ignore
        m, n = self.shape
        smat: list[list[REF]] 
        omat: list[list[REF]]
        res: list[list[bool]]
        other_: REF
        if type(other) is int:
            other_ = REF(self.p, Fraction(other))
            smat = self.mat
            res = [[smat[i][j]==other_ for j in range(n)] for i in range(m)]
            return res # list[list[bool]]
        elif type(other) is Fraction:
            other_ = REF(self.p, other)
            smat = self.mat
            res = [[smat[i][j]==other_ for j in range(n)] for i in range(m)]
            return res # list[list[bool]]
        elif isinstance(other, MatrixREF):
            if self.p == other.p:
                smat = self.mat
                omat = other.mat
                res = [[smat[i][j]==omat[i][j] for j in range(n)] for i in range(m)]
                return res # list[list[bool]]
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"incompatible type(s) for : '{type(self).__name__}' and '{type(other).__name__}'")
        
    def __ne__(self: MatrixREFchild, other: Any) -> bool:
        if type(self).__name__ != type(other).__name__:
            return True
        if self.shape != other.shape:
            return True
        m, n = self.shape
        for i in range(m):
            for j in range(n):
                if self.mat[i][j] != other.mat[i][j]:
                    return True
        return False

    def __neg__(self: MatrixREFchild) -> MatrixREFchild:
        m, n = self.shape
        res: list[list[REF]] = [[REF(self.p)]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                res[i][j] = -self.mat[i][j]
        ret: MatrixREFchild = self.__class__(self.p, mat=None)
        ret.mat = res
        ret.shape = (m, n)
        return ret

    def __matmul__(self: MatrixREFchild, other: MatrixREFchild) -> MatrixREFchild:
        if isinstance(other, MatrixREF):
            if self.p == other.p:
                if self.shape[1] == other.shape[0]:
                    # (l,m)*(m,n) -> (l,n)
                    l: int = self.shape[0]
                    m: int = self.shape[1]
                    n: int = other.shape[1]
                    smat: list[list[REF]] = self.mat
                    omat: list[list[REF]] = other.mat
                    res: list[list[REF]] = [[REF(self.p)]*n for _ in range(l)]
                    for i in range(l):
                        for k in range(n):
                            val: REF = REF(self.p)
                            for j in range(m):
                                val += smat[i][j]*omat[j][k]
                            res[i][k] = val
                    ret: MatrixREFchild = self.__class__(self.p, mat=None)
                    ret.mat = res
                    ret.shape = (l, n)
                    return ret
                else:
                    raise TypeError(f"matrix shape is inappropriate")
            else:
                raise TypeError(f"REF generator is not same")
        else:
            raise TypeError(f"unsupported operand type(s) for @: '{type(self).__name__}' and '{type(other).__name__}'")
    
    @overload
    def __getitem__(self: MatrixREFchild, key: int) -> list[REF]:
        ...
    @overload
    def __getitem__(self: MatrixREFchild, key: slice) -> Iterable[list[REF]]:
        ...
    def __getitem__(self: MatrixREFchild, key: Any) -> Any:
        return self.mat[key]
    
    def __setitem__(self: MatrixREFchild, key: Any, value: Any) -> None:
        self.mat[key] = value
        
    def __copy__(self: MatrixREFchild) -> MatrixREFchild:
        return self
    
    def copy(self: MatrixREFchild) -> MatrixREFchild:
        return self

    def __deepcopy__(self: MatrixREFchild) -> MatrixREFchild:
        ret: MatrixREFchild = self.__class__(self.p)
        ret.mat = [[self.mat[i][j].deepcopy() for j in range(self.shape[1])] for i in range(self.shape[0])]
        ret.shape = self.shape
        return ret

    def deepcopy(self: MatrixREFchild) -> MatrixREFchild:
        ret: MatrixREFchild = self.__class__(self.p)
        ret.mat = [[self.mat[i][j].deepcopy() for j in range(self.shape[1])] for i in range(self.shape[0])]
        ret.shape = self.shape
        return ret

    def identity(self: MatrixREFchild, shape: tuple[int, int] | None = None) -> MatrixREFchild:
        # m×n単位行列
        if shape is None:
            m, n = self.shape
        else:
            m, n = shape
        p: int = self.p
        res: list[list[REF]] = [[REF(p)]*n for _ in range(m)]
        for i in range(min(m,n)):
            res[i][i] = REF(p, Fraction(1,1), Fraction())
        ret: MatrixREFchild = self.__class__(p)
        ret.shape = (m,n)
        ret.mat = res
        return ret
    
    def sum(self: MatrixREFchild, axis: int | None = None) -> REF | list[REF]:
        m, n = self.shape
        smat = self.mat
        res: REF | list[REF]
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
                return res # list[REF]
            elif axis == 1 or axis == -1:
                res = [REF(self.p)]*m
                for i in range(m):
                    for j in range(n):
                        res[i] += smat[i][j]
                return res # list[REF]
            else:
                raise KeyError(f"axis {axis} is out of bounds for MatrixREF of dimension {len(self.shape)}")

    def to_ndarray(self: MatrixREFchild) -> npt.NDArray:
        return np.array([[elem.to_float() for elem in row] for row in self.mat])

class _UnionFind:
    def __init__(self, n: int): # O(n)
        self.parent: list[int] = [-1]*n
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
    def family(self, x:int) -> list[int]: # xが属する連結成分を返す O(n)
        return [i for i in range(self.n) if self.issame(i,x)]
    def maximum(self) -> list[int]: # 最大連結成分を返す O(n)
        return self.family(self.parent.index(min(self.parent)))
    def all_root(self) -> list[int]: # 全ての根の番号を返す O(n)
        return [i for i in range(self.n) if self.parent[i] < 0]
    def decompose(self) -> list[list[int]]: # 連結成分分解を返す O(nα(n))
        return [self.family(i) for i in self.all_root()]


class SymmetryOperation(MatrixREF):
    """Symmetry operation represented as matrix.

    This class is inherited from `MatrixREF`.

    Args:
        p (int): Square of the generator of simple extension Q(√p).
        mat (list[list[REF]]): 2-dimension matrix of `REF` instance.
        name (str): Name of the symmetry operation.
        time_reversal (bool): True if the symmetry operation contains time-reversal operation.

    Attributes:
        p (int): Square of the generator of simple extension Q(√p).
        mat (list[list[REF]]): 2-dimension matrix of `REF` instance.
        shape (tuple[int, int]): Shape of `MatrixREF.mat`. First element is the number of row, second is the number of column.
        name (str): Name of the symmetry operation.
        time_reversal (bool): True if the symmetry operation contains time-reversal operation.
        determinant (int): Determinant of the symmetry operation. This value should be +1 or -1 due to orthogonal transformation.
    """
    def __init__(self, p: int, mat: Matrix, name: str = "", time_reversal: bool = False):
        super().__init__(p, mat=mat)
        self.name: str = name
        self.time_reversal: bool = time_reversal
    
    @property
    def determinant(self) -> int:
        return int(np.sign(np.linalg.det(self.to_ndarray())))

    def __neg__(self: SymmetryOperationchild) -> SymmetryOperationchild:
        m, n = self.shape
        res: list[list[REF]] = [[REF(self.p)]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                res[i][j] = -self.mat[i][j]
        ret: SymmetryOperationchild = self.__class__(
            self.p, 
            mat=res, 
            name="-" + self.name,
            time_reversal=self.time_reversal
        )
        return ret
    
    def __matmul__(self: SymmetryOperationchild, other: SymmetryOperationchild) -> SymmetryOperationchild:
        res: MatrixREF = super().__matmul__(other)
        ret: SymmetryOperationchild = self.__class__(
            self.p, 
            mat=res.mat, 
            name=f"{self.name} {other.name}",
            time_reversal=self.time_reversal ^ other.time_reversal
        )
        return ret

def spacegroup_to_pointgroup(name: str) -> str:
    """Convert space group name to point group name.

    1. Capital alphabets and spaces are removed.
    2. Subscripts (represented by "_" with numbers) are removed.
    3. All "1" except a component of monoclinic symbols ("1" or "-1") are removed.
    4. "a","b","c","d","e", and "n" are converted to "m"
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
    name = re.sub(r"[A-Z]|\s", "", name)
    name = re.sub(r"_\d|_\{\d\}", "", name)
    name = re.sub(r"([^-])1", "\\1", name)
    name = re.sub(r"[a-en]", "m", name)
    return name

def decompose_jahn_symbol(symbol: str) -> tuple[int, str, bool, bool]:
    axiality: bool = "e" in symbol
    time_reversality: bool = "a" in symbol
    symbol = re.sub(r"a|e|\^", "", symbol) # aとeを消去
    symbol = re.sub(r"V([^\d]|$)", r"V1\1", symbol) # 'V' -> 'V1'
    symbol = re.sub(r"\[(.*)\]\*", r"(\1)", symbol) # []* -> ()
    symbol = re.sub(r"\{(.*)\}\*", r"<\1>", symbol) # {}* -> <>
    
    def symbol_to_expression(symbol: str, i: int, base: list[str], expr_list: list[str], n_char: int) -> tuple[str, int, list[str], list[str], int]:
        def regenerate_base_expr_list(base: list[str], expr_list: list[str], symmetry_type: int) -> tuple[list[str], list[str]]:
            if len(base) == 1:
                return base, expr_list
            new_base: list[str] = ["".join(base)]
            new_expr_list: list[str] = []
            divided_expr_list: list[str] = [[expr for expr in expr_list if b in expr] for b in base]
            # baseの統合
            for i in range(len(base)):
                left_bases: str = "".join(base[:i])
                right_bases: str = "".join(base[i+1:]) if i+1 < len(base) else ""
                for expr in divided_expr_list[i]:
                    # '=' の前半のindexをnew_baseに合わせて拡張
                    new_expr: str = re.sub(r"^(.*)=", left_bases+r"\1"+right_bases+"=", expr)
                    # '=' の後半のindexをnew_baseに合わせて拡張
                    new_expr = re.sub(r"(=-|=)(.*)$", r"\1"+left_bases+r"\2"+right_bases, new_expr)
                    # 時間反転操作がある場合は後ろにずらす
                    if "'" in new_expr:
                        new_expr = re.sub(r"'", "", new_expr) + "'"
                    new_expr_list.append(new_expr)
            for i, j in combinations(range(len(base)), 2): # 全てのbaseの置換に対して対称性ごとに処理
                ji: str = "".join([base[j] if n == i else (base[i] if n == j else b) for n,b in enumerate(base)])
                # 対称性に応じてbase間の関係を追加
                if symmetry_type == 0:
                    # 単純統合
                    pass
                elif symmetry_type == 1:
                    # 対称
                    new_expr_list.append(f"{new_base[0]}={ji}")
                elif symmetry_type == 2:
                    # 反対称
                    new_expr_list.append(f"{new_base[0]}=-{ji}")
                elif symmetry_type == 3:
                    # 時間反転で対称
                    new_expr_list.append(f"{new_base[0]}={ji}'")
                elif symmetry_type == 4:
                    # 時間反転で反対称
                    new_expr_list.append(f"{new_base[0]}=-{ji}'")
                else:
                    pass
            return new_base, new_expr_list

        characters: str = "ijklmnopqrstuvwxyzabcdefgh"
        left_parentheses: str = "[{(<"
        right_parentheses: str = "]})>"

        if i == -1: # 最終的な結果
            i += 1
            symbol, i, base, expr_list, n_char = symbol_to_expression(symbol, i, base, expr_list, n_char)
            base, expr_list = regenerate_base_expr_list(base, expr_list, symmetry_type=0)
            return symbol, i, base, expr_list, n_char
        while i < len(symbol):
            if symbol[i] in left_parentheses:
                symmetry_type: int = 1 + left_parentheses.index(symbol[i])
                i += 1
                symbol, i, new_base, new_expr_list, n_char = symbol_to_expression(symbol, i, [], [], n_char)
                new_base, new_expr_list = regenerate_base_expr_list(new_base, new_expr_list, symmetry_type=symmetry_type)
                base = base + new_base
                expr_list = expr_list + new_expr_list
            elif symbol[i] in right_parentheses:
                i += 1
                return symbol_to_expression(symbol, i, base, expr_list, n_char)
            elif symbol[i] == "V":
                base = base + [characters[n_char+b] for b in range(int(symbol[i+1]))]
                n_char += int(symbol[i+1])
                i += 2
                return symbol, i, base, expr_list, n_char
            else:
                raise ValueError
        return symbol, i, base, expr_list, n_char
    _, _, _, expr_list, rank = symbol_to_expression(symbol, -1, [], [], 0)

    return rank, ",".join(expr_list), axiality, time_reversality

Relation = List[Tuple[REF, int]]
Relations = List[List[Tuple[REF, int]]]
Relation_ternary = List[Tuple[REF, Tuple[int, ...]]]
Relations_ternary = List[List[Tuple[REF, Tuple[int, ...]]]]

class PhysicalPropertyTensorAnalyzer:
    """Analyze non-zero elements of physical property tensors based on the symmetry of crystallographic point group.

    There is software on the browser that performs similar or better than this class.
    'MTENSOR": https://www.cryst.ehu.es/cgi-bin/cryst/programs/mtensor.pl
    If you want to double-check the result of this class, use 'MTENSOR'.

    Note:
        All symmetry operations of crystallographic point groups can be represented as a 3×3 matrix on a simple rational extension field: 'M_{3×3}(Q(√3))' in an appropriate orthogonal basis.
        Therefore, it is possible to determine which elements are equivalent or zero by straightforward exact calculation.

    Attributes:
        point_group_name (str): Target point group name written in Schönflies notation.
        unitary_matrices (list[MatrixREF]): list of the symmetry operations of the crystallographic point group represented as a matrix in an appropriate orthogonal basis.

    """
    # 適切に定めた直交座標系を基底にとったときの対称操作の行列表示
    # 基本並進ベクトルたちが直交系をなすならそれを使う
    C1 = SymmetryOperation(3, [[1,0,0],
                                [0,1,0],
                                [0,0,1]], name="C1")

    C2_100 = SymmetryOperation(3, [[1,0,0],
                                    [0,-1,0],
                                    [0,0,-1]], name="C2_100")
    C2_010 = SymmetryOperation(3, [[-1,0,0],
                                    [0,1,0],
                                    [0,0,-1]], name="C2_010")
    C2_001 = SymmetryOperation(3, [[-1,0,0],
                                    [0,-1,0],
                                    [0,0,1]], name="C2_001")

    C2_n101 = SymmetryOperation(3, [[0,0,-1],
                                   [0,-1,0],
                                   [-1,0,0]], name="C2_n101")
    C2_1n10 = SymmetryOperation(3, [[0,-1,0],
                                    [-1,0,0],
                                    [0,0,-1]], name="C2_1n10")
    C2_01n1 = SymmetryOperation(3, [[-1,0,0],
                                    [0,0,-1],
                                    [0,-1,0]], name="C2_01n1")

    C2_110 = SymmetryOperation(3, [[0,1,0],
                                    [1,0,0],
                                    [0,0,-1]], name="C2_110")

    C3_111 = SymmetryOperation(3, [[0,0,1],
                                    [1,0,0],
                                    [0,1,0]], name="C3_111")
    C3_1n1n1 = SymmetryOperation(3, [[0,0,-1],
                                    [-1,0,0],
                                    [0,1,0]], name="C3_1n1n1")
    C3_n11n1 = SymmetryOperation(3, [[0,0,1],
                                    [-1,0,0],
                                    [0,-1,0]], name="C3_n11n1")
    C3_n1n11 = SymmetryOperation(3, [[0,1,0],
                                    [0,0,-1],
                                    [-1,0,0]], name="C3_n1n11")

    C3_001 = SymmetryOperation(3, [[0,-1,0],
                                    [1,-1,0],
                                    [0,0,1]], name="C3_001")
    C3_001 = SymmetryOperation(3,  [[REF(3,Fraction(-1,2)),REF(3,b=Fraction(1,2)),0],
                                    [REF(3,b=Fraction(-1,2)),REF(3,Fraction(-1,2)),0],
                                    [0,0,1]], name="C3_001")

    C4_001 = SymmetryOperation(3, [[0,-1,0],
                                    [1,0,0],
                                    [0,0,1]], name="C4_001")
    C4_010 = SymmetryOperation(3, [[0,0,1],
                                    [0,1,0],
                                    [-1,0,0]], name="C4_010")
    C4_100 = SymmetryOperation(3, [[1,0,0],
                                    [0,0,-1],
                                    [0,1,0]], name="C4_100")


    m_100 = SymmetryOperation(3, [[-1,0,0],
                                    [0,1,0],
                                    [0,0,1]], name="m_100")
    m_010 = SymmetryOperation(3, [[1,0,0],
                                    [0,-1,0],
                                    [0,0,1]], name="m_010")
    m_001 = SymmetryOperation(3, [[1,0,0],
                                    [0,1,0],
                                    [0,0,-1]], name="m_001")

    m_110 = SymmetryOperation(3, [[0,-1,0],
                                    [-1,0,0],
                                    [0,0,1]], name="m_110")
    m_1n10 = SymmetryOperation(3, [[0,1,0],
                                    [1,0,0],
                                    [0,0,1]], name="m_1n10")
    m_n101 = SymmetryOperation(3, [[0,0,1],
                                    [0,1,0],
                                    [1,0,0]], name="m_n101")
    m_01n1 = SymmetryOperation(3, [[1,0,0],
                                    [0,0,1],
                                    [0,1,0]], name="m_01n1")

    inversion = SymmetryOperation(3, [[-1,0,0],
                                        [0,-1,0],
                                        [0,0,-1]], name="inversion")

    time_reversal = SymmetryOperation(3, [[1,0,0],
                                            [0,1,0],
                                            [0,0,1]], name="time-reversal", time_reversal=True)

    # 国際表記 -> Schönflies表記
    international_to_schoenflies_notation: dict[str, str] = {
        # 立方晶(cubic system)
        "m-3m": "Oh",
        "4/m-32/m": "Oh", # 表記揺れ
        "-43m": "Td",
        "432": "O",
        "m-3":"Th",
        "23": "T",
        # 正方晶(tetragonal system)
        "4/mmm": "D4h",
        "-42m": "D2d",
        "-4m2": "D2d", # 表記揺れ
        "4mm" :"C4v",
        "422" :"D4",
        "4/m": "C4h",
        "-4": "S4",
        "4": "C4",
        # 直方晶(orthorhombic system)
        "mmm": "D2h",
        "mm2": "C2v",
        "2mm": "C2v", # 表記揺れ
        "222": "D2",
        # 六方晶(hexagonal system)
        "6/mmm": "D6h",
        "-62m": "D3h",
        "-6m2": "D3h", # 表記揺れ
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

    point_groups: list[str] = [
        "1", "-1",
        "2", "m", "2/m",
        "3", "-3", "32", "3m", "-3m",
        "6", "-6", "6/m", "622", "6mm", "-62m", "6/mmm",
        "222", "mmm", "mm2",
        "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
        "23", "m-3", "432", "-43m", "m-3m",
    ]

    enantiomorphous_point_groups: list[str] = [
        "1", "2", "222", "4", "422", "3", "32", "6", "622", "23", "432"
    ]

    chiral_point_groups: list[str] = [
        "1", "2", "222", "4", "422", "3", "32", "6", "622", "23", "432"
    ]

    polar_point_groups: list[str] = [
        "1", "m", "2", "mm2", "4", "4mm", "3", "3m", "6", "6mm",
    ]

    centrosymmetric_point_groups: list[str] = [
        "-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m",
    ]

    noncentrosymmetric_point_groups: list[str] = [
        "1", "2", "m", "222", "mm2", "4", "-4", "422", "4mm", "-42m", "3", "3m", "32", "6", "-6", "622", "6mm", "-62m", "23", "432", "-43m",
    ]


    # 参照元 https://www.cryst.ehu.es/cryst/get_point_genpos.html (三方晶やC3_001の行列は改変)
    # 各結晶点群の生成元(適切に定めた直交座標系を基底にとっていることに注意)
    # 三方晶では
    # (default):   [001]方向をz軸，2回回転がある場合はa軸が回転軸，鏡映がある場合はa軸に垂直に鏡映面
    # _rombohedral: [111]方向をxyzでの(1,1,1)方向，c軸とz軸と[111]がx=y上になるようにとる
    pointgroup_generators: dict[str, list[SymmetryOperation]] = {
        # 立方晶
        "Oh": [C2_001, C2_010, C3_111, C2_110, inversion],
        "Td": [C2_001, C2_010, C3_111, m_1n10],
        "O":  [C2_001, C2_010, C3_111, C2_110],
        "Th": [C2_001, C2_010, C3_111, inversion],
        "T":  [C2_001, C2_010, C3_111],
        # 正方晶
        "D4h":[C2_001, C4_001, C2_010, inversion],
        "D2d":[C2_001, -C4_001, C2_010],
        "C4v":[C2_001, C4_001, m_010],
        "D4": [C2_001, C4_001, C2_010],
        "C4h":[C2_001, C4_001, inversion],
        "S4": [C2_001, -C4_001],
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
        self.unitary_matrices: list[SymmetryOperation] = self.pointgroup_generators[pgname_in_schoenflies_notation]

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
            i: int = m-1
            for idx in range(k,m):
                if B[idx][k] != zero:
                    flag = False
                    i = idx
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
    def _ternary(cls, n: int, fill: int, is_0indexed: bool = True) -> tuple[int, ...]: # fill: 桁数
        res: list[int] = [] # res[i] := nの3進展開の"下から"i桁目
        r: int
        while n:
            n, r = divmod(n,3)
            res.append(r)
        if is_0indexed: # 0-indexed
            return tuple(res + [0]*max(0,fill-len(res)))
        else: # 1-indexed
            res = [i+1 for i in res]
            return tuple(res + [1]*max(0,fill-len(res)))
    
    @classmethod
    def _ternary2int(cls, ternary_ijk: tuple[int, ...]) -> int:
        return sum([n * 3**i for i,n in enumerate(ternary_ijk)])
    
    @classmethod
    def _relations_from_symmetry_operation(cls, rank: int, R: SymmetryOperation, axiality: bool, time_reversality: bool = False) -> Relations:
        # N階のテンソルに直交変換Rを施したときの，要素間の関係式(つまりテンソルに関わる物理量は全て共変と仮定)
        # relations[i]: \sum_{j} relations[i][j][0]*A_{relations[i][j][1]} = 0 という関係式を表す
        # rank=2: A_{ij}  = R_{il}R{jm} A_{lm}
        # rank=3 and axial: A_{ijk} = det(R)R_{il}R{jm}R_{kn} A_{lmn} 
        # などを元に計算
        n_elem: int = 3**rank
        zero: REF = REF(3) # 有理拡大体の零元
        relations: Relations = []
        for ijk in range(n_elem):
            now: Relation = []
            val_ijk: REF = REF(3, Fraction(1,1))
            for lmn in range(n_elem): # 添字について和をとる
                val: REF = REF(3, Fraction(1,1))
                for a, b in zip(cls._ternary(ijk,rank), cls._ternary(lmn,rank)):
                    val *= R[a][b]
                if val != zero: # 非ゼロの項のみ記録
                    sign: REF = REF(3, Fraction(-1,1)) # -1
                    # 軸性テンソルにimproper xor 磁気的テンソルに時間反転 -> 符号は-1倍
                    if (axiality and R.determinant == -1) ^ (time_reversality and R.time_reversal):
                        sign = -sign
                    if ijk == lmn:
                        val_ijk += sign * val
                    else:
                        now.append((sign * val, lmn))
            if val_ijk != zero:
                now.append((val_ijk, ijk))
            if now:
                relations.append(now)
        return relations

    @classmethod
    def _relations_from_expression(cls, rank: int, expr: str) -> Relations:
        # 対称性から同値になるテンソル要素たちの関係式を生成
        # "ijk = -jik = kij" のように，添字で物理的な制約に基づくテンソルの対称性を導入する
        # 複数の条件がある場合はカンマで区切る
        # マイナス符号も使える
        # 時間反転操作で等価になる場合"'"(prime)で表示できるが，この関数では無視する
        n_elem: int = 3**rank
        expressions: list[list[str]] = [s.split("=") for s in re.sub(r"[\u3000 \t]", "", expr).split(",") if not "'" in s]
        expressions_data: list[tuple[list[str], int, defaultdict[str,list[int]], int, defaultdict[str,list[int]]]] = []
        relations: Relations = []
        for expression in expressions:
            for i in range(1,len(expression)):
                expr0: str = expression[0]
                expr1: str = expression[i]
                sign0: int = 1
                sign1: int = 1
                if expr0[0] == "-":
                    sign0 = -1
                    expr0 = expr0[1:]
                if expr1[0] == "-":
                    sign1 = -1
                    expr1 = expr1[1:]
                if len(expr0) != rank or len(expr1) != rank:
                    raise ValueError(f"expressions must be all the same length: {rank}")
                characters: list[str] = list(set(list(expr0+expr1)))
                d0: defaultdict[str, list[int]] = defaultdict(list) # 添字表現=expressionにおいて，文字cが出現するindexのリスト
                d1: defaultdict[str, list[int]] = defaultdict(list)
                for j in range(rank):
                    d0[expr0[j]].append(j)
                    d1[expr1[j]].append(j)
                expressions_data.append((characters, sign0, d0, sign1, d1)) # この関係式で出現する文字全体の集合も持っておく
        for ijk in range(n_elem):
            for lmn in range(ijk,n_elem):
                ijk_ternary: tuple[int, ...] = cls._ternary(ijk, rank)
                lmn_ternary: tuple[int, ...] = cls._ternary(lmn, rank)
                for characters, s0, d0, s1, d1 in expressions_data:
                    flag: bool = True
                    for c in characters: # 表現の文字ごとに見る
                        numbers: set[int] = set()
                        for i in d0[c]:
                            numbers.add(ijk_ternary[i])
                        for l in d1[c]:
                            numbers.add(lmn_ternary[l])
                        if len(numbers) != 1: # 同じ文字に対応する添字の数字は1種類でないといけない
                            flag = False
                            break
                    if flag: # 全ての文字について条件を満たすならそれらのテンソル要素は表現に従う
                        # s0とs1の符号が一致するなら ijk-lmn=0, 違うなら ijk+lmn=0
                        relations.append([(REF(3, Fraction(1,1)), ijk), (REF(3, Fraction(-s0*s1,1)), lmn)])
                        break
        return relations

    @classmethod
    def _relation_to_ternary(cls, relation: Relation, rank: int) -> Relation_ternary:
        # relationの添字を3進数に変換(printでの出力用)
        return [(val, cls._ternary(ijk,rank)) for val,ijk in relation]
    
    @classmethod
    def _relations_to_ternary(cls, relations: Relations, rank: int) -> list[Relation_ternary]:
        # relationの添字を3進数に変換(printでの出力用)
        return [cls._relation_to_ternary(relation, rank) for relation in relations]
    
    @classmethod
    def _extract_relation(cls, relations: Relations, rank: int, ternary_ijk: tuple[int, ...]) -> Relations:
        # ternary_ijkが入っている関係式を抽出
        res: Relations = []
        for relation in relations:
            if cls._ternary2int(ternary_ijk) in [cls._ternary(ijk,rank) for _,ijk in relation]:
                res.append(relation)
        return res
    
    @classmethod
    def _formulize_relations(cls, relations: Relations, rank: int) -> list[str]:
        """ToDo: is_0indexed に対応
        """
        # 関係式を数学的に表示
        res: list[str] = []
        for relation in relations:
            if len(relation) > 0:
                res.append(" + ".join([f"{val} ({''.join([str(i) for i in ijk])})" for val,ijk in cls._relation_to_ternary(relation,rank)]) + " = 0")
        return res

    @classmethod
    def _nonzero_matrix(cls, nonzero: set[int], direction: int, rank: int) -> str:
        res: list[list[str]] = [[" " if i>j else "." for j in range(3)] for i in range(3)]
        for ijk in nonzero:
            i, j, *klm = cls._ternary(ijk, rank)
            if i <= j and all([k == direction for k in klm]):
                # 3階: k=direction について ijk を行列表示
                # 4階: k=direction について ijkk を行列表示
                # n階: k=direction について ijk...k を行列表示
                res[i][j] = "o"
        return "\n".join([" ".join(r) for r in res])

    @classmethod
    def _summarize_same_term(cls, relations: Relations) -> Relations:
        # 同じ項をまとめる
        renewed_relations: Relations = []
        for relation in relations:
            dic: defaultdict[int, REF] = defaultdict(lambda: REF(3))
            for val, ijk in relation:
                dic[ijk] += val
            renewed_relations.append([(v,k) for k,v in dic.items() if v != REF(3)])
        return renewed_relations

    @classmethod
    def _delete_zero_term(cls, relations: Relations, nonzero: set[int]) -> tuple[Relations, set[int], bool]:
        # 0の要素を削除
        flag: bool = False # 更新が起こったかどうか
        relations = cls._summarize_same_term(relations) # 同じ項をまとめる
        renewed: Relations = []
        for relation in relations:
            if len(relation) == 1: # 長さ1の式なのでこの項は0
                flag = True
                ijk: int = relation[0][1]
                if ijk in nonzero:
                    nonzero.discard(ijk) # 0の要素を除外
            else:
                now: Relation = []
                for val, ijk in relation:
                    if not ijk in nonzero: # 式中の0の要素を削除
                        flag = True
                        continue
                    now.append((val,ijk))
                if now:
                    renewed.append(now)
        return renewed, nonzero, flag

    @classmethod
    def _untangle_relations(cls, rank: int, relations: Relations, nonzero: set[int]) -> tuple[Relations, set[int], bool]:
        # 係数行列簡約化で関係式を簡約化
        flag: bool = False
        zero: REF = REF(3)
        n_elem: int = 3**rank
        U: _UnionFind = _UnionFind(n_elem)
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
            family_inv: dict[int, int] = {a:i for i, a in enumerate(family)}
            A: list[list[REF]] = [] # 係数行列
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
            zero_elements: list[int] = []
            m, n = AA.shape
            for i in range(m):
                idxs: list[int] = []
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
        return renewed, nonzero, flag # tuple[list[list[tuple[REF,int]]], set[int], bool]

    @classmethod
    def _remove_duplicate(cls, relations: Relations, nonzero: set[int]) -> tuple[Relations, set[int], bool]:
        # 複数の等価な式を一本化
        flag: bool = False
        renewed: Relations = []
        for idx1 in range(len(relations)):
            d1: defaultdict[int, REF] = defaultdict(lambda: REF(3))
            for val, ijk in relations[idx1]:
                d1[ijk] += val
            flag2: bool = True # relations[idx1]と等価なものが存在するならFalse
            for idx2 in range(idx1,len(relations)):
                if idx1 == idx2:
                    continue
                d2: defaultdict[int, REF] = defaultdict(lambda: REF(3)) # (関係式)=0において，関係式の各テンソル要素の係数を計算
                d2_inv: defaultdict[int, REF] = defaultdict(lambda: REF(3)) # 両辺に-1をかけた関係式
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
        return renewed, nonzero, flag # tuple[list[list[tuple[REF,int]]], set[int], bool]

    @classmethod
    def _simplify_coefficient(cls, R: list[REF]) -> list[REF]: # O(len(A))
        # Rの要素たちの比を保ったままREFの有理部と無理部の係数を整数に簡約化
        def lcm(a: int, b: int) -> int:
            return a * b // gcd(a, b) # int 
        L: list[int] = []
        G: list[int] = []
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
        res: list[REF]
        if flag:
            res = [r*f for r in R]
        else: # 有理部と無理部の入れ替え
            res = [r.swap()*f for r in R]
        return res # list[REF]

    @classmethod
    def _simplify_relations_value(cls, relations: Relations) -> Relations:
        # relation: list[tuple[REF,int]] (in relations)の比を保ったままREFの有理部と無理部の係数を整数に簡約化
        res: Relations = []
        for relation in relations:
            if len(relation) == 0:
                continue
            vals: list[REF] = []
            ijks: list[int] = []
            for val, ijk in relation:
                vals.append(val)
                ijks.append(ijk)
            res.append(list(zip(cls._simplify_coefficient(vals), ijks))) # REFの有理部と無理部の簡約化
        return res # list[list[tuple[REF, int]]]

    @classmethod
    def _extract_independent(cls, rank: int, relations: Relations, nonzero: set[int]) -> tuple[set[int], Relations]:
        def printer_indep_dep(dep):
            print([cls._ternary(dep, rank)])
        if len(relations) == 0:
            return nonzero, []
        # 独立な成分と従属成分に分離
        indep: set[int] = set()
        dep: set[int] = set()
        relations = [relation for relation in relations if len(relation) > 0]
        # 式に現れる項が少ない順に処理
        relations = sorted([sorted(relation, key=lambda x:cls._ternary(x[1], rank)) for relation in relations], key=lambda x:len(x))
        for relation in relations:
            family: list[int] = [ijk for _, ijk in relation]
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
        for ijk in nonzero: # 上の処理の後に従属成分にない非ゼロ項はすべて独立
            if not ijk in dep:
                indep.add(ijk)
        # 行列の列を入れ替えて，左側にdepを，右側にindepを寄せる
        # 掃き出せばdepをindepたちだけで表現できる
        active: list[int] = sorted(list(nonzero), key=lambda x:cls._ternary(x, rank))
        l: list[int] = []
        r: list[int] = []
        for ijk in active:
            if ijk in dep:
                l.append(ijk)
            elif ijk in indep:
                r.append(ijk)
            else:
                indep.add(ijk)
                r.append(ijk)
        active = l + r
        active_inv: dict[int, int] = {a:i for i, a in enumerate(active)}
        
        # printer_indep_dep(indep)
        # printer_indep_dep(dep)
        
        A: list[list[REF]] = [] # 係数行列
        for relation in relations:
            a = [REF(3)]*len(active)
            for val, ijk in relation:
                a[active_inv[ijk]] = val
            A.append(a)
        AA: MatrixREF = cls._Gauss_Elimination_REF(MatrixREF(3, A)) # 掃き出し法で行簡約
        
        dep_represented_by_indep: Relations = [] # 従属成分たちのみを対象とした関係式(基本的に引数のrelationsと等価)
        for i in range(len(AA)):
            idxs: list[int] = []
            for j in range(len(active)):
                if AA[i][j] != REF(3):
                    idxs.append(j)
            if len(idxs) > 0:
                dep_represented_by_indep.append([(AA[i][j], active[j]) for j in idxs])
        return indep, dep_represented_by_indep

    @classmethod
    def _represent_dep_by_indep(cls, rank: int, relation: Relation, is_0indexed: bool = True) -> str:
        # 独立成分のみで従属成分を表示
        r: REF
        ijk0: int
        r, ijk0 = relation[0]
        res: list[str] = []
        for val, ijk in relation[1:]:
            v: REF = -val/r
            if v.b == 0:
                if v.a == 1:
                    res.append(str(cls._ternary(ijk, rank, is_0indexed=is_0indexed)))
                elif v.a == -1:
                    res.append(f"-{cls._ternary(ijk, rank, is_0indexed=is_0indexed)}")
                else:
                    res.append(str(v.a)+str(cls._ternary(ijk, rank, is_0indexed=is_0indexed)))
            elif v.a == 0:
                if v.b.denominator == 1:
                    if v.b == 1:
                        res.append(f"√{v.p}"+str(cls._ternary(ijk, rank, is_0indexed=is_0indexed)))
                    else:
                        res.append(f"{v.b}√{v.p}"+str(cls._ternary(ijk, rank, is_0indexed=is_0indexed)))
                else:
                    if v.b.numerator == 1:
                        res.append(f"√{v.p}/{v.b.denominator}"+str(cls._ternary(ijk, rank, is_0indexed=is_0indexed)))
                    else:
                        res.append(f"{v.b.numerator}√{v.p}/{v.b.denominator}"+str(cls._ternary(ijk, rank, is_0indexed=is_0indexed)))
            else:    
                res.append(f"({v.a}+{v.b}√{v.p})"+str(cls._ternary(ijk, rank, is_0indexed=is_0indexed)))
        return f"{cls._ternary(ijk0, rank, is_0indexed=is_0indexed)} = " + " + ".join(res)


    @classmethod
    def _contract_relations_along_with_direction(cls, relations: Relations, rank: int, magnetic_field_direction: int) -> Relations:
        new_relations: Relations = []
        for relation in relations:
            flag: bool = True
            new_relation: Relation = []
            for val, ijk in relation:
                i, j, *klm = cls._ternary(ijk, rank)
                if len(klm) > 0 and not all([k == klm[0] for k in klm]): # 一軸磁場のみ考慮
                    flag = False
                    break
                else:
                    if len(klm) == 0: # 磁場依存しない(B^0に比例する)項を求めていることに注意
                        new_relation.append((val, cls._ternary2int((i,j,magnetic_field_direction))))
                    else: # 縮約してできる3階のテンソルの中で，別方向の一軸磁場での要素同士が関係を持つ可能性があるので，それは残す
                        new_relation.append((val, cls._ternary2int((i,j,klm[0]))))
            if flag:
                new_relations.append(new_relation)
        return new_relations

    @classmethod
    def _contract_nonzero_along_with_direction(cls, nonzero: set[int], rank: int, direction: int) -> set[int]:
        # 指定した印加磁場方向の輸送テンソルの非ゼロ要素
        res: set[int] = set()
        for ijk in nonzero:
            i, j, *klm = cls._ternary(ijk, rank)
            if all([k == direction for k in klm]): # direction方向の一軸磁場に対する応答に対応する要素
                res.add(cls._ternary2int((i,j,direction)))
        return res

    @classmethod
    def _contract_along_with_direction(cls, relations: Relations, rank: int, nonzero: set[int], magnetic_field_direction: int) -> tuple[Relations, set[int], set[int], set[int]]:
        # rank 階のテンソルの磁場に対応する部分の足を縮約して，一軸磁場の2階のテンソルに直す
        n_nonzero: int = len(nonzero)
        vertex: list[int] = [ijk for ijk in sorted(nonzero)]
        ijk2idx: dict[int, int] = {ijk:i for i,ijk in enumerate(vertex)}
        G: list[dict[int, REF]] = [dict() for _ in range(n_nonzero)] # 絶対値が一致する項同士に辺を張ったグラフ
        pair_relations: Relations = sorted([relation for relation in relations if len(relation) == 2], key=lambda x:len(x))
        more_than_two_relations: Relations = sorted([relation for relation in relations if len(relation) > 2], key=lambda x:len(x))
        for relation in pair_relations:
            val0,ijk0 = relation[0]
            val1,ijk1 = relation[1]
            if ijk0 in ijk2idx and ijk1 in ijk2idx:
                idx0: int = ijk2idx[ijk0]
                idx1: int = ijk2idx[ijk1]
                G[idx0][idx1] = -val1/val0 # 辺の重みは符号が同じなら-1, 符号が逆なら+1 (ax+by=0 の x=-b/ay の-b/aの部分)
                G[idx1][idx0] = -val1/val0 # 辺の重みは符号が同じなら-1, 符号が逆なら+1 (ax+by=0 の x=-b/ay の-b/aの部分)
        
        def check_direction(ijk: int) -> bool:
            return all([i == magnetic_field_direction for i in cls._ternary(ijk, rank)[2:]])

        def check_uniaxiality(ijk: int) -> bool:
            ternary_ijk: tuple[int, ...] = cls._ternary(ijk, rank)
            if len(ternary_ijk) <= 2:
                return True
            else:
                return all([i == ternary_ijk[2] for i in ternary_ijk[2:]])

        def printer(relations: Relations) -> None:
            # デバッグ用
            print(*cls._formulize_relations(relations, rank), sep="\n")

        def printer_nonzero(nonzero: set[int], rank: int) -> None:
            # デバッグ用
            print(*sorted([cls._ternary(i,rank) for i in nonzero]), sep="\n")
        # printer(relations)
        # printer_nonzero(nonzero,rank)
        
        # 足の磁場部分が求めたい方向の一軸磁場になっている添字
        searched_vertex: list[int] = sorted([ijk for ijk in vertex if check_direction(ijk)], key=lambda x:cls._ternary(x,rank))
        searched_idx: list[int] = [ijk2idx[ijk] for ijk in searched_vertex] # 番号を振っておく
        zeros: set[int] = set() # 結果的に0になる項の添字(閉路上の辺重みの積が-1になる場合)
        connections: list[dict[int, REF]] = [dict() for _ in range(n_nonzero)] # UnionFindでいうところの，辺の根への付け替え
        used: set[int] = set()
        indep: set[int] = set()
        for start in searched_idx:
            if start in used: # もし既に訪れていたら，どれかの従属成分であることがわかる
                continue
            indep.add(vertex[start])
            visited: list[REF | None] = [None] * n_nonzero
            visited[start] = REF(3, Fraction(1,1))
            que = deque([start])
            all_zero_flag: bool = False
            while que: # ここからBFS
                idx: int = que.popleft()
                used.add(idx)
                for nex in G[idx]:
                    val_edge: REF = G[idx][nex]
                    val_nex: REF = val_edge * visited[idx]
                    if visited[nex] is None: # 未訪問の場合は更新
                        que.append(nex)
                        visited[nex] = val_nex
                    else:
                        # 符号が衝突したらある閉路上の辺重みの積が-1となり，連結成分は全てゼロだとわかる
                        if visited[nex] is not None and visited[nex] != val_nex:
                            # print("all_zero_flag")
                            # print(cls._ternary(vertex[idx],rank))
                            # print(cls._ternary(vertex[nex],rank))
                            # print(visited[nex], val_nex)
                            all_zero_flag = True
                    # 方向が別でも一軸性があればOK
                    if check_uniaxiality(vertex[nex]) and not nex in connections[start]: # 子孫(nex)を根(start)に繋ぐ
                        connections[start][nex] = val_nex
            if all_zero_flag:
                for idx in range(n_nonzero):
                    if visited[idx] is not None: # 訪問したidxは全て0になる
                        zeros.add(idx)
        
        new_relations: Relations = []
        dep: set[int] = set()
        for start in searched_idx:
            if len(connections[start]) > 0:
                for nex in connections[start]:
                    dep.add(vertex[nex])
                    new_relations.append(
                        [
                            (REF(3, Fraction(1,1)), vertex[start]),
                            (-connections[start][nex], vertex[nex]),
                        ]
                    )
        new_relations = new_relations + more_than_two_relations
        # print("indep")
        # printer_nonzero(indep,rank)
        # print("dep")
        # printer_nonzero(dep,rank)
        for ijk in nonzero:
            if not (ijk in zeros or ijk in dep):
                indep.add(ijk)
        for z in zeros:
            nonzero.discard(z)
        
        relations, nonzero, _ = cls._delete_zero_term(new_relations, nonzero)
        # print("nonzero",)
        # printer_nonzero(nonzero, 4)

        # 磁場の足1個は残してある
        contracted_relations: Relations = cls._contract_relations_along_with_direction(relations, rank, magnetic_field_direction)
        contracted_nonzero: set[int] = cls._contract_nonzero_along_with_direction(nonzero, rank, magnetic_field_direction)
        # print("c_nonzero")
        # printer_nonzero(contracted_nonzero,2)
        contracted_indep: set[int] = cls._contract_nonzero_along_with_direction(indep, rank, magnetic_field_direction)
        contracted_dep: set[int] = cls._contract_nonzero_along_with_direction(dep, rank, magnetic_field_direction)
        return contracted_relations, contracted_nonzero, contracted_indep, contracted_dep

    def get_elements_info(self, rank: int, axiality: bool, expr: str | None = None) -> None: # N階の極性テンソルがR∈self.unitary_matricesの対称操作で不変となるときの非ゼロになりうる要素の添字を計算
        """Determine which elements of a tensor are equivalent or zero by straightforward exact calculation based on Neumann's principle.

        Note:
            All analysis results are output to stdout.

        Args:
            rank (int): Rank of target physical property tensor.
            axiality (bool): True if the tensor is an axial tensor.
            expr (str | None): String representing a relation between elements that is already known. Some relations shall be separated by comma.

        Returns: 
            (list[tuple[int, ...]]): Indices(0-indexed) of non-zero elements of the tensor.

        Examples:
            >>> point_group_name = spacegroup_to_pointgroup("Fd-3m") # Pyrochlore
            >>> assert point_group_name == "m-3m"
            >>> PPTA = PhysicalPropertyTensorAnalyzer(point_group_name)
            >>> PPTA.get_elements_info(rank=4, axiality=False, expr="ijkl=ijlk=jikl=klij") # elastic modulus tensor (4-tensor): 4階の弾性率テンソル
            >>> PPTA.get_elements_info(rank=3, axiality=False, expr="ijk=ikj") # Optical Parametric Oscillator: 光パラメトリック発振

        """
        n_elem: int = 3**rank
        nonzero: set[int] = set(range(n_elem))
        relations: Relations = []
        if expr is not None: # 添字で直接表現したテンソル要素間の対称性を元に関係式を構築
            relations.extend(self._relations_from_expression(rank, expr))
        # print(self._relations_to_ternary(relations, rank))

        for R in self.unitary_matrices: # (結晶点群に属する)直交変換を元にテンソル要素間の関係式を構築
            relations.extend(self._relations_from_symmetry_operation(rank, R, axiality))

        # print("aaa",self._relations_to_ternary(relations, rank))

        def printer(relations: Relations) -> None:
            # デバッグ用
            print(*self._formulize_relations(relations, rank), sep="\n")
        
        flag: bool = True
        while flag:
            flag1: bool = True
            flag2: bool = True
            flag3: bool = True
            while flag1:
                relations, nonzero, flag1 = self._delete_zero_term(relations,nonzero) # 無効な式を削除
            relations, nonzero, flag2 = self._untangle_relations(rank,relations,nonzero) # 行簡約化で関係式を簡単化
            relations, nonzero, flag3 = self._remove_duplicate(relations,nonzero) # 重複した等価な式たちを一本化
            flag = flag1 or flag2 or flag3
            relations = self._simplify_relations_value(relations) # 関係式の係数を簡単な比に変換

        # 独立成分と従属成分を分離
        indep, dep_represented_by_indep = self._extract_independent(rank, relations, nonzero)
        res1: list[tuple[int, ...]] = sorted([self._ternary(ijk, rank) for ijk in indep])
        res2: list[str] = [self._represent_dep_by_indep(rank, relation) for relation in dep_represented_by_indep]
        
        print()
        print(f"number of nonzero elements: {len(nonzero)}")
        res0: list[tuple[int, ...]] = sorted([self._ternary(ijk, rank) for ijk in nonzero])
        print(f"nonzero elements: {res0}")
        print()
        # print(self._nonzero_matrix(nonzero,0,rank))
        # print(self._nonzero_matrix(nonzero,1,rank))
        # print(self._nonzero_matrix(nonzero,2,rank))
        print(f"number of independent elements: {len(res1)}")
        print(f"--independent elements--")
        print(*res1, sep="\n")
        print(f"------------------------")
        print()
        print(f"--dependent elements represented by indp.--")
        print(*res2, sep="\n")
        print(f"-------------------------------------------")
        return
    
    def get_elements_info_from_jahn_symbol(self, jahn_symbol: str) -> None: # N階の極性テンソルがR∈self.unitary_matricesの対称操作で不変となるときの非ゼロになりうる要素の添字を計算
        """Determine which elements of a tensor are equivalent or zero by straightforward exact calculation based on Neumann's principle.

        Note:
            All analysis results are output to stdout.

        Args:
            jahn_symbol (str): String designating the sort of tensor according to its transformation properties.

        Returns: 
            (list[tuple[int, ...]]): Indices(0-indexed) of non-zero elements of the tensor.

        Examples:
            >>> point_group_name = spacegroup_to_pointgroup("Fd-3m") # Pyrochlore
            >>> assert point_group_name == "m-3m"
            >>> PPTA = PhysicalPropertyTensorAnalyzer(point_group_name)
            >>> PPTA.get_elements_info(rank=4, axiality=False, expr="ijkl=ijlk=jikl=klij") # elastic modulus tensor (4-tensor): 4階の弾性率テンソル
            >>> PPTA.get_elements_info(rank=3, axiality=False, expr="ijk=ikj") # Optical Parametric Oscillator: 光パラメトリック発振

        """
        rank, expr, axiality, time_reversality = decompose_jahn_symbol(jahn_symbol)
        n_elem: int = 3**rank
        nonzero: set[int] = set(range(n_elem))
        relations: Relations = []
        # 添字で直接表現したテンソル要素間の対称性を元に関係式を構築
        relations.extend(self._relations_from_expression(rank, expr))
        # print(self._relations_to_ternary(relations, rank))

        for R in self.unitary_matrices: # (結晶点群に属する)直交変換を元にテンソル要素間の関係式を構築
            relations.extend(self._relations_from_symmetry_operation(rank, R, axiality))

        # print("aaa",self._relations_to_ternary(relations, rank))

        def printer(relations: Relations) -> None:
            # デバッグ用
            print(*self._formulize_relations(relations, rank), sep="\n")
        
        flag: bool = True
        while flag:
            flag1: bool = True
            flag2: bool = True
            flag3: bool = True
            while flag1:
                relations, nonzero, flag1 = self._delete_zero_term(relations,nonzero) # 無効な式を削除
            relations, nonzero, flag2 = self._untangle_relations(rank,relations,nonzero) # 行簡約化で関係式を簡単化
            relations, nonzero, flag3 = self._remove_duplicate(relations,nonzero) # 重複した等価な式たちを一本化
            flag = flag1 or flag2 or flag3
            relations = self._simplify_relations_value(relations) # 関係式の係数を簡単な比に変換

        # 独立成分と従属成分を分離
        indep, dep_represented_by_indep = self._extract_independent(rank, relations, nonzero)
        res1: list[tuple[int, ...]] = sorted([self._ternary(ijk, rank) for ijk in indep])
        res2: list[str] = [self._represent_dep_by_indep(rank, relation) for relation in dep_represented_by_indep]
        
        print()
        print(f"number of nonzero elements: {len(nonzero)}")
        res0: list[tuple[int, ...]] = sorted([self._ternary(ijk, rank) for ijk in nonzero])
        print(f"nonzero elements: {res0}")
        print()
        # print(self._nonzero_matrix(nonzero,0,rank))
        # print(self._nonzero_matrix(nonzero,1,rank))
        # print(self._nonzero_matrix(nonzero,2,rank))
        print(f"number of independent elements: {len(res1)}")
        print(f"--independent elements--")
        print(*res1, sep="\n")
        print(f"------------------------")
        print()
        print(f"--dependent elements represented by indp.--")
        print(*res2, sep="\n")
        print(f"-------------------------------------------")
        return


    def get_info_transport_tensor_under_magnetic_field(self, magnetic_field_dependence_dimension: int = 0, is_0indexed: bool = False) -> None: # 指定された点群上の2階の輸送テンソルの非ゼロになりうる要素の添字を計算
        """Determine which elements of a transport tensor are equivalent or zero by straightforward exact calculation based on Neumann's principle.

        Note:
            All analysis results are output to stdout.
        
        Args:
            magnetic_field_dependence_dimension (int): 0 or 1 or 2 or 3 or 4 are allowed.
                0: constant components in magnetic field.
                1: linear components in magnetic field.
                2: quadratic components in magnetic field.
                3: cubic components in magnetic field.
                4: quartic components in magnetic field.
            is_0indexed (bool): True if the indices of tensor elements in the results are written in 0-indexed form. Defaults to False.

        Examples:
            >>> point_group_name = spacegroup_to_pointgroup("Fd-3m") # Diamond
            >>> assert point_group_name == "m-3m"
            >>> PPTA = PhysicalPropertyTensorAnalyzer(point_group_name)
            >>> PPTA.get_info_transport_tensor_under_magnetic_field(1) # odd terms of a transport tensor

        ToDo:
            implement Jahn's symbol analyzer to determine automatically 'rank', 'expr' and 'axiality'.
        """
        rank: int
        expr: str
        axiality: bool
        if magnetic_field_dependence_dimension == 0:
            # 磁場に対してConstant
            rank = 2
            expr = ""
            axiality = False
        elif magnetic_field_dependence_dimension == 1:
            # 磁場に対してOdd
            rank = 3
            expr = "ijk=-jik"
            axiality = True
        elif magnetic_field_dependence_dimension == 2:
            # 磁場に対してEven
            rank = 4
            expr = "ijkl=jikl"
            axiality = False
        elif magnetic_field_dependence_dimension == 3:
            # 磁場に対してOdd
            rank = 5
            expr = "ijklm=-jiklm"
            axiality = True
        elif magnetic_field_dependence_dimension == 4:
            # 磁場に対してEven
            rank = 6
            expr = "ijklmn=jiklmn"
            axiality = False
        else:
            raise ValueError()
        
        def printer(relations: Relations) -> None:
            # デバッグ用
            print(*self._formulize_relations(relations, rank), sep="\n")
        
        def printer_nonzero(nonzero: set[int], rank: int) -> None:
            # デバッグ用
            print(*sorted([self._ternary(i,rank) for i in nonzero]), sep="\n")
        

        n_elem: int = 3**rank
        nonzero: set[int] = set(range(n_elem))
        relations: Relations = []
        if expr is not None: # 添字で直接表現したテンソル要素間の対称性を元に関係式を構築
            relations.extend(self._relations_from_expression(rank, expr))

        for R in self.unitary_matrices: # (結晶点群に属する)直交変換を元にテンソル要素間の関係式を構築
            # print(R.name)
            # print(self._untangle_relations(rank,self._relations_from_expression(rank, expr)+self._relations_from_symmetry_operation(rank, R, axiality),set(range(27))))
            # print(*self._formulize_relations(self._extract_relation(self._relations_from_symmetry_operation(rank, R, axiality), rank, (0,1,1)), rank), sep="\n")
            relations.extend(self._relations_from_symmetry_operation(rank, R, axiality))

        print("### initial process: done")

        flag: bool = True
        while flag:
            flag1: bool = True
            flag2: bool = True
            flag3: bool = True
            while flag1:
                relations, nonzero, flag1 = self._delete_zero_term(relations,nonzero) # 無効な式を削除
            relations, nonzero, flag2 = self._untangle_relations(rank,relations,nonzero) # 行簡約化で関係式を簡単化
            relations, nonzero, flag3 = self._remove_duplicate(relations,nonzero) # 重複した等価な式たちを一本化
            flag = flag1 or flag2 or flag3
            relations = self._simplify_relations_value(relations) # 関係式の係数を簡単な比に変換
        print("### simplification: done")
        printer(relations)
        # printer_nonzero(nonzero, rank)

        for direction0 in range(3):
            contracted_relations, contracted_nonzero, contracted_indep, contracted_dep = self._contract_along_with_direction(relations, rank, nonzero, direction0)
            contracted_relations, _, _ = self._remove_duplicate(contracted_relations, contracted_nonzero)

            result_rank: int = 3 # 足はijkで表示
            res0: list[tuple[int, ...]] = sorted([self._ternary(ijk, result_rank, is_0indexed=is_0indexed) for ijk in contracted_nonzero])
            res1: list[tuple[int, ...]] = sorted([self._ternary(ijk, result_rank, is_0indexed=is_0indexed) for ijk in contracted_indep])
            res2: list[str] = [self._represent_dep_by_indep(result_rank, relation, is_0indexed=is_0indexed) for relation in contracted_relations]
            
            direction1: int
            if not is_0indexed:
                direction1 = direction0 + 1 # 1-indexed
            else:
                direction1 = direction0 # 0-indexed
            print(f"###### magnetic field direction: B_{direction1} ######")
            print(fr"tensor element: $\rho_{{ij}} = \lambda_{{ij{direction1}}} B_{direction1}^{magnetic_field_dependence_dimension}$")
            print(f"number of nonzero elements: {len(res0)}")
            print(fr"nonzero elements of $\lambda$: {res0}")
            print(self._nonzero_matrix(nonzero,direction0,rank))
            print(f"number of independent elements: {len(res1)}")
            print(f"--independent elements--")
            print(*res1, sep="\n")
            print(f"--dependent elements represented by indp.--")
            print(*res2, sep="\n")
            print("###################################################")
            print()


        return 


def main() -> None:
    pass

if __name__ == "__main__":
    main()

