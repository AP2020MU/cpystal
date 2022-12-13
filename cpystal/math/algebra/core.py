"""`cpystal.math.core` is a module for mathematical objects.

Functions:
    `visualize_spherical_harmonics`

Classes:
    `REF`
    `MatrixREF`
    `IREF`
    `PolyInt`
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import reduce
from itertools import product
from math import factorial, gcd
from typing import Any, Iterable, Iterator, List, overload, TypeVar

import matplotlib.pyplot as plt # type: ignore
from matplotlib.widgets import Slider, Button # type: ignore
import numpy as np
import numpy.typing as npt
from scipy.special import sph_harm # type: ignore


REFchild = TypeVar("REFchild", bound="REF")
MatrixREFchild = TypeVar("MatrixREFchild", bound="MatrixREF")

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


class IREF:
    """Infinite Rational Extention Field.

    The set of all rational numbers plus the square roots of all square-free integers 
    forms a field with the usual addition, subtraction, and division operations.
    An instance of this class corresponds to an element of this Infinite Rational Extention Field (IREF).

    The following types of the argument of the constructor of this class `iref_number` are allowed:
        `int`, `Fraction`, `tuple[int, int | Fraction]`, `list[tuple[int, int | Fraction]]`, or `defaultdict[int, Fraction]`.
    
    The keys of the attribute `root_value_dict` are square roots of square-free integers in IREF.
    The values of the attribute `root_value_dict` are the coefficients of the keys which represent square roots of square-free integers.

    Examples:
    ```
    print(IREF())
    >>> 0
    a = IREF([(2,1), (3,-1)]            # √2-√3
    a += IREF((-1,Fraction(4,3))) * 3   # a == √2-√3+4√-1
    a[5] = Fraction(1,2)                # a == √2-√3+4√-1+1/2√5
    b = IREF(3)                         # 3
    print(a * b)
    >>> 3√2 + -3√3 + 12√-1 + 3/2√5
    print(a.root_value_dict)
    >>> defaultdict(<class 'fractions.Fraction'>, {2: Fraction(1, 1), 3: Fraction(-1, 1), -1: Fraction(4, 1), 5: Fraction(1, 2)})
    print(b.root_value_dict)
    >>> defaultdict(<class 'fractions.Fraction'>, {1: Fraction(3, 1)})
    ```
    """
    def __make_square_factor_list(n: int = 10**5) -> tuple[list[int], list[int], list[int]]: # type: ignore
        """Calculate a list of square factor.

        Note:
            Time complexity: Θ(nloglogn).

        Args:
            n (int, optional): Non-negative integer. Defaults to 10**5.

        Returns:
            tuple[list[int], list[int], list[int]]: 
                List of smallest prime factor. The i-th element represents the smallest prime factor of i.
                List of square factor. The i-th element represents the square factor of i.
                List of prime number.

        """
        # spf[i] := Smallest Prime Factor of i. ([0,0,2,3,2,5,2,7,...])
        # __square_factor[i] := Square Factor of i. ([1,1,1,1,2,1,1,1,2,3,...])
        def isqrt(n: int) -> int: # floor(sqrt(n)): Θ(logn)
            if n < 0:
                raise ValueError
            ok: int = 0
            ng: int = n+1
            while abs(ok-ng) > 1:
                mid = (ng+ok)//2
                if mid**2 <= n:
                    ok = mid
                else:
                    ng = mid
            return ok
        spf: list[int] = list(range(n+1))
        spf[0] = spf[1] = 0
        square_factor: list[int] = [1] * (n+1)
        primes: list[int] = []
        for i in range(1, isqrt(n)+1):
            if spf[i] == i:
                primes.append(i)
                for j in range(i**2,n+1,i):
                    spf[j] = i
                k: int = 1
                while i ** (2*k) < n+1:
                    for j in range(i ** (2*k), n+1, i ** (2*k)):
                        square_factor[j] *= i
                    k += 1
        return spf, square_factor, primes
    __smallest_prime_factor, __square_factor, __primes = __make_square_factor_list()

    def __init__(self, iref_number: int | Fraction | tuple[int, int | Fraction] | list[tuple[int, int | Fraction]] | defaultdict[int, Fraction] = 0) -> None:
        self._d: defaultdict[int, Fraction] = defaultdict(Fraction)
        message: str = "type of 'iref_number' must be `int`, `Fraction`, `tuple[int, int | Fraction]`, `list[tuple[int, int | Fraction]]`, or `defaultdict[int, Fraction]`"
        sf: int
        if isinstance(iref_number, int) or isinstance(iref_number, Fraction):
            self._d[1] = Fraction(iref_number)
        elif isinstance(iref_number, tuple):
            root, val = iref_number
            if isinstance(root, int):
                sf = self.cal_square_factor(abs(root))
                if isinstance(val, int):
                    self._d[root//(sf**2)] = Fraction(val) * sf
                elif isinstance(val, Fraction):
                    self._d[root//(sf**2)] = val * sf
                else:
                    raise TypeError(message)
            else:
                raise TypeError(message)
        elif isinstance(iref_number, list):
            for root, val in iref_number:
                if isinstance(root, int):
                    sf = self.cal_square_factor(abs(root))
                    if isinstance(val, int):
                        self._d[root//(sf**2)] = Fraction(val) * sf
                    elif isinstance(val, Fraction):
                        self._d[root//(sf**2)] = val * sf
                    else:
                        raise TypeError(message)
                else:
                    raise TypeError(message)
        elif isinstance(iref_number, defaultdict):
            for root, val in iref_number.items():
                sf = self.cal_square_factor(abs(root))
                if isinstance(root, int) and isinstance(val, Fraction):
                    self._d[root//(sf**2)] = val * sf
                else:
                    raise TypeError(message)
        else:
            raise TypeError(message)

    def __str__(self) -> str:
        res: str = ""
        for root, val in self._d.items():
            if val != 0:
                if root == 1:
                    res = res + f"{val} + "
                else:
                    if val == 1:
                        res = res + f"√{root} + "
                    else:
                        res = res + f"{val}√{root} + "
        res = res.strip("+ ")
        if len(res) == 0:
            return "0"
        else:
            return res
    
    def __repr__(self) -> str:
        res: str = ""
        for root, val in self._d.items():
            if val != 0:
                if root == 1:
                    res = res + f"{val} + "
                else:
                    if val == 1:
                        res = res + f"√{root} + "
                    else:
                        res = res + f"{val}√{root} + "
        res = res.strip("+ ")
        if len(res) == 0:
            return "0"
        else:
            return res

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IREF):
            if all([v == other._d[root] for root, v in self._d.items()]) and \
                all([v == self._d[root] for root, v in other._d.items()]):
                return True
            else:
                return False
        else:
            return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
    
    def __pos__(self) -> IREF:
        return self.__class__(self._d)

    def __neg__(self) -> IREF:
        new: IREF = self.__class__()
        for root, val in self._d.items():
            self._d[root] = -val
        return new

    def __lt__(self, other: object) -> bool:
        if isinstance(other, IREF):
            return self.to_float() < other.to_float()
        elif isinstance(other, int) or isinstance(other, float):
            return self.to_float() < other
        else:
            raise TypeError(f"'<' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __le__(self, other: object) -> bool:
        if isinstance(other, IREF):
            return self.to_float() <= other.to_float()
        elif isinstance(other, int) or isinstance(other, float):
            return self.to_float() <= other
        else:
            raise TypeError(f"'<=' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __gt__(self, other: object) -> bool:
        if isinstance(other, IREF):
            return self.to_float() > other.to_float()
        elif isinstance(other, int) or isinstance(other, float):
            return self.to_float() > other
        else:
            raise TypeError(f"'>' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __ge__(self, other: object) -> bool:
        if isinstance(other, IREF):
            return self.to_float() >= other.to_float()
        elif isinstance(other, int) or isinstance(other, float):
            return self.to_float() >= other
        else:
            raise TypeError(f"'>=' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")

    def __add__(self, other: object) -> IREF:
        new: IREF = IREF(self._d)
        if isinstance(other, IREF):
            for root_other, val_other in other._d.items():
                new._d[root_other] += val_other
            return new
        elif isinstance(other, int):
            new._d[1] += Fraction(other)
            return new
        elif isinstance(other, Fraction):
            new._d[1] += other
            return new
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __radd__(self, other: object) -> IREF:
        new: IREF = IREF(self._d)
        if isinstance(other, IREF):
            for root_other, val_other in other._d.items():
                new._d[root_other] += val_other
            return new
        elif isinstance(other, int):
            new._d[1] += Fraction(other)
            return new
        elif isinstance(other, Fraction):
            new._d[1] += other
            return new
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __iadd__(self, other: object) -> IREF:
        if isinstance(other, IREF):
            for root_other, val_other in other._d.items():
                self._d[root_other] += val_other
            return self
        elif isinstance(other, int):
            self._d[1] += Fraction(other)
            return self
        elif isinstance(other, Fraction):
            self._d[1] += other
            return self
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")

    def __sub__(self, other: object) -> IREF:
        new: IREF = IREF(self._d)
        if isinstance(other, IREF):
            for root_other, val_other in other._d.items():
                new._d[root_other] -= val_other
            return new
        elif isinstance(other, int):
            new._d[1] -= Fraction(other)
            return new
        elif isinstance(other, Fraction):
            new._d[1] -= other
            return new
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __rsub__(self, other: object) -> IREF:
        new: IREF = IREF(self._d)
        if isinstance(other, IREF):
            for root_other, val_other in other._d.items():
                new._d[root_other] -= val_other
            return new
        elif isinstance(other, int):
            new._d[1] -= Fraction(other)
            return new
        elif isinstance(other, Fraction):
            new._d[1] -= other
            return new
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __isub__(self, other: object) -> IREF:
        if isinstance(other, IREF):
            for root_other, val_other in other._d.items():
                self._d[root_other] -= val_other
            return self
        elif isinstance(other, int):
            self._d[1] -= Fraction(other)
            return self
        elif isinstance(other, Fraction):
            self._d[1] -= other
            return self
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")

    def __mul__(self, other: object) -> IREF:
        new: IREF = IREF()
        if isinstance(other, IREF):
            for root, val in self._d.items():
                for root_other, val_other in other._d.items():
                    sign: int = 1
                    if root < 0 and root_other < 0:
                        sign = -1
                    g: int = gcd(abs(root), abs(root_other))
                    new._d[root*root_other//(g**2)] += sign * g * val * val_other
            return new
        elif isinstance(other, int):
            for root, val in self._d.items():
                new._d[root] += val * Fraction(other)
            return new
        elif isinstance(other, Fraction):
            for root, val in self._d.items():
                new._d[root] += val * other
            return new
        else:
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")

    def __rmul__(self, other: object) -> IREF:
        new: IREF = IREF()
        if isinstance(other, IREF):
            for root, val in self._d.items():
                for root_other, val_other in other._d.items():
                    sign: int = 1
                    if root < 0 and root_other < 0:
                        sign = -1
                    g: int = gcd(abs(root), abs(root_other))
                    new._d[root*root_other//(g**2)] += sign * g * val * val_other
            return new
        elif isinstance(other, int):
            for root, val in self._d.items():
                new._d[root] += val * Fraction(other)
            return new
        elif isinstance(other, Fraction):
            for root, val in self._d.items():
                new._d[root] += val * other
            return new
        else:
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")

    def __imul__(self, other: object) -> IREF:
        new: IREF = IREF()
        if isinstance(other, IREF):
            for root, val in self._d.items():
                for root_other, val_other in other._d.items():
                    sign: int = 1
                    if root < 0 and root_other < 0:
                        sign = -1
                    g: int = gcd(abs(root), abs(root_other))
                    new._d[root*root_other//(g**2)] += sign * g * val * val_other
            self._d = new._d
            return self
        elif isinstance(other, int):
            for root, val in self._d.items():
                new._d[root] += val * Fraction(other)
            self._d = new._d
            return self
        elif isinstance(other, Fraction):
            for root, val in self._d.items():
                new._d[root] += val * other
            self._d = new._d
            return self
        else:
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")

    def __truediv__(self, other: object) -> IREF:
        new: IREF
        if isinstance(other, IREF):
            if len(other) == 0:
                raise ValueError
            elif len(other) == 1:
                root_other, val_other = list(other._d.items())[0]
                return IREF(self._d) * IREF((root_other, 1/val_other/root_other))
            else:
                irefs: list[IREF] = [IREF(rv) for rv in other._d.items()]
                a0: IREF = irefs[0]
                nume: IREF = IREF(self._d)
                deno: IREF = IREF(other._d)
                for i, sign in enumerate(product([1,-1], repeat=len(other)-1)):
                    if i == 0:
                        continue
                    poly: IREF = sum([a0]+[a*i for i, a in zip(sign, irefs[1:])], start=IREF())
                    nume *= poly
                    deno *= poly
                res: IREF = IREF()
                for root, val in nume._d.items():
                    res._d[root] = val / deno._d[1]
                return res
        elif isinstance(other, int):
            new = IREF()
            for root, val in self._d.items():
                new._d[root] = val / Fraction(other)
            return new
        elif isinstance(other, Fraction):
            new = IREF()
            for root, val in self._d.items():
                new._d[root] = val / other
            return new
        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(self).__name__}' and '{type(other).__name__}'")

    def __rtruediv__(self, other: object) -> IREF:
        new: IREF
        if isinstance(other, IREF):
            if len(other) == 0:
                raise ValueError
            elif len(other) == 1:
                root_other, val_other = list(other._d.items())[0]
                return IREF(self._d) * IREF((root_other, 1/val_other/root_other))
            else:
                irefs: list[IREF] = [IREF(rv) for rv in other._d.items()]
                a0: IREF = irefs[0]
                nume: IREF = IREF(self._d)
                deno: IREF = IREF(other._d)
                for i, sign in enumerate(product([1,-1], repeat=len(other)-1)):
                    if i == 0:
                        continue
                    poly: IREF = sum([a0]+[a*i for i, a in zip(sign, irefs[1:])], start=IREF())
                    nume *= poly
                    deno *= poly
                res: IREF = IREF()
                for root, val in nume._d.items():
                    res._d[root] = val / deno._d[1]
                return res
        elif isinstance(other, int):
            new = IREF()
            for root, val in self._d.items():
                new._d[root] = val / Fraction(other)
            return new
        elif isinstance(other, Fraction):
            new = IREF()
            for root, val in self._d.items():
                new._d[root] = val / other
            return new
        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(self).__name__}' and '{type(other).__name__}'")

    def __itruediv__(self, other: object) -> IREF:
        if isinstance(other, IREF):
            if len(other) == 0:
                raise ValueError
            elif len(other) == 1:
                root_other, val_other = list(other._d.items())[0]
                return IREF(self._d) * IREF((root_other, 1/val_other/root_other))
            else:
                irefs: list[IREF] = [IREF(rv) for rv in other._d.items()]
                a0: IREF = irefs[0]
                nume: IREF = IREF(self._d)
                deno: IREF = IREF(other._d)
                for i, sign in enumerate(product([1,-1], repeat=len(other)-1)):
                    if i == 0:
                        continue
                    poly: IREF = sum([a0]+[a*i for i, a in zip(sign, irefs[1:])], start=IREF())
                    nume *= poly
                    deno *= poly
                res: IREF = IREF()
                for root, val in nume._d.items():
                    res._d[root] = val / deno._d[1]
                self._d = res._d
                return self
        elif isinstance(other, int):
            for root, val in self._d.items():
                self._d[root] = val / other
            return self
        elif isinstance(other, Fraction):
            for root, val in self._d.items():
                self._d[root] = val / other
            return self
        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(self).__name__}' and '{type(other).__name__}'")

    def __iter__(self) -> Iterator[tuple[int, int | Fraction]]:
        return iter([(root, val) for root, val in self._d.items() if val != 0])
    
    def __len__(self) -> int:
        return len([(root, val) for root, val in self._d.items() if val != 0])
    
    def __getitem__(self, key: int) -> Fraction:
        return self._d[key]
    
    def __setitem__(self, key: int, value: Fraction) -> None:
        if not (isinstance(key, int) and (isinstance(value, int) or isinstance(value, Fraction))):
            raise TypeError(f"key must be `int` type and value must be `Fraction` type")
        g: int = self.cal_square_factor(abs(key))
        if isinstance(value, int):
            self._d[key//(g**2)] = g * Fraction(value)
        else:
            self._d[key//(g**2)] = g * value
        
    def __copy__(self) -> IREF:
        return self

    def __deepcopy__(self) -> IREF:
        return self.__class__(self._d)

    @property
    def root_value_dict(self) -> defaultdict[int, Fraction]:
        return self._d
    
    def copy(self) -> IREF:
        return self
    
    def deepcopy(self) -> IREF:
        return self.__class__(self._d)
    
    def to_float(self) -> float:
        res: float = 0
        for root, val in self._d.items():
            if root < 0:
                raise ValueError(f"negative root: {root}; to get the value of this IREF instance, use 'to_complex'")
            res += val.numerator / val.denominator * (root**0.5)
        return res

    def to_complex(self) -> complex:
        res: float = 0
        for root, val in self._d.items():
            res += val.numerator / val.denominator * (root**0.5)
        return complex(res)

    @staticmethod
    def isqrt(n: int) -> int:
        """Calculate floor(sqrt(n)) by binary search.

        Note:
            Time complexity: Θ(logn).

        Args:
            n (int): Non-negative integer.

        Raises:
            ValueError: Argument `n` must be a non-negative integer.

        Returns:
            int: floor(sqrt(n)).
        """
        if n < 0:
            raise ValueError
        ok: int = 0
        ng: int = n+1
        while abs(ok-ng) > 1:
            mid = (ng+ok)//2
            if mid**2 <= n:
                ok = mid
            else:
                ng = mid
        return ok

    @classmethod
    def make_square_factor_list(cls, n: int = 10**5) -> list[int]:
        """Calculate a list of square factor.

        Note:
            Time complexity: Θ(nloglogn).

        Args:
            n (int, optional): Non-negative integer. Defaults to 10**5.

        Returns:
            list[int]: List of square factor. The i-th element represents the square factor of i.
        """
        # spf[i] := Smallest Prime Factor of i. ([0,0,2,3,2,5,2,7,...])
        # __square_factor[i] := Square Factor of i. ([1,1,1,1,2,1,1,1,2,3,...])
        spf: list[int] = list(range(n+1))
        spf[0] = spf[1] = 0
        __square_factor: list[int] = [1] * (n+1)
        for i in range(1, cls.isqrt(n)+1):
            if spf[i] == i:
                for j in range(i**2, n+1, i):
                    spf[j] = i
                k: int = 1
                while i**(2*k) < n+1:
                    for j in range(i**(2*k), n+1, i**(2*k)):
                        __square_factor[j] *= i
                    k += 1
        return __square_factor

    @classmethod
    def factorize_fast(cls, n: int) -> defaultdict[int, int]:
        res: defaultdict[int, int] = defaultdict(int)
        idx: int = 0
        p: int = cls.__primes[idx]
        while p**2 < n:
            e: int = 0
            while n % p == 0:
                n //= p
                e += 1
            if e != 0:
                res[p] = e
                idx += 1
            p = cls.__primes[idx]
        if n > 1:
            res[n] = 1
        return res

    @staticmethod
    def factorize(n: int) -> defaultdict[int, int]:
        p: int = 2
        m: int = 1
        sign: int = -1
        table: defaultdict[int, int] = defaultdict(int)
        while p * p <= n:
            e: int = 0
            while n % p == 0:
                n //= p
                e += 1
            if e != 0:
                table[p] = e
                
            if p >= 5:
                if sign > 0:
                    m += 1
                sign *= -1
                p = 6*m + sign
            elif p == 2: p = 3
            elif p == 3: p = 5
        if n > 1:
            table[n] = 1
        return table
    
    @classmethod
    def cal_square_factor(cls, n: int) -> int:
        res: int
        if n < len(cls.__square_factor):
            return cls.__square_factor[n]
        elif n < cls.__primes[-1] ** 2:
            res = 1
            for k, v in cls.factorize_fast(n).items():
                res *= k ** (v//2)
            return res
        else:
            res = 1
            for k, v in cls.factorize(n).items():
                res *= k ** (v//2)
            return res


class PolyInt:
    """Polynomial with integer coefficients.

    Attributes:
        P (list[int] | npt.NDArray, optional): List of coefficients of polynomial.
    """
    def __init__(self, P: list[int] | npt.NDArray = []):
        """Constructor.

        Args:
            P (list[int] | npt.NDArray, optional): List of coefficients of polynomial. Defaults to [].
        
        Note:
            P(x) := \sum_{i} P[i] * x**i
        """
        self.P: npt.NDArray = np.array(P) # P:np.ndarray[int,...] P[i]==多項式P[X]のX**iの係数

    def __pos__(self):
        return self.__class__(self.P)

    def __neg__(self):
        return self.__class__(-self.P)

    def __str__(self):
        return str(self.P)
    
    def __repr__(self):
        return str(self.P)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PolyInt):
            return bool(np.all(self.P == other.P))
        else:
            return False

    def __add__(self, other: object) -> PolyInt: # P[X]+Q[X]: 和
        if isinstance(other, PolyInt):
            return self.__class__(np.array(np.poly1d(self.P[::-1]) + np.poly1d(other.P[::-1]))[::-1])
        elif isinstance(other, int) or isinstance(other, float):
            return self.__class__(self.P[0] + other)
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")

    def __sub__(self, other: object) -> PolyInt: # P[X]-Q[X]: 差
        if isinstance(other, PolyInt):
            return self.__class__(np.array(np.poly1d(self.P[::-1]) - np.poly1d(other.P[::-1]))[::-1])
        elif isinstance(other, int) or isinstance(other, float):
            return self.__class__(self.P[0] - other)
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")

    def __mul__(self, other: object) -> PolyInt: # P[X]*Q[X]: 積
        if isinstance(other, PolyInt):
            return self.__class__(np.array(np.poly1d(self.P[::-1]) * np.poly1d(other.P[::-1]))[::-1])
        elif isinstance(other, int) or isinstance(other, float):
            return self.__class__(self.P * other)
        else:
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __truediv__(self, other: object) -> PolyInt: # P[X]/Q[X] の商を返す
        if isinstance(other, PolyInt):
            n: int = self.cal_deg()
            m: int = other.cal_deg()
            if n < m:
                return self.__class__(self.P)
            else:
                new: PolyInt = self.__class__(self.P)
                Q: list = []
                for i in range(n-m+1):
                    q: npt.NDArray = new.P[n-1-i] / other.P[m-1]
                    new.P[n-m-i:n-i] = new.P[n-m-i:n-i] - q*other.P
                    Q.append(q)
                return self.__class__(np.array(Q[::-1])) # 商Q[X]
        elif isinstance(other, int):
            return self.__class__(self.P // other)
        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(self).__name__}' and '{type(other).__name__}'")

    def __mod__(self, other: object) -> PolyInt: # P[X]/Q[X] の剰余R[X]を返す
        if isinstance(other, PolyInt):
            n: int = self.cal_deg()
            m: int = other.cal_deg()
            if n < m:
                return self.__class__(self.P)
            else:
                new: PolyInt = self.__class__(self.P)
                for i in range(n-m+1):
                    q: npt.NDArray = new.P[n-1-i] / other.P[m-1]
                    new.P[n-m-i:n-i] = new.P[n-m-i:n-i] - q*other.P
                return self.__class__(new.P[:np.max(np.where(new.P!=0))+1]) # 剰余R[X]
        else:
            raise TypeError(f"unsupported operand type(s) for %: '{type(self).__name__}' and '{type(other).__name__}'")
        

    def __divmod__(self, other: object) -> tuple[PolyInt, PolyInt]: # P[X]/Q[X] の商と剰余R[X]を返す
        if isinstance(other, PolyInt):
            n: int = self.cal_deg()
            m: int = other.cal_deg()
            if n < m:
                return self.__class__([0]), self.__class__(self.P)
            else:
                new: PolyInt = self.__class__(self.P)
                Q: list = []
                for i in range(n-m+1):
                    q: npt.NDArray = new.P[n-1-i] / other.P[m-1]
                    new.P[n-m-i:n-i] = new.P[n-m-i:n-i] - q*other.P
                    Q.append(q)
                return self.__class__(np.array(Q[::-1])), self.__class__(new.P[:np.max(np.where(new.P!=0))+1]) # 商Q[X], 剰余R[X]
        else:
            raise TypeError(f"unsupported operand type(s) for __divmod__: '{type(self).__name__}' and '{type(other).__name__}'")
        
    def cal_deg(self) -> int:
        """Calculate maximum degree of the polynomial.

        Returns:
            int: Maximum degree of the polynomial.
        
        Note:
            The polynomial is shortened by excuting this method to the maximum degree.
        """
        if len(np.where(self.P!=0)[0]) == 0:
            return 0
        n: int = np.max(np.where(self.P!=0)) + 1
        self.P = self.P[:n]
        return n

    def deriv(self) -> PolyInt:
        """Derivative of the polynomial.

        Returns:
            PolyInt: Derivative of the polynomial.
        """
        return self.__class__(np.array([self.P[i]*i for i in range(1,len(self.P))]))

    def subs(self, x: float) -> float:
        """Substitute a value for the variable in the polynomial.

        Args:
            x (float): Value of the variable.

        Returns:
            float: Result of substituting a given value for the variable.
        """
        return np.sum([(self.P[i]*x**i) for i in range(len(self.P))])
    
    def simplify(self) -> tuple[PolyInt, int]:
        """Simplify the polynomial by dividing common factor.

        Returns:
            tuple[PolyInt, int]: Simplified polynomial and the common factor.
        """
        g: int = reduce(gcd, self.P)
        return self.__class__(self.P // g), g
    
    @staticmethod
    def fac2(n: int) -> int:
        """Calculate double factorial.

        Args:
            n (int): Non-negative integer.

        Returns:
            int: Double factorial n!!.
        """
        ans: int = 1
        for i in range(n, 0, -2):
            ans *= i
        return ans

    @classmethod
    def binomial_coeff(cls, n: int, m: int) -> int:
        """Value of binomial coefficient nCm.

        Args:
            n (int): Integer.
            m (int): Integer.

        Returns:
            int: Binomial coefficient nCm.
        """
        if n < m or n < 0 or m < 0:
            return 0
        return factorial(n) // factorial(m) // factorial(n-m)
    
    @classmethod
    def Pn_coeff(cls, n: int, m: int) -> int:
        """Coefficient of x^m in Legendre polynomial P_n(x)*2**n.

        Args:
            n (int): Degree of Legendre polynomial P_n(x)*2**n.
            m (int): Degree of x.

        Returns:
            float: Coefficient of x^m in Legendre polynomial P_n(x)*2**n.
        """
        if (n-m) % 2 or n < m or n < 0 or m < 0:
            return 0
        l: int = (n-m)//2
        return (-1)**l * factorial(2*n-2*l) // factorial(l) // factorial(n-l) // factorial(n-2*l)
    
    @classmethod
    def Plm_coeff(cls, l: int, m: int, k: int) -> int:
        """Coefficient of x^k in Associated Legendre polynomial P_l^m(x)/(1-x^2)^{|m|/2}*2**l.

        Args:
            l (int): Degree of Legendre polynomial P_l^m(x).
            m (int): Order of Legendre polynomial P_l^m(x). Only non-negative values are allowed.
            k (int): Degree of x.

        Returns:
            int: Coefficient of x^k in Legendre polynomial P_l^m(x)/(1-x^2)^{|m|/2}*2**l.
        """
        if m < 0:
            raise ValueError(f"'m' must be a non-negative number")
        if not (-l <= m <= l):
            return 0
        if (l-m-k) % 2 or l < k or l < 0 or k < 0 or l-m < k:
            return 0
        j: int = (l-m-k)//2
        return (-1)**j * factorial(2*l-2*j) // factorial(j) // factorial(l-j) // factorial(l-2*j-m)
    
    @classmethod
    def Plm_coeff_float(cls, l: int, m: int, k: int) -> int | float:
        """Coefficient of x^k in Associated Legendre polynomial P_l^m(x)/(1-x^2)^{|m|/2}*2**l.

        Args:
            l (int): Degree of Legendre polynomial P_l^m(x)/(1-x^2)^{|m|/2}*2**l.
            m (int): Order of Legendre polynomial P_l^m(x)/(1-x^2)^{|m|/2}*2**l.
            k (int): Degree of x.

        Returns:
            float: Coefficient of x^k in Legendre polynomial P_l^m(x)/(1-x^2)^{|m|/2}*2**l.
        """
        if not (-l <= m <= l):
            return 0
        if (l-m-k) % 2 or l < k or l < 0 or k < 0 or l-m < k:
            return 0
        
        j: int = (l-m-k)//2
        positive_m_res: int = (-1)**j * factorial(2*l-2*j) // factorial(j) // factorial(l-j) // factorial(l-2*j-m)
        if m < 0:
            c: float = (-1)**abs(m) * factorial(l-abs(m)) / factorial(l+abs(m))
            return c * m
        else:
            return positive_m_res
    
    @classmethod
    def Ln_coeff(cls, n: int, m: int) -> int:
        """Coefficient of x^m in Laguerre polynomial L_n(x).

        Args:
            n (int): Degree of Laguerre polynomial L_n(x).
            m (int): Degree of x.

        Returns:
            int: Coefficient of x^m in Laguerre polynomial L_n(x).
        """
        nm: int = factorial(n) // factorial(m)
        return (-1)**m * nm // factorial(n-m) * nm

    @classmethod
    def Hn_coeff(cls, n: int, m: int) -> int:
        """Coefficient of x^m in Hermite polynomial H_n(x).

        Args:
            n (int): Degree of Hermite polynomial H_n(x).
            m (int): Degree of x.

        Returns:
            int: Coefficient of x^m in Hermite polynomial H_n(x).
        """
        if (n-m) % 2 or n < m or n < 0 or m < 0:
            return 0
        l: int = (n-m) // 2
        return (-1)**l * factorial(n) // factorial(l) // factorial(n-2*l) * 2**(n-2*l)

    @classmethod
    def Tn_coeff(cls, n: int, m: int) -> int:
        """Coefficient of x^m in Chebyshev polynomial of the first kind T_n(x).

        Note:
            cos(n\theta) = T_n(cos(\theta)).

        Args:
            n (int): Degree of Chebyshev polynomial of the first kind T_n(x).
            m (int): Degree of x.

        Returns:
            int: Coefficient of x^m in Chebyshev polynomial of the first kind T_n(x).
        """
        if n == 0 and m == 0:
            return 1
        if (n-m) % 2 or n < m or n < 0 or m < 0:
            return 0
        l: int = (n-m)//2
        return (-1)**l * n * cls.fac2(2*n-2*l-2) // cls.fac2(2*l) // factorial(n-2*l)

    @classmethod
    def Un_coeff(cls, n: int, m: int) -> int:
        """Coefficient of x^m in Chebyshev polynomial of the second kind U_n(x).

        Args:
            n (int): Degree of Chebyshev polynomial of the second kind U_n(x).
            m (int): Degree of x.

        Returns:
            int: Coefficient of x^m in Chebyshev polynomial of the second kind U_n(x).
        
        Note:
            U_n(x) = 1/(n+1) d/dx T_n+1(x).
            sin(n\theta) = sin(\theta) U_{n-1}(cos(\theta)).
        """
        return cls.Tn_coeff(n+1, m+1) * (m+1) // (n+1)
    
    @classmethod
    def Pn(cls, n: int) -> PolyInt:
        """Legendre polynomial whose coefficients are integerized P_n(x)*2**n.

        Args:
            n (int): Degree of Legendre polynomial P_n(x)*2**n.

        Returns:
            PolyInt: Legendre polynomial whose coefficients are integerized P_n(x)*2**n.

        Note:
            The result of this method equals to `cls([cls.Pn_coeff(n, m) for m in range(n+1)])`.

            [(1-x^2)(d/dx)^2 - 2x(d/dx) + n(n+1)] P_n(x) = 0
            P_n(x) = 1/(n!*2**n) (d/dx)^n (x^2-1)^n
            Ordinary generating function: 1/(1-2tx+t^2)^{1/2} = sum P_n(x)t^n
            (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
        """
        coeff: list[int] = [0]*(n+1)
        for k in range(n//2+1):
            coeff[n-2*k] = (-1)**k * factorial(2*n-2*k) // factorial(n-k) // factorial(k) // factorial(n-2*k)
        return cls(coeff)

    @classmethod
    def Plm(cls, l: int, m: int) -> PolyInt:
        """Associated Legendre polynomial whose coefficients are integerized P_l^m(x)/(1-x^2)^{|m|/2}*2**l.

        Args:
            l (int): Degree of associated Legendre polynomial P_l^m(x).
            m (int): Order of associated Legendre polynomial P_l^m(x). Only non-negative values are allowed.

        Returns:
            PolyInt: Associated Legendre polynomial whose coefficients are integerized P_l^m(x)/(1-x^2)^{|m|/2}*2**l.
        
        Note:
            [(1-x^2)(d/dx)^2 - 2x(d/dx) + n(n+1) - m^2/(1-x^2)] P_n^l(x) = 0
            P_l^m(x) = (-1)^m (1-x^2)^{m/2} (d/dx)^m P_l(x)
            P_l^{-m}(x) = (-1)^m (l-m)!/(l+m)! P_l^m(x)
        """
        if m < 0:
            raise ValueError(f"'m' must be a non-negative number")
        coeff: list[int] = [0]*(l-m+1)
        for k in range((l-m)//2+1):
            coeff[l-m-2*k] = (-1)**k * factorial(2*l-2*k) // factorial(l-k) // factorial(k) // factorial(l-m-2*k)
        return cls(coeff)

    @classmethod
    def Ln(cls, n: int) -> PolyInt:
        """Laguerre polynomial whose coefficients are integerized L_n(x).

        Args:
            n (int): Degree of Laguerre polynomial L_n(x).

        Returns:
            PolyInt: Laguerre polynomial whose coefficients are integerized L_n(x).
        
        Note:
            The result of this method equals to `cls([cls.Ln_coeff(n, m) for m in range(n+1)])`.

            [x(d/dx)^2 + (1-x)(d/dx) + n] L_n(x) = 0
            L_n(x) = e^x (d/dx)^n (x^n e^{-x})
            Exponential generating function: 1/(1-t) exp(-xt(1-t)) = sum L_n(x)t^{n}/n!
            L_{n+1}(x) = (2n+1-x)L_n(x) - n^2 L_{n-1}(x)
        """
        coeff: list[int] = [0]*(n+1)
        for m in range(n+1):
            nm: int = factorial(n) // factorial(m)
            coeff[m] = (-1)**m * nm // factorial(n-m) * nm
        return cls(coeff)
    
    @classmethod
    def Lnk(cls, n: int, k: int) -> PolyInt:
        """Associated Laguerre polynomial whose coefficients are integerized L_n^k(x).

        Args:
            n (int): Degree of Laguerre polynomial L_n^k(x).
            k (int): Order of Laguerre polynomial L_n^k(x).

        Returns:
            PolyInt: Associated Laguerre polynomial whose coefficients are integerized L_n^k(x).
        
        Note:
            The result of this method equals to `cls([cls.Ln_coeff(n, m) for m in range(n+1)])`.
            Note that this polynomial is not L_n^{(k)}(x) = e^x x^{-k} (d/dx)^n (x^{n+k} e^{-x}).

            [x(d/dx)^2 + (k+1-x)(d/dx) + n-k] L_n^k(x) = 0
            L_n^k(x) = (d/dx)^k L_n(x)
            generating function: (-1)^k/(1-t)^{k+1} exp(-xt(1-t)) = sum L_n^k(x)t^{n-k}/n!
        """
        coeff: list[int] = [0]*(n+1)
        for m in range(n+1):
            coeff[m] = (-1)**(m+k) * factorial(n) // factorial(n-m-k) * factorial(n) // factorial(k+m) // factorial(m)
        return cls(coeff)

    @classmethod
    def Hn(cls, n: int) -> PolyInt:
        """Hermite polynomial H_n(x).

        Args:
            n (int): Degree of Hermite polynomial H_n(x).

        Returns:
            PolyInt: Hermite polynomial H_n(x).
        
        Note:
            The result of this method equals to `cls([cls.Hn_coeff(n, m) for m in range(n+1)])`.
            
            [(d/dx)^2 - 2x(d/dx) + 2n] H_n(x) = 0
            H_n(x) = (-1)^n e^{x^2} (d/dx)^n e^{-x^2}
            Exponential generating function: exp(2xt-t^2) = sum H_n(x)t^n/n!
            H_{n+1}(x) = 2xH_n(x) - 2nH_{n-1}(x) (H_0=1, H_1=2x)
        """
        coeff: list[int] = [0]*(n+1)
        for k in range(n//2+1):
            coeff[n-2*k] = (-1)**k * factorial(n) // factorial(k) // factorial(n-2*k) * 2**(n-2*k)
        return cls(coeff)

    @classmethod
    def Tn(cls, n: int) -> PolyInt:
        """Chebyshev polynomial of the first kind T_n(x).

        Args:
            n (int): Degree of Chebyshev polynomial of the first kind T_n(x).

        Returns:
            PolyInt: Chebyshev polynomial of the first kind T_n(x).
        
        Note:
            The result of this method equals to `cls([cls.Tn_coeff(n, m) for m in range(n+1)])`.

            Ordinary generating function: (1-tx)/(1-2tx+t^2) = sum T_n(x)t^n
            T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x) (T_0=1, T_1=x).
            cos(n\theta) = T_n(cos\theta).
        """
        coeff: list[int] = [0]*(n+1)
        for k in range(n//2+1):
            coeff[n-2*k] = (-1)**k * n * cls.fac2(2*n-2*k-2) // cls.fac2(2*k) // factorial(n-2*k)
        return cls(coeff)

    @classmethod
    def Un(cls, n: int) -> PolyInt:
        """Chebyshev polynomial of the second kind U_n(x).

        Args:
            n (int): Degree of Chebyshev polynomial of the second kind U_n(x).

        Returns:
            PolyInt: Chebyshev polynomial of the second kind U_n(x).
        
        Note:
            The result of this method equals to `cls([cls.Un_coeff(n, m) for m in range(n+1)])`.
            
            Ordinary generating function: 1/(1-2tx+t^2) = sum U_n(x)t^n
            U_{n+1}(x) = 2xU_n(x) - U_{n-1}(x) (U_0=1, U_1=2x).
            U_n(x) = 1/(n+1) d/dx T_{n+1}(x).
            sin(n\theta) = sin(\theta) U_{n-1}(cos(\theta)).
            U_n(x) = sum_{l=0}^{n} P_l(x) * P_{n-l}(x).
        """
        coeff: list[int] = [0]*(n+1)
        for k in range(n//2+1):
            coeff[n-2*k] = (-1)**k * factorial(n-k) // factorial(k) // factorial(n-2*k) * 2**(n-2*k)
        return cls(coeff)

    @classmethod
    def Pilm(cls, l: int, m: int) -> PolyInt:
        """z-dependence polynomial \Pi_l^m(z)*2**l in spherical harmonics.

        Args:
            l (int): Azumuthal quantum number.
            m (int): Magnetic quantum number.

        Returns:
            PolyInt: \Pi_l^m(z)*2**l.
        
        Note:
            △R_l^m(r,θ,φ) = 0
            R_l^m(r,θ,φ) := (4\pi/(2l+1))^{1/2} r^l Y_l^m(θ,φ)
            Y_l^m(θ,φ) := (-1)^{(m+|m|)/2} ((2l+1)/4\pi)^{1/2} Q_l^{|m|} e^{imφ}
            Q_l^{|m|} := [(l-|m|)!/(l+|m|)!]^{1/2} P_l^{|m|}(cosθ)
            P_l^{|m|}(cosθ) := sin^{|m|}θ sum_{k=0}^{(l-|m|)//2} c_{lkm} cos^{l-|m|-2k}θ
            c_{lkm} := (-1)^k/(2^l) (2l-2k)! / (k!(l-k)!(l-|m|-2k)!)

            C_l^m := r^l Re[Q_l^{|m|} e^{imφ}] = r^l Q_l^{|m|} cos(mφ)
            S_l^m := r^l Im[Q_l^{|m|} e^{imφ}] = r^l Q_l^{|m|} sin(mφ)
            Π_l^m(z) := r^{l-|m|} P_l^{|m|}(cosθ) / sin^{|m|}θ
                      = sum_{k=0}^{(l-|m|)//2} c_{lkm} r^{2k} z^{l-2k-|m|}
            A_m(x,y) := r^{|m|} sin^{|m|}θ cos(mφ)
                      = [(x+iy)^{|m|} + (x-iy)^{|m|}] / 2
            B_m(x,y) := r^{|m|} sin^{|m|}θ sin(mφ)
                      = [(x+iy)^{|m|} - (x-iy)^{|m|}] / 2i
            then,
            - C_l^m = [(l-|m|)!/(l+|m|)!]^{1/2} Π_l^m(z) A_m(x,y)
            - S_l^m = [(l-|m|)!/(l+|m|)!]^{1/2} Π_l^m(z) B_m(x,y)
            - R_l^m(r,θ,φ) = (-1)^{(m+|m|)/2} (C_l^m + iS_l^m).
        """
        return cls.Plm(l, abs(m))
    
    @classmethod
    def Am(cls, m: int) -> PolyInt:
        """(x,y)-dependence polynomial A_m(x,y) in spherical harmonics.

        Args:
            m (int): Magnetic quantum number.

        Returns:
            PolyInt: A_m(x,y).
        
        Note:
            △R_l^m(r,θ,φ) = 0
            R_l^m(r,θ,φ) := (4\pi/(2l+1))^{1/2} r^l Y_l^m(θ,φ)
            Y_l^m(θ,φ) := (-1)^{(m+|m|)/2} ((2l+1)/4\pi)^{1/2} Q_l^{|m|} e^{imφ}
            Q_l^{|m|} := [(l-|m|)!/(l+|m|)!]^{1/2} P_l^{|m|}(cosθ)
            P_l^{|m|}(cosθ) := sin^{|m|}θ sum_{k=0}^{(l-|m|)//2} c_{lkm} cos^{l-|m|-2k}θ
            c_{lkm} := (-1)^k/(2^l) (2l-2k)! / (k!(l-k)!(l-|m|-2k)!)

            C_l^m := r^l Re[Q_l^{|m|} e^{imφ}] = r^l Q_l^{|m|} cos(mφ)
            S_l^m := r^l Im[Q_l^{|m|} e^{imφ}] = r^l Q_l^{|m|} sin(mφ)
            Π_l^m(z) := r^{l-|m|} P_l^{|m|}(cosθ) / sin^{|m|}θ
                      = sum_{k=0}^{(l-|m|)//2} c_{lkm} r^{2k} z^{l-2k-|m|}
            A_m(x,y) := r^{|m|} sin^{|m|}θ cos(mφ)
                      = [(x+iy)^{|m|} + (x-iy)^{|m|}] / 2
            B_m(x,y) := r^{|m|} sin^{|m|}θ sin(mφ)
                      = [(x+iy)^{|m|} - (x-iy)^{|m|}] / 2i
            then,
            - C_l^m = [(l-|m|)!/(l+|m|)!]^{1/2} Π_l^m(z) A_m(x,y)
            - S_l^m = [(l-|m|)!/(l+|m|)!]^{1/2} Π_l^m(z) B_m(x,y)
            - R_l^m(r,θ,φ) = (-1)^{(m+|m|)/2} (C_l^m + iS_l^m).
        """
        def cos_sign(i: int) -> int:
            if i % 2:
                return 0
            elif (i//2) % 2:
                return -1
            else:
                return 1
        return cls([cls.binomial_coeff(m, p)*cos_sign(m-p) for p in range(m+1)])
    
    @classmethod
    def Bm(cls, m: int) -> PolyInt:
        """(x,y)-dependence polynomial B_m(x,y) in spherical harmonics.

        Args:
            m (int): Magnetic quantum number.

        Returns:
            PolyInt: B_m(x,y).
        
        Note:
            △R_l^m(r,θ,φ) = 0
            R_l^m(r,θ,φ) := (4\pi/(2l+1))^{1/2} r^l Y_l^m(θ,φ)
            Y_l^m(θ,φ) := (-1)^{(m+|m|)/2} ((2l+1)/4\pi)^{1/2} Q_l^{|m|} e^{imφ}
            Q_l^{|m|} := [(l-|m|)!/(l+|m|)!]^{1/2} P_l^{|m|}(cosθ)
            P_l^{|m|}(cosθ) := sin^{|m|}θ sum_{k=0}^{(l-|m|)//2} c_{lkm} cos^{l-|m|-2k}θ
            c_{lkm} := (-1)^k/(2^l) (2l-2k)! / (k!(l-k)!(l-|m|-2k)!)

            C_l^m := r^l Re[Q_l^{|m|} e^{imφ}] = r^l Q_l^{|m|} cos(mφ)
            S_l^m := r^l Im[Q_l^{|m|} e^{imφ}] = r^l Q_l^{|m|} sin(mφ)
            Π_l^m(z) := r^{l-|m|} P_l^{|m|}(cosθ) / sin^{|m|}θ
                      = sum_{k=0}^{(l-|m|)//2} c_{lkm} r^{2k} z^{l-2k-|m|}
            A_m(x,y) := r^{|m|} sin^{|m|}θ cos(mφ)
                      = [(x+iy)^{|m|} + (x-iy)^{|m|}] / 2
            B_m(x,y) := r^{|m|} sin^{|m|}θ sin(mφ)
                      = [(x+iy)^{|m|} - (x-iy)^{|m|}] / 2i
            then,
            - C_l^m = [(l-|m|)!/(l+|m|)!]^{1/2} Π_l^m(z) A_m(x,y)
            - S_l^m = [(l-|m|)!/(l+|m|)!]^{1/2} Π_l^m(z) B_m(x,y)
            - R_l^m(r,θ,φ) = (-1)^{(m+|m|)/2} (C_l^m + iS_l^m).
        """
        def sin_sign(i: int) -> int:
            if i % 2 == 0:
                return 0
            elif ((i-1)//2) % 2 == 0:
                return 1
            else:
                return -1
        return cls([cls.binomial_coeff(m, p)*sin_sign(m-p) for p in range(m+1)])
    
    @classmethod
    def multipole_notation(cls, l: int, m: int) -> str:
        """Multipole represented by (x,y,z).

        Args:
            l (int): Azumuthal quantum number.
            m (int): Magnetic quantum number.

        Returns:
            str: Multipole represented by (x,y,z).
        
         Note:
            #   (l, m): result
                (0, 0): 1
                (1, -1): y
                (1, 0): z
                (1, 1): x
                (2, -2): x y
                (2, -1): y z
                (2, 0): 3z^2 - r^2
                (2, 1): x z
                (2, 2): x^2 - y^2
        """
        if l == m == 0:
            return "1"
        ZR: PolyInt = cls.Pilm(l, abs(m)).simplify()[0]
        XY: PolyInt
        if m >= 0:
            XY = cls.Am(abs(m)).simplify()[0]
        else:
            XY = cls.Bm(abs(m)).simplify()[0]
        def expstr(b: str, e: int) -> str:
            if e == 0:
                return f""
            elif e == 1:
                return f"{b}"
            else:
                return f"{b}^{e}"
        def coeffstr(c: int) -> str:
            if c == 1:
                return f""
            elif c == -1:
                return f"-"
            else:
                return f"{c}"
                    
        def zr(i: int, c: int, l: int, m: int) -> str:
            if i == 0:
                if l-abs(m) == 0:
                    return ""
                else:
                    return f"{coeffstr(c)}{expstr('r',l-abs(m))}"
            elif i == l-abs(m):
                return f"{coeffstr(c)}{expstr('z',i)}"
            else:
                return f"{coeffstr(c)}{expstr('z',i)} {expstr('r',l-abs(m)-i)}"
        
        def xy(i: int, c: int, l: int, m: int) -> str:
            if m == 0:
                return ""
            elif m == 1:
                return f"{coeffstr(c)}{expstr('x',1)}"
            elif m == -1:
                return f"{coeffstr(c)}{expstr('y',1)}"
            else:
                return f"{coeffstr(c)}{expstr('x',i)} {expstr('y',abs(m)-i)}"

        zrstr: str = ""
        for i in range(len(ZR.P)):
            if ZR.P[i] != 0:
                zrstr = f"{zr(i, ZR.P[i], l, m)} + {zrstr}"
        zrstr = zrstr.strip(' + ')

        xystr: str = ""
        for i in range(len(XY.P)):
            if XY.P[i] != 0:
                xystr = f"{xy(i, XY.P[i], l, m)} + {xystr}"
        xystr = xystr.strip(' + ')

        if len(np.where(ZR.P != 0)[0]) > 1 and len(xystr) > 0:
            zrstr = f"({zrstr})"
        if len(np.where(XY.P != 0)[0]) > 1 and len(zrstr) > 0:
            xystr = f"({xystr})"

        res: str = f"{xystr} {zrstr}".strip(" ").replace("+ -", "- ").replace("  ", " ")
        return res


def visualize_spherical_harmonics(l_max: int = 10) -> None:
    """Spherical harmonics visualizer.

    Note:
        Visualized points are calculated as below:
            1. absolute mode:
                r = |Y_l^m(theta, phi)|
            2. Real mode:
                r = |Re[Y_l^m(theta, phi)]|
            3. Imaginary mode:
                r = |Im[Y_l^m(theta, phi)]|
            then,
            x = r * sin(theta) * cos(phi)
            y = r * sin(theta) * sin(phi)
            z = r * cos(theta)

    Args:
        l_max (int, optional): Maximum azimuthal quantum number. Defaults to 10.
    """
    def spherical_harmonics_real_form(theta: npt.NDArray, phi: npt.NDArray, l: int, m: int) -> npt.NDArray:
        Y: npt.NDArray = sph_harm(abs(m), l, phi, theta)
        if m < 0:
            Y = np.sqrt(2) * (-1)**m * Y.imag
        elif m > 0:
            Y = np.sqrt(2) * (-1)**m * Y.real
        return Y.real
    
    def paramed_sph_harm(l: int, m: int, mode: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Spherical harmonics function.

        Args:
            l (int): Azimuthal quantum number.
            m (int): Magnetic quantum number.
            mode (int): Visualizer mode: {0: |Y_l^m|, 1: |Re(Y_l^m)|, 2: |Im(Y_l^m)|}.

        Raises:
            ValueError: The value of mode (param[1]) must be 0, 1 or 2.

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]: x, y, z coordination values and the color of points.
        """
        n: int = 512 + 1
        theta: npt.NDArray = np.linspace(0, np.pi, n)
        phi: npt.NDArray = np.linspace(0, 2*np.pi, n)
        theta, phi = np.meshgrid(theta, phi)
        r: npt.NDArray
        if mode == 0:
            # real-form
            r = np.abs(spherical_harmonics_real_form(theta, phi, l, m))
        elif mode == 1:
            # abs
            r = np.abs(sph_harm(m, l, phi, theta))
        elif mode == 2:
            # Re
            r = np.abs(np.real(sph_harm(m, l, phi, theta)))
        elif mode == 3:
            # Im
            r = np.abs(np.imag(sph_harm(m, l, phi, theta)))
        else:
            raise ValueError

        x: npt.NDArray = r * np.sin(theta) * np.cos(phi)
        y: npt.NDArray = r * np.sin(theta) * np.sin(phi)
        z: npt.NDArray = r * np.cos(theta)

        plus_minus: tuple[str, str] = ('r','b')
        color: npt.NDArray = np.empty(theta.shape, dtype=str)
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                if mode == 0:
                    if spherical_harmonics_real_form(theta[i,j], phi[i,j], l, m) >= 0:
                    # if spherical_harmonics_sign_tp(theta[i,j], phi[i,j], l, m) >= 0:
                    # if spherical_harmonics_sign_xyz(x[i,j], y[i,j], z[i,j], l, m) >= 0:
                        color[i,j] = plus_minus[0]
                    else:
                        color[i,j] = plus_minus[1]
                elif mode == 1:
                    color[i,j] = 'magenta'
                elif mode == 2:
                    color[i,j] = 'red'
                elif mode == 3:
                    color[i,j] = 'blue'
        return x, y, z, color
    
    orbit_name: list[str] = ["s", "p", "d", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]

    fig: plt.Figure = plt.figure(figsize=(7, 7))
    # 領域を小区画に分けて番号で指定できるようにする
    gs: plt.GridSpec = fig.add_gridspec(20+4, 20+4)
    ax: plt.Subplot = fig.add_subplot(gs[:21,:], projection='3d')
    ax.set_title(rf'${orbit_name[0]}_{{{PolyInt.multipole_notation(0, 0)}}}$')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    axcolor: str = 'silver'
    # グラフの下の方に4段のスライダーを設置する
    AXS: list[plt.Subplot] = []
    AXS.append(fig.add_subplot(gs[21, 9:14], facecolor=axcolor))
    AXS.append(fig.add_subplot(gs[22, 3:20], facecolor=axcolor))
    AXS.append(fig.add_subplot(gs[23, 3:20], facecolor=axcolor))
    AXS.append(fig.add_subplot(gs[20, 3:20], facecolor=axcolor)) # あとで不可視化するのでどこに置いてもいい

    mode: int = 0 # initial mode is real-form representation
    l_min: int = 0
    
    # 初期値で曲面を描画
    X, Y, Z, color = paramed_sph_harm(l_min, 0, mode)
    # 曲面の描画(ここは適宜カスタマイズする)
    ax.plot_surface(X, Y, Z, facecolors=color)
    # ax.plot_wireframe(X, Y, Z, color="skyblue", linewidth=0.5)

    # Slider(ax=plotするAxis, label=ラベル, valmin=sliderの最小値, valmax=sliderの最大値, 
    ###      valinit=sliderの初期値, valstep=sliderのstep値, 
    ###      slidermin=Sliderオブジェクト:あるSliderオブジェクトのvalをこのSliderが取れる値の下限とする,
    ###      slidermax=Sliderオブジェクト:あるSliderオブジェクトのvalをこのSliderが取れる値の上限とする)
    SLIDERS: list[Slider] = []
    SLIDERS.append(Slider(ax=AXS[0], label='XYZ/Abs/Re/Im', valmin=0, valmax=3, valinit=mode, valstep=1))
    SLIDERS.append(Slider(ax=AXS[1], label='l', valmin=l_min, valmax=l_max, valinit=l_min, valstep=1))
    SLIDERS.append(Slider(ax=AXS[2], label='m', valmin=-l_max, valmax=l_max, valinit=0, valstep=1, slidermin=None, slidermax=None))
    SLIDERS.append(Slider(ax=AXS[3], label='-l', valmin=-l_max, valmax=-l_min, valinit=-l_min, valstep=1)) # -l <= m <= l に制限する用
    
    # mの値の上限をl,下限を-lにする
    SLIDERS[2].slidermax = SLIDERS[1]
    SLIDERS[2].slidermin = SLIDERS[3]
    
    # AXS[3]のSliderは，裏でmの値を制御するために使うだけなので不可視化する
    AXS[3].set_visible(False)
    
    def update(val: int) -> None:
        # .valはSliderで新たにセットされた値を表す
        mode_next, l_next, m_next = SLIDERS[0].val, SLIDERS[1].val, SLIDERS[2].val
        SLIDERS[3].val = -l_next
        ax.clear() # 前回の結果を削除
        X_next, Y_next, Z_next, color_next = paramed_sph_harm(l_next, m_next, mode_next)
        ax.plot_surface(X_next, Y_next, Z_next, facecolors=color_next)
        # ax.plot_wireframe(X_next, Y_next, Z_next, color=color_next, linewidth=0.5)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if mode_next == 0:
            ax.set_title(rf'${orbit_name[l_next]}_{{{PolyInt.multipole_notation(l_next,m_next)}}}$')
        elif mode_next == 1:
            ax.set_title(rf'spherical harmonics $|Y_{l_next}^{{{m_next}}}(\theta, \phi)|$')
        elif mode_next == 2:
            ax.set_title(rf'spherical harmonics $|\mathrm{{Re}}Y_{l_next}^{{{m_next}}}(\theta, \phi)|$')
        elif mode_next == 3:
            ax.set_title(rf'spherical harmonics $|\mathrm{{Im}}Y_{l_next}^{{{m_next}}}(\theta, \phi)|$')
        fig.canvas.draw_idle() # ここで描画を一旦ストップ

    # 設定したSliderを.on_changed(update)でactivate
    SLIDERS[0].on_changed(update)
    SLIDERS[1].on_changed(update)
    SLIDERS[2].on_changed(update)
    
    # リセットボタンの設置
    # 押されたら初期化
    resetax: plt.Axes = plt.axes([0.85, 0.05, 0.1, 0.03])
    button: Button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    def reset(event: Any) -> None:
        for i in range(3):
            SLIDERS[i].reset()
    button.on_clicked(reset)
    plt.show()


def main() -> None:
    pass
    return

if __name__ == "__main__":
    main()

