"""`cpystal.math.matrixperator` is a module for JOperators of quantum mechanics.

Functions:

Classes:

"""
from __future__ import annotations

from math import factorial
from typing import Any, Iterable, TypeVar

from more_itertools import distinct_permutations
import numpy as np
import numpy.typing as npt
import sympy


from core import PolyInt

HalfInt = TypeVar("HalfInt", int, float)
JOperatorChild = TypeVar("JOperatorChild", bound="JOperator")

class JOperator(object):
    def __init__(self, J: HalfInt) -> None:
        self.matrix: npt.NDArray = np.zeros((int(2*J+1), int(2*J+1)))
        self.J: HalfInt = J
        self.M: npt.NDArray = np.arange(-J,J+1,1)
        self.JJ: HalfInt = J*(J+1)

    def _constructor(self) -> JOperator:
        return self.__class__(self.J)

    @staticmethod
    def Jz(J) -> JOperator:
        res: JOperator = JOperator(J)
        res.matrix = np.diag(res.M)
        return res

    @staticmethod
    def Jp(J) -> JOperator:
        res: JOperator = JOperator(J)
        res.matrix = np.diag(np.sqrt((res.J-res.M)*(res.J+res.M+1))[:-1], 1)
        return res

    @staticmethod
    def Jm(J) -> JOperator:
        res: JOperator = JOperator(J)
        res.matrix = np.diag(np.sqrt((res.J+res.M)*(res.J-res.M+1))[1:], -1)
        return res

    @staticmethod
    def Jx(J) -> JOperator:
        return 0.5 * (JOperator.Jp(J) + JOperator.Jm(J))

    @staticmethod
    def Jy(J) -> JOperator:
        return -0.5j * (JOperator.Jp(J) - JOperator.Jm(J))
    
    def __len__(self) -> int:
        return int(2*self.J + 1)
    
    def __str__(self) -> str:
        return str(self.round().matrix)
    
    def __repr__(self):
        return repr(self.round().matrix)
    
    def __pos__(self) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        res.matrix = +self.matrix
        return res
    
    def __neg__(self) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        res.matrix = -self.matrix
        return res
    
    def __eq__(self, other: Any) -> bool:
        return np.all(self.round().matrix == other.round().matrix)
    
    def __neq__(self, other: Any) -> bool:
        return not self.__eq__(other)
    
    def __iter__(self) -> Iterable:
        return iter(self.matrix)

    def __add__(self, other: Any) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        if isinstance(other, JOperator):
           res.matrix = self.matrix + other.matrix
        else:
           res.matrix = self.matrix + other * np.identity(len(self))
        return res

    def __radd__(self, other: Any) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        if isinstance(other, JOperator):
            res.matrix = self.matrix + other.matrix
        else:
            res.matrix = self.matrix + other * np.identity(len(self))
        return res

    def __sub__(self, other: Any) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        if isinstance(other, JOperator):
            res.matrix = self.matrix - other.matrix
        else:
            res.matrix = self.matrix - other * np.identity(len(self))
        return res

    def __mul__(self, other: Any) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        if (isinstance(other, int) or isinstance(other, float) or isinstance(other, complex)):
           res.matrix = self.matrix * other
        else:
           res.matrix = self.matrix @ other.matrix
        return res

    def __rmul__(self, other: Any) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        if (isinstance(other, int) or isinstance(other, float)  or isinstance(other, complex)):
           res.matrix = other * self.matrix
        else:
           res.matrix = other.matrix @ self.matrix
        return res
    
    def __truediv__(self, other: Any) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        if (isinstance(other, int) or isinstance(other, float) or isinstance(other, complex)):
           res.matrix = self.matrix / other
        else:
           raise ValueError
        return res
    
    def __matmul__(self, other: Any) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        res.matrix = self.matrix @ other.matrix
        return res

    def __rmatmul__(self, other: Any) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        res.matrix = other.matrix @ self.matrix
        return res

    def __pow__(self, n: int) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        res.matrix = np.linalg.matrix_power(self.matrix, n)
        return res

    def __xor__(self, other: Any) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        res.matrix = self.matrix @ other.matrix + other.matrix @ self.matrix
        return res
    
    def __rxor__(self, other: Any) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        res.matrix = self.matrix @ other.matrix + other.matrix @ self.matrix
        return res
    
    def __mod__(self, other: Any) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        res.matrix = self.matrix @ other.matrix - other.matrix @ self.matrix
        return res
    
    def __rmod__(self, other: Any) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        res.matrix = other.matrix @ self.matrix - self.matrix @ other.matrix
        return res
    
    def round(self) -> JOperatorChild:
        res: JOperatorChild = self._constructor()
        res.matrix = np.round(self.matrix, decimals=12)
        return res

class StevensJOperator(JOperator):
    def __init__(self, J: HalfInt, l: HalfInt, m: HalfInt) -> None:
        super().__init__(J)
        self.l: HalfInt = l
        self.m: HalfInt = m
        Jx: JOperator = JOperator.Jx(J)
        Jy: JOperator = JOperator.Jy(J)
        Jz: JOperator = JOperator.Jz(J)
        symmetrized_expr: sympy.Add = self.symmetrize_xyz(PolyInt.to_symbol(l, m))
        x, y, z = sympy.var("x, y, z", commutative=False)
        self.matrix = sympy.lambdify([x,y,z], symmetrized_expr)(Jx, Jy, Jz).matrix

    def _constructor(self) -> JOperatorChild:
        return self.__class__(self.J, self.l, self.m)
    
    @staticmethod
    def symmetrize_xyz(expr_str: str) -> sympy.Add:
        x, y, z = sympy.var("x, y, z", commutative=False)
        local_dict: dict[str, sympy.Symbol] = {'x': x, 'y': y, 'z': z}
        pol = sympy.simplify(sympy.parse_expr(expr_str, local_dict=local_dict)).expand()
        res = sympy.simplify("0")
        for term in pol.as_ordered_terms():
            px: int = 0
            py: int = 0
            pz: int = 0
            for factor in sympy.Mul.make_args(term):
                if x in factor.as_powers_dict():
                    px = int(factor.as_powers_dict()[x])
                if y in factor.as_powers_dict():
                    py = int(factor.as_powers_dict()[y])
                if z in factor.as_powers_dict():
                    pz = int(factor.as_powers_dict()[z])
            c = term.coeff(x**px * y**py * z**pz) * factorial(px)*factorial(py)*factorial(pz) / factorial(px+py+pz)
            for expr in distinct_permutations("x"*px + "y"*py + "z"*pz):
                res += c * sympy.parse_expr("*".join(expr), local_dict=local_dict)
        return res


def main() -> None:
    J = 2
    l = 3
    m = 2
    Jx: JOperator = JOperator.Jx(J)
    Jy: JOperator = JOperator.Jy(J)
    Jz: JOperator = JOperator.Jz(J)
    sop = (Jx**2*Jz + Jx*Jz*Jx + Jz*Jx**2)/3 - (Jy**2*Jz + Jy*Jz*Jy + Jz*Jy**2)/3
    print(StevensJOperator(J,l,m) == sop)

    return

if __name__ == "__main__":
    main()

