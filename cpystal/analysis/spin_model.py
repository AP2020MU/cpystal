"""`cpystal.analysis.spin_model` is a module for analyzing some spin models.

Classes:
    `SpinOperator`
    `MultiSpinSystemOperator`
"""
from __future__ import annotations
from functools import reduce

import numpy as np
import numpy.typing as npt

class SpinOperator:
    """Spin operators of spin quantum number S.

    Note:
        The bases for the matrix representation are as follows:
            {|S>,|S-1>,...,|-S>}
    """
    def __init__(self, S: float) -> None:
        """
        Args:
            S (float): Spin quantum number (integer or half-integer).
        """
        self.S: float = S
        self.N: int = int(2*S+1)

    def Sz(self) -> npt.NDArray[np.complex64]:
        """Sz
        """
        return np.diag(np.arange(self.S,-self.S-1,-1))

    def Sp(self) -> npt.NDArray[np.complex64]:
        """Raising operator: S+
        """
        res: npt.NDArray[np.complex64] = np.zeros((self.N, self.N))
        for m in range(1,self.N):
            # <S,M+1|S+|S,M> = sqrt((S-M)*(S+M+1))
            # m = S-M
            res[m-1][m] = np.sqrt(m*(self.N-m))
        return res

    def Sm(self) -> npt.NDArray[np.complex64]:
        """Lowering operator: S-
        """
        res: npt.NDArray[np.complex64] = np.zeros((self.N, self.N))
        for m in range(self.N-1):
            # <S,M-1|S-|S,M> = sqrt((S+M)*(S-M+1))
            # m = S-M
            res[m+1][m] = np.sqrt((self.N-1-m)*(m+1))
        return res
    
    def Sx(self) ->npt.NDArray[np.complex64]:
        """Sx
        """
        return (self.Sp() + self.Sm()) / 2.0

    def Sy(self) -> npt.NDArray[np.complex64]:
        return (self.Sp() - self.Sm()) / 2.0j


class MultiSpinSystemOperator:
    """Calculation of matrix elements of various spin operators 
        for a system of K spins whose i-th spin quantum number is S_i.

    Note:
        The bases for the matrix representation are as follows:
        {|0,S_0>,|0,S_0-1>,...,|0,-S_0>,|1,S_1>,|1,S_1-1>,...}

        The shape of returns of each class method is (self.dim, self.dim).

        However, only `Si_cross_Sj`, (3, self.dim, self.dim) 
            <- Taking the inner product of the DM vector and this, 
                the DM interaction term will be obtained.
    """
    def __init__(self, S: float | list[float], K: int) -> None:
        if isinstance(S, float) or isinstance(S,int):
            S = [S for _ in range(K)]
        assert len(S) == K
        self.S: list[float] = S
        self.spins: list[SpinOperator] = [SpinOperator(s) for s in S]
        self.K: int = K
        self.N: list[int] = [int(2*s+1) for s in S]
        self.dim: int = reduce(lambda x,y:x*y, self.N)

    def _tensorproduct(self, i: int, j: int, A: npt.NDArray[np.complex64], B: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
        """By taking the tensor product (Kronecker product) of the K operators, 
            it will be calculated that the (self.dim × self.dim) matrix, which represents the operations 
            such that operator A acts on the i-th spin, operator B on the j-th spin, and nothing on the other spins.
        """
        a: list[npt.NDArray[np.complex64]] = [np.eye(n) for n in self.N]
        a[i] = A
        a[j] = B
        return reduce(np.kron, a)

    def SpSm(self, i: int, j: int) -> npt.NDArray[np.complex64]:
        """(Si+)(Sj-)
        """
        return self._tensorproduct(i, j, self.spins[i].Sp(), self.spins[j].Sm())

    def SmSp(self, i: int, j: int) -> npt.NDArray[np.complex64]:
        """(Si-)(Sj+)
        """
        return self._tensorproduct(i, j, self.spins[i].Sm(), self.spins[j].Sp())

    def SzSz(self, i: int, j: int) -> npt.NDArray[np.complex64]:
        """(Siz)(Sjz)
        """
        return self._tensorproduct(i, j, self.spins[i].Sz(), self.spins[j].Sz())

    def Si_dot_Sj(self, i: int, j: int) -> npt.NDArray[np.complex64]:
        """Si \dot Sj
        """
        return (self.SpSm(i,j)+self.SmSp(i,j)) / 2 + self.SzSz(i,j)

    def Si_cross_Sj(self, i: int, j: int) -> npt.NDArray[np.complex64]:
        """Si x Sj

        Note:
            SixSjy - Siysjx == 1.0j/2 (Si+Sj- - Si-Sj+)
        """
        Si: SpinOperator = self.spins[i]
        Sj: SpinOperator = self.spins[j]
        sysz: npt.NDArray[np.complex64] = self._tensorproduct(i, j, Si.Sy(), Sj.Sz())
        szsy: npt.NDArray[np.complex64] = self._tensorproduct(i, j, Si.Sz(), Sj.Sy())

        szsx: npt.NDArray[np.complex64] = self._tensorproduct(i, j, Si.Sz(), Sj.Sx())
        sxsz: npt.NDArray[np.complex64] = self._tensorproduct(i, j, Si.Sx(), Sj.Sz())

        sxsy: npt.NDArray[np.complex64] = self._tensorproduct(i, j, Si.Sx(), Sj.Sy())
        sysx: npt.NDArray[np.complex64] = self._tensorproduct(i, j, Si.Sy(), Sj.Sx())
        return np.array([sysz-szsy, szsx-sxsz, sxsy-sysx])

def main() -> None:
    pass
    return

if __name__ == "__main__":
    main()
