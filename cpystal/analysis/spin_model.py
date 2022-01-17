"""`cpystal.analysis.spin_model` is a module for analyzing some spin models.

Classes:
    `SpinOperator`
    `MultiSpinSystemOperator`
    `MultiSpinSystemOperator2`
"""
import numpy as np
from functools import reduce
from typing import List, Union


class SpinOperator:
    """スピン量子数Sのスピン演算子

    Note:
        行列表示の際の基底は
        {|S>,|S-1>,...,|-S>}
        のようにとる．
    """
    def __init__(self, S: float) -> None:
        self.S: float = S
        self.N: int = int(2*S+1)

    def Sz(self) -> np.ndarray:
        return np.diag(np.arange(self.S,-self.S-1,-1))

    def Sp(self) -> np.ndarray:
        res: np.ndarray = np.zeros((self.N, self.N))
        for m in range(1,self.N):
            # <S,M+1|S+|S,M> = sqrt((S-M)*(S+M+1))
            # m = S-M
            res[m-1][m] = np.sqrt(m*(self.N-m))
        return res

    def Sm(self) -> np.ndarray:
        res: np.ndarray = np.zeros((self.N, self.N))
        for m in range(self.N-1):
            # <S,M-1|S-|S,M> = sqrt((S+M)*(S-M+1))
            # m = S-M
            res[m+1][m] = np.sqrt((self.N-1-m)*(m+1))
        return res
    
    def Sx(self) -> np.ndarray:
        return (self.Sp() + self.Sm()) / 2.0

    def Sy(self) -> np.ndarray:
        return (self.Sp() - self.Sm()) / 2.0j


class MultiSpinSystemOperator(SpinOperator):
    """スピン量子数がSであるK個のスピンからなる系の諸スピン演算子の行列要素 の計算

    Note:
        行列表示の際の基底は
        {|0,S>,|0,S-1>,...,|0,-S>,|1,S>,|1,S-1>,...}
        のようにとる．
        各methodの返り値のshapeは，(self.dim, self.dim)
        ただし，Si_cross_Sj のみ (3, self.dim, self.dim) <- DMベクトルと内積@をとればDM相互作用項になる
    """
    def __init__(self, S: float, K: int) -> None:
        super().__init__(S)
        self.K: int = K
        self.dim: int = self.N ** self.K

    def _tensorproduct(self, i: int, j: int, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        a: list[np.ndarray] = [np.eye(self.N)]*self.K
        a[i] = A
        a[j] = B
        return reduce(np.kron, a)

    def SpSm(self, i: int, j: int) -> np.ndarray:
        """(Si+)(Sj-) の計算
        """
        return self._tensorproduct(i, j, self.Sp(), self.Sm())

    def SmSp(self, i: int, j: int) -> np.ndarray:
        """(Si-)(Sj+) の計算
        """
        return self._tensorproduct(i, j, self.Sm(), self.Sp())

    def SzSz(self, i: int, j: int) -> np.ndarray:
        """(Siz)(Sjz) の計算
        """
        return self._tensorproduct(i, j, self.Sz(), self.Sz())

    def Si_dot_Sj(self, i: int, j: int) -> np.ndarray:
        """Si \dot Sj の計算
        """
        return (self.SpSm(i,j)+self.SmSp(i,j)) / 2 + self.SzSz(i,j)

    def Si_cross_Sj(self, i: int, j: int) -> np.ndarray:
        """Si x Sj の計算

        Note:
            SixSjy - Siysjx == 1.0j/2 (Si+Sj- - Si-Sj+)
        """
        sysz: np.ndarray = self._tensorproduct(i, j, self.Sy(), self.Sz())
        szsy: np.ndarray = self._tensorproduct(i, j, self.Sz(), self.Sy())

        szsx: np.ndarray = self._tensorproduct(i, j, self.Sz(), self.Sx())
        sxsz: np.ndarray = self._tensorproduct(i, j, self.Sx(), self.Sz())

        sxsy: np.ndarray = self._tensorproduct(i, j, self.Sx(), self.Sy())
        sysx: np.ndarray = self._tensorproduct(i, j, self.Sy(), self.Sx())
        return np.array([sysz-szsy, szsx-sxsz, sxsy-sysx])


class MultiSpinSystemOperator2():
    """i番目のスピン量子数がS_iであるK個のスピンからなる系の諸スピン演算子の行列要素の計算

    Note:
        行列表示の際の基底は
        {|0,S_0>,|0,S_0-1>,...,|0,-S_0>,|1,S_1>,|1,S_1-1>,...}
        のようにとる．

        各methodの返り値のshapeは，(self.dim, self.dim)
        ただし，Si_cross_Sj のみ (3, self.dim, self.dim) <- DMベクトルと内積@をとればDM相互作用項になる
    """
    def __init__(self, S: Union[float, List[float]], K: int) -> None:
        if isinstance(S, float) or isinstance(S,int):
            S = [S for _ in range(K)]
        assert len(S) == K
        self.S: List[float] = S
        self.spins: List[SpinOperator] = [SpinOperator(s) for s in S]
        self.K: int = K
        self.N: List[int] = [int(2*s+1) for s in S]
        self.dim: int = reduce(lambda x,y:x*y, self.N)

    def _tensorproduct(self, i: int, j: int, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        a: list[np.ndarray] = [np.eye(n) for n in self.N]
        a[i] = A
        a[j] = B
        return reduce(np.kron, a)

    def SpSm(self, i: int, j: int) -> np.ndarray:
        """(Si+)(Sj-) の計算
        """
        return self._tensorproduct(i, j, self.spins[i].Sp(), self.spins[j].Sm())

    def SmSp(self, i: int, j: int) -> np.ndarray:
        """(Si-)(Sj+) の計算
        """
        return self._tensorproduct(i, j, self.spins[i].Sm(), self.spins[j].Sp())

    def SzSz(self, i: int, j: int) -> np.ndarray:
        """(Siz)(Sjz) の計算
        """
        return self._tensorproduct(i, j, self.spins[i].Sz(), self.spins[j].Sz())

    def Si_dot_Sj(self, i: int, j: int) -> np.ndarray:
        """Si \dot Sj の計算
        """
        return (self.SpSm(i,j)+self.SmSp(i,j)) / 2 + self.SzSz(i,j)

    def Si_cross_Sj(self, i: int, j: int) -> np.ndarray:
        """Si x Sj の計算

        Note:
            SixSjy - Siysjx == 1.0j/2 (Si+Sj- - Si-Sj+)
        """
        sysz: np.ndarray = self._tensorproduct(i, j, self.spins[i].Sy(), self.spins[j].Sz())
        szsy: np.ndarray = self._tensorproduct(i, j, self.spins[i].Sz(), self.spins[j].Sy())

        szsx: np.ndarray = self._tensorproduct(i, j, self.spins[i].Sz(), self.spins[j].Sx())
        sxsz: np.ndarray = self._tensorproduct(i, j, self.spins[i].Sx(), self.spins[j].Sz())

        sxsy: np.ndarray = self._tensorproduct(i, j, self.spins[i].Sx(), self.spins[j].Sy())
        sysx: np.ndarray = self._tensorproduct(i, j, self.spins[i].Sy(), self.spins[j].Sx())
        return np.array([sysz-szsy, szsx-sxsz, sxsy-sysx])

def main() -> None:
    pass
    return

if __name__ == "__main__":
    main()
