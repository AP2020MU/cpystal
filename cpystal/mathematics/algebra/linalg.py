"""`cpystal.math.linalg` is a module for linear alogorithms.

Functions:
    `triu_inv`
        -Upper triangular matrix linear simultaneous equation.
    `tril_inv`
        -Solve a linear simultaneous equation with lower triangular matrix.
    `Jacobi`
        -Solve a linear simultaneous equation by Jacobi method.
    `GaussSeidel`
        -Solve a linear simultaneous equation by Gauss-Seidel method.
    `TDMA`
        -Solve a linear simultaneous equation by Tri-Diagonal Matrix Algorithm.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def triu_inv(U: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    """Solve a linear simultaneous equation with upper triangular matrix.

    Args:
        U (npt.NDArray): Upper triangular matrix.
        b (npt.NDArray): Vector.

    Returns:
        npt.NDArray: Answer of the linear simultaneous equation.
    """
    n: int = len(U)
    x: npt.NDArray = np.zeros(n)
    for i in reversed(range(n)):
        s: float = 0.
        for j in range(i+1,n):
            s += U[i][j] * x[j]
        x[i] = (b[i]-s) / U[i][i]
    return x

def tril_inv(L: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    """Solve a linear simultaneous equation with lower triangular matrix.

    Args:
        L (npt.NDArray): Lower triangular matrix.
        b (npt.NDArray): Vector.

    Returns:
        npt.NDArray: Answer of the linear simultaneous equation.
    """
    n: int = len(L)
    x: npt.NDArray = np.zeros(n)
    for i in range(n):
        s: float = 0.
        for j in range(i):
            s += L[i][j] * x[j]
        x[i] = (b[i]-s) / L[i][i]
    return x

def Jacobi(A: npt.NDArray, b: npt.NDArray, tol: float = 1e-9):
    """Solve a linear simultaneous equation by Jacobi method.

    Args:
        A (npt.NDArray): Coefficient matrix.
        b (npt.NDArray): Vector.
        tol (float, optional): Tolerance. Defaults to 1e-9.

    Returns:
        npt.NDArray: Answer of the linear simultaneous equation.
    
    ToDo:
        Pivoting for 0 elements in D.
    """
    k: int = 0
    x_k: npt.NDArray = np.empty_like(b, dtype=np.float64)
    error: float = float('inf')

    A_diag_vector: npt.NDArray = np.diag(A)
    D: npt.NDArray = np.diag(A_diag_vector)
    LU: npt.NDArray = A-D # LU分解ではなく、LU==L+U==A-D
    D_inv: npt.NDArray = np.diag(1/A_diag_vector) # Dの中に0があったらどうするの？

    #while error  > tol: # 更新量がtol以下になったら終了
    while np.linalg.norm(b-np.dot(A,x_k)) > tol: # 残差がtol以下になったら終了
        x: npt.NDArray = np.dot(D_inv, b-np.dot(LU, x_k))
        k += 1
        error = np.linalg.norm(x-x_k)/np.linalg.norm(x)
        x_k = x
    return x

def GaussSeidel(A: npt.NDArray, b: npt.NDArray, tol: float = 1e-9) -> npt.NDArray:
    """Solve a linear simultaneous equation by Gauss-Seidel method.

    Args:
        A (npt.NDArray): Coefficient matrix.
        b (npt.NDArray): Vector.
        tol (float, optional): Tolerance. Defaults to 1e-9.

    Returns:
        npt.NDArray: Answer of the linear simultaneous equation.
    """
    k: int = 0
    x_k: npt.NDArray = np.empty_like(b, dtype=np.float64)
    error: float = float('inf')

    L: npt.NDArray = np.tril(A) # 下三角行列(対角成分含む)
    U: npt.NDArray = A - L # 上三角行列
    
    # while error > tol: # 更新量がtol以下になったら終了
    while np.linalg.norm(b-np.dot(A,x_k)) > tol: # 残差がtol以下になったら終了
        x: npt.NDArray = tril_inv(L, b-np.dot(U, x_k))
        k += 1
        # error = np.linalg.norm(x-x_k)/np.linalg.norm(x)
        x_k = x
    return x

def TDMA(d: npt.NDArray, u: npt.NDArray, l: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    """Solve a linear simultaneous equation by Tri-Diagonal Matrix Algorithm.

    Args:
        d (npt.NDArray): Diagonal elements.
        u (npt.NDArray): Upper diagonal elements.
        l (npt.NDArray): Lower diagonal elements.
        b (npt.NDArray): Right side vector.

    Returns:
        npt.NDArray: Answer of the linear simultaneous equation.
    """
    n: int = len(d)
    P: npt.NDArray = np.zeros(n)
    Q: npt.NDArray = np.zeros(n)
    x: npt.NDArray = np.zeros(n)
    for i in range(n):
        P[i] = -u[i] / (d[i]+l[i]*P[i-1])
        Q[i] = (b[i]-l[i]*Q[i-1]) / (d[i]+l[i]*P[i-1])
    x[-1] = Q[-1]
    for i in range(n-2,-1,-1):
        x[i] = P[i] * x[i+1] + Q[i]
    return x


def main() -> None:
    pass
    return

if __name__ == "__main__":
    main()

