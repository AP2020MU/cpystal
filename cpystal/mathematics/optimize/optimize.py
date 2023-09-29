"""`cpystal.mathematics.optimize` is a module for linear alogorithms.

Functions:
    `IterativeMethod`
        - optimizer
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

class IterativeMethod:
    def __init__(self, 
                 f: Callable[[npt.NDArray], float], 
                 grad_f: Callable[[npt.NDArray], float] | None = None, 
                 Hesse_f: Callable[[npt.NDArray], float] | None = None, 
                 alpha: float = 20.0, 
                 tol: float = 1e-6, 
                 sup_K: int = 1*10**3, 
                 p: float = 0.90, 
                 c1: float = 0.5, 
                 c2: float = 0.9
        ) -> None:
        self.f: Callable[[npt.NDArray], float] = f # 目的関数 f:R^n -> R 
        if grad_f is None: # 関数の勾配 ∇f:R^n -> R^n
            self.grad_f = lambda y: self.grad(f, y)
        else:
            self.grad_f = grad_f
        if Hesse_f is None: # 関数のヘッセ行列 ∇^2f:R^n -> R^n×n
            self.Hesse_f = lambda y: self.Hesse(f, y)
        else:
            self.Hesse_f = Hesse_f

        self.optimal_x: npt.NDArray | None = None # 最適解 ∈ R^n
        self.min_f: float | None = None # min f(x) ∈ R
        self.xk_seq: list[npt.NDArray] | None = None # 暫定解の列{xk} 可視化のために取っておく

        # パラメータ(最適化手法の講義に準拠)
        self.alpha: float = alpha # バックトラック法の初期パラメータα>0
        self.tol: float = tol # 勾配ベクトルのノルムの許容誤差ε>0(停止条件)
        self.sup_K: int = sup_K # 最大反復回数
        self.p: float = p # バックトラック法の係数(レジュメではρ) 0<p<1が必要
        self.c1: float = c1 # Armijo条件の定数 0<c1<1が必要
        self.c2: float = c2 # Wolfeの曲率条件の定数 0<c1<c2<1が必要
    
    @staticmethod
    def numerical_diff(f: Callable[[npt.NDArray], float], x: npt.NDArray, i: int, h: float = 1e-5):
        dh: npt.NDArray = np.zeros_like(x, dtype=np.float64)
        dh[i] = h
        return (f(x + dh) - f(x - dh)) / (2 * h)

    @classmethod
    def grad(cls, f: Callable[[npt.NDArray], float], x: npt.NDArray) -> npt.NDArray:
        grad: npt.NDArray = np.array([cls.numerical_diff(f, x, i) for i in range(len(x))])
        return grad

    @classmethod
    def Hesse(cls, f: Callable[[npt.NDArray], float], x: npt.NDArray) -> npt.NDArray:
        grad: npt.NDArray = lambda y: cls.grad(f, y)
        hesse: npt.NDArray = np.array([cls.numerical_diff(grad, x, i) for i in range(len(x))])
        return hesse
    
    def GradientDescent(self, x0: npt.NDArray, timeout: float = 300) -> npt.NDArray: # 最急降下法  x0 ∈ R^n:初期点
        f: Callable[[npt.NDArray], float] = self.f
        def wolfe_condition(xk: npt.NDArray, dk: npt.NDArray, a: float) -> bool:
            cond1: bool = f(xk+a*dk) > f(xk) + self.c1 * a * (self.grad_f(xk) @ dk)
            cond2: bool = self.grad_f(xk+a*dk) @ dk < self.c2 * (self.grad_f(xk) @ dk)
            return cond1 or cond2

        start = time.process_time()
        xk_seq: list[npt.NDArray] = []
        xk: npt.NDArray = x0 # 初期点
        for k in range(self.sup_K):
            xk_seq.append(xk) # 暫定解の保存
            dk: npt.NDArray = -self.grad_f(xk) # 降下方向dk
            if np.linalg.norm(dk) < self.tol: # 停止条件(|∇f(xk)|<ε)を満たしていたら終了
                break

            # バックトラック法パート
            a: float = self.alpha # a>0が必要
            count: int = 0 # 下のwhileループで無限ループになる可能性があるので停止条件をかけておく
            while wolfe_condition(xk, dk, a):
                # Wolfeの条件(=Armijo条件と曲率条件)を満たすまでバックトラック
                a *= self.p
                count += 1
                if count >= 10**4:
                    break
            xk = xk + a*dk # 解の更新
            if time.process_time() - start > timeout:
                break
        finish = time.process_time()

        # 解の探索終了
        self.optimal_x = xk # 最適解
        self.min_f = self.f(xk) # min f(x)
        self.xk_seq = np.array(xk_seq) # あとで可視化

        print(f"{k}回目で停止")
        print(f"optimal_x = [{' '.join(['{:.15g}'.format(xk[i]) for i in range(len(xk))])}")
        print(f"実行時間={finish-start}")
        print(f"1反復あたりの実行時間={(finish-start)/k}")
        return xk


    def Newton(self, x0: npt.NDArray, timeout: float = 300) -> npt.NDArray: # ニュートン法  x0 ∈ R^n:初期点
        f: Callable[[npt.NDArray], float] = self.f
        def wolfe_condition(xk: npt.NDArray, dk: npt.NDArray, a: float) -> bool:
            cond1: bool = f(xk+a*dk) > f(xk) + self.c1 * a * (self.grad_f(xk) @ dk)
            cond2: bool = self.grad_f(xk+a*dk) @ dk < self.c2 * (self.grad_f(xk) @ dk)
            return cond1 or cond2

        start = time.process_time()
        xk_seq: list[npt.NDArray] = []
        xk: npt.NDArray = x0 # 初期点
        for k in range(self.sup_K):
            xk_seq.append(xk) # 暫定解の保存
            dk: npt.NDArray = np.linalg.solve(self.Hesse_f(xk), -self.grad_f(xk)) # 降下方向dk: ∇^2f(xk)dk+∇f(xk)=0を満たす(最急降下法との違いはここだけ)
            if np.linalg.norm(dk) < self.tol: # 停止条件(|∇f(xk)|<ε)を満たしていたら終了
                break

            # バックトラック法パート
            a: float = self.alpha # a>0が必要
            count: int = 0 # 下のwhileループで無限ループになる可能性があるので停止条件をかけておく
            while wolfe_condition(xk, dk, a):
                # Wolfeの条件(=Armijo条件と曲率条件)を満たすまでバックトラック
                a *= self.p
                count += 1
                if count >= 10**4:
                    break
            xk = xk + a*dk # 解の更新
            if time.process_time() - start > timeout:
                break
        finish = time.process_time()

        # 解の探索終了
        self.optimal_x = xk # 最適解
        self.min_f = self.f(xk) # min f(x)
        self.xk_seq = np.array(xk_seq) # あとで可視化

        print(f"{k}回目で停止")
        print(f"optimal_x = [{' '.join(['{:.15g}'.format(xk[i]) for i in range(len(xk))])}")
        print(f"実行時間={finish-start}")
        print(f"1反復あたりの実行時間={(finish-start)/k}")
        return xk

    
    def SemiNewton(self, x0: npt.NDArray, timeout: float = 300) -> npt.NDArray: # 準ニュートン法  x0 ∈ R^n:初期点
        f: Callable[[npt.NDArray], float] = self.f
        def wolfe_condition(xk: npt.NDArray, dk: npt.NDArray, a: float) -> bool:
            cond1: bool = f(xk+a*dk) > f(xk) + self.c1 * a * (self.grad_f(xk) @ dk)
            cond2: bool = self.grad_f(xk+a*dk) @ dk < self.c2 * (self.grad_f(xk) @ dk)
            return cond1 or cond2

        start = time.process_time()
        xk_seq: list[npt.NDArray] = []
        xk: npt.NDArray = x0 # 初期点
        n: int = len(x0) # 次元
        Hk: npt.NDArray = np.identity(n) # 単位行列
        for k in range(self.sup_K):
            xk_seq.append(xk) # 暫定解の保存
            dk: npt.NDArray = np.linalg.solve(self.Hesse_f(xk), -self.grad_f(xk)) # 降下方向dk: ∇^2f(xk)dk+∇f(xk)=0を満たす(最急降下法との違いはここだけ)
            if np.linalg.norm(dk) < self.tol: # 停止条件(|∇f(xk)|<ε)を満たしていたら終了
                break

            # バックトラック法パート
            a: float = self.alpha # a>0が必要
            count: int = 0 # 下のwhileループで無限ループになる可能性があるので停止条件をかけておく
            while wolfe_condition(xk, dk, a):
                # Wolfeの条件(=Armijo条件と曲率条件)を満たすまでバックトラック
                a *= self.p
                count += 1
                if count >= 10**4:
                    break
            
            sk = a*dk
            sk_C = sk.reshape(n,1) # n×1の行列(列ベクトル)
            sk_R = sk.reshape(1,n) # 1×nの行列(行ベクトル)
            yk = self.grad_f(xk+sk) - self.grad_f(xk)
            yk_C = yk.reshape(n,1) # n×1の行列(列ベクトル)
            yk_R = yk.reshape(1,n) # 1×nの行列(行ベクトル)
            sk_yk = np.dot(sk,yk) # skとykの内積
            En = np.identity(n) # 単位行列
            # BFGS公式
            Hk = np.dot(np.dot(En-np.dot(sk_C,yk_R)/sk_yk, Hk), (En-np.dot(yk_C, sk_R)/sk_yk)) + np.dot(sk_C,sk_R)/sk_yk
            # DFP公式
            #Hk = Hk - np.dot(np.dot(Hk,yk_C),np.dot(Hk,yk_C).T)/np.dot(np.dot(yk,Hk),yk) + np.dot(sk_C,sk_R)/sk_yk
            xk = xk + sk # 解の更新
            if time.process_time() - start > timeout:
                break
        finish = time.process_time()

        # 解の探索終了
        self.optimal_x = xk # 最適解
        self.min_f = self.f(xk) # min f(x)
        self.xk_seq = np.array(xk_seq) # あとで可視化

        print(f"{k}回目で停止")
        print(f"optimal_x = [{' '.join(['{:.15g}'.format(xk[i]) for i in range(len(xk))])}")
        print(f"実行時間={finish-start}")
        print(f"1反復あたりの実行時間={(finish-start)/k}")
        return xk



def main() -> None:
    pass
    return

if __name__ == "__main__":
    main()

