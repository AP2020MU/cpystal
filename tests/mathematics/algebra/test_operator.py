from __future__ import annotations

from cpystal.mathematics.algebra.operator import JOperator

def test_commutation_001() -> None:
    for J in [1/2, 1, 3/2, 2, 5/2, 3, 7/2]:
        Jx: JOperator = JOperator.Jx(J)
        Jy: JOperator = JOperator.Jy(J)
        Jz: JOperator = JOperator.Jz(J)
        Jp: JOperator = JOperator.Jp(J)
        Jm: JOperator = JOperator.Jm(J)
        assert Jx ^ Jy == Jy ^ Jx
        assert Jx % Jy == -Jy % Jx
        assert Jx @ Jy == Jx * Jy
        assert Jx % Jy == 1.0j * Jz
        assert Jy % Jz == 1.0j * Jx
        assert Jz % Jx == 1.0j * Jy
        assert Jp % Jz == -Jp
        assert Jm % Jz == +Jm
        assert Jp @ Jm == J*(J+1) - Jz**2 + Jz
        assert Jm @ Jp == J*(J+1) - Jz**2 - Jz
        assert Jp % Jm == 2 * Jz
        assert (Jx**2 + Jy**2 + Jz**2) % Jp == 0
        assert (Jx**2 + Jy**2 + Jz**2) % Jm == 0
        assert Jy**2 % Jz == 1.0j * Jx ^ Jy
        assert Jz**2 % Jx == 1.0j * Jy ^ Jz
        assert Jx**2 % Jy == 1.0j * Jx ^ Jz

def test_commutation_002() -> None:
    for J in [1/2, 1, 3/2, 2, 5/2, 3, 7/2]:
        Jp: JOperator = JOperator.Jp(J)
        Jm: JOperator = JOperator.Jm(J)
        Jz: JOperator = JOperator.Jz(J)
        for k in range(1,5):
            assert (Jp**k) % Jz == -k*(Jp**k)
            assert (Jm**k) % Jz == +k*(Jm**k)
            assert Jm % (Jp**k) == -Jp**(k-1) * (2*k*Jz + k*(k-1))
            assert Jm % (Jp**k * Jz) == -Jp**(k-1) * ((2*k+1)*Jz**2 + (k**2-k-1)*Jz - J*(J+1))
