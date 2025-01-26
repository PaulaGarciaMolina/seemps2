from seemps.state import MPS
from seemps.operators import MPO
from seemps.optimization.arnoldi import arnoldi_eigh, OptimizeResults
from .tools import TestOptimizeCase


class TestArnoldiEigH(TestOptimizeCase):
    def solve(self, H: MPO, state: MPS, **kwdargs) -> OptimizeResults:
        tol = kwdargs.get("tol", 1e-10)
        return arnoldi_eigh(H, state, tol=tol, **kwdargs)
