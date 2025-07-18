from operator import is_
import numpy as np
import scipy.linalg  # type: ignore
from dataclasses import dataclass
from typing import Callable

from .cross import (
    BlackBox,
    CrossResults,
    CrossInterpolation,
    CrossStrategy,
    maxvol_square,
    _check_convergence,
    i2s
)
from ..sampling import random_mps_indices
from ...state import Strategy, DEFAULT_TOLERANCE
from scipy.linalg import svd
from ...state.core import destructively_truncate_vector
from ...truncate import SIMPLIFICATION_STRATEGY
from ...tools import make_logger

DEFAULT_CROSS_STRATEGY = SIMPLIFICATION_STRATEGY.replace(
    normalize=False,
    tolerance=DEFAULT_TOLERANCE**2,
    simplification_tolerance=DEFAULT_TOLERANCE**2,
)

# TODO: Implement local error evaluation

def get_max_D(mps):
    return max([max(t.shape) for t in mps])


@dataclass
class CrossStrategyDMRG(CrossStrategy):
    strategy: Strategy = DEFAULT_CROSS_STRATEGY
    tol_maxvol_square: float = 1.05
    maxiter_maxvol_square: int = 10
    """
    Dataclass containing the parameters for the DMRG-based TCI.
    The common parameters are documented in the base `CrossStrategy` class.

    Parameters
    ----------
    strategy : Strategy, default=DEFAULT_CROSS_STRATEGY
        Simplification strategy used at the truncation of Schmidt values
        at each SVD split of the DMRG superblocks.
    tol_maxvol_square : float, default=1.05
        Sensibility for the square maxvol decomposition.
    maxiter_maxvol_square : int, default=10
        Maximum number of iterations for the square maxvol decomposition.
    """

class CrossInterpolationDMRG(CrossInterpolation):
    def __init__(
        self,
        black_box: BlackBox,
        initial_point: np.ndarray | None = None,
        cross_strategy: CrossStrategyDMRG = CrossStrategyDMRG(),
        with_cache: bool = False
    ):
        self.cross_strategy = cross_strategy  # Make sure this is set first
        if initial_point is None:
            initial_point = random_mps_indices(
                black_box.physical_dimensions,
                num_indices=1,
                allowed_indices=getattr(black_box, "allowed_indices", None),
                rng=self.cross_strategy.rng,
            )
        super().__init__(black_box, initial_point)
        self.with_cache = with_cache
        self.cache = {}
        self.i_opt = None
        self.y_opt = None
        self.opt_trajectory = []
        self.k = None
        self.forward = None
    def refresh(self):
        self.cache = {}
        self.opt_trajectory = []
        self.black_box.evals = 0
    def sample_superblock(self, k: int) -> np.ndarray:
        i_l, i_g = self.I_l[k], self.I_g[k + 1]
        i_s1, i_s2 = self.I_s[k], self.I_s[k + 1]
        mps_indices = self.combine_indices(i_l, i_s1, i_s2, i_g)
        results = []
        new_rows = []
        new_rows_strs = []
        new_rows_pos = []

        for pos, idx in enumerate(mps_indices):
            idx_str = i2s(idx)
            if self.with_cache and idx_str in self.cache:
                results.append(self.cache[idx_str])
            else:
                results.append(None)
                new_rows.append(idx)
                new_rows_strs.append(idx_str)
                new_rows_pos.append(pos)

        if new_rows:
            new_rows = np.array(new_rows)
            new_vals = self.black_box[new_rows]  # <--- Vectorized batch call!

            for idx_str, val, pos in zip(new_rows_strs, new_vals, new_rows_pos):
                self.cache[idx_str] = val
                results[pos] = val
        evals = np.array(results).reshape(
            (len(i_l), len(i_s1), len(i_s2), len(i_g))
        )
        return evals, mps_indices
    
    def smooth_func(self, y, y0=0., opt=1.):
        """Smooth function that transforms max to min."""
        return np.pi/2 - np.arctan(y - y0)
    
    def find_opt(self, I, y, i_opt, y_opt, is_min=True):
        """Find the minimum or maximum value on a set of sampled points."""
        y = y.flatten()
        if is_min:
            ind = np.argmin(y)
        else:
            ind = np.argmax(y)
        y_opt_curr = y[ind]
        self.opt_trajectory.append((i2s(I[ind, :]), y_opt_curr, self.black_box.evals, get_max_D(self.mps)))
        if is_min and y_opt is not None and y_opt_curr >= y_opt:
            return i_opt, y_opt
        if not is_min and y_opt is not None and y_opt_curr <= y_opt:
            return i_opt, y_opt

        return I[ind, :], y_opt_curr

    def _update_dmrg(
        self,
        k: int,
        forward: bool,
    ) -> None:
        superblock, _ = self.sample_superblock(k)
        r_l, s1, s2, r_g = superblock.shape
        A = superblock.reshape(r_l * s1, s2 * r_g)
        ## Non-destructive SVD
        U, S, V = svd(A, check_finite=False)
        destructively_truncate_vector(S, self.cross_strategy.strategy)
        r = S.size
        U, S, V = U[:, :r], np.diag(S), V[:r, :]
        ##
        if forward:
            if k < self.sites - 2:
                C = U.reshape(r_l * s1, r)
                Q, _ = scipy.linalg.qr(
                    C, mode="economic", overwrite_a=True, check_finite=False
                )  # type: ignore
                I, G = maxvol_square(
                    Q,
                    self.cross_strategy.maxiter_maxvol_square,
                    self.cross_strategy.tol_maxvol_square,  # type: ignore
                )
                self.I_l[k + 1] = self.combine_indices(self.I_l[k], self.I_s[k])[I]
                self.mps[k] = G.reshape(r_l, s1, r)
                self.mps[k+1] =  (S @ V).reshape(r, s2, r_g)
            else:
                self.mps[k] = U.reshape(r_l, s1, r)
                self.mps[k + 1] = (S @ V).reshape(r, s2, r_g)
        else:
            if k > 0:
                R = V.reshape(r, s2 * r_g)
                Q, _ = scipy.linalg.qr(  # type: ignore
                    R.T, mode="economic", overwrite_a=True, check_finite=False
                )
                I, G = maxvol_square(
                    Q,
                    self.cross_strategy.maxiter_maxvol_square,
                    self.cross_strategy.tol_maxvol_square,  # type: ignore
                )
                self.I_g[k] = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])[I]
                self.mps[k + 1] = (G.T).reshape(r, s2, r_g)
                self.mps[k] = (U @ S).reshape(r_l, s1, r)
            else:
                self.mps[k] = (U @ S).reshape(r_l, s1, r)
                self.mps[k + 1] = V.reshape(r, s2, r_g)

    def _update_dmrg_opt(
        self,
        k: int,
        forward: bool,
        is_min: bool=True
    ) -> None:
        superblock, mps_indices = self.sample_superblock(k)
        if self.y_opt is None:
            self.y_opt = np.inf if is_min else -np.inf
        self.i_opt, self.y_opt = self.find_opt(mps_indices, superblock, self.i_opt, self.y_opt, is_min)
        r_l, s1, s2, r_g = superblock.shape
        if is_min:
            superblock = self.smooth_func(superblock, self.y_opt)
        A = superblock.reshape(r_l * s1, s2 * r_g)
        ## Non-destructive SVD
        U, S, V = svd(A, check_finite=False)
        destructively_truncate_vector(S, self.cross_strategy.strategy)
        r = S.size
        U, S, V = U[:, :r], np.diag(S), V[:r, :]
        ##
        if forward:
            if k < self.sites - 2:
                C = U.reshape(r_l * s1, r)
                Q, _ = scipy.linalg.qr(
                    C, mode="economic", overwrite_a=True, check_finite=False
                )  # type: ignore
                I, G = maxvol_square(
                    Q,
                    self.cross_strategy.maxiter_maxvol_square,
                    self.cross_strategy.tol_maxvol_square,  # type: ignore
                )
                self.I_l[k + 1] = self.combine_indices(self.I_l[k], self.I_s[k])[I]
                self.mps[k] = G.reshape(r_l, s1, r)
                self.mps[k+1] =  (S @ V).reshape(r, s2, r_g)
            else:
                self.mps[k] = U.reshape(r_l, s1, r)
                self.mps[k + 1] = (S @ V).reshape(r, s2, r_g)
        else:
            if k > 0:
                R = V.reshape(r, s2 * r_g)
                Q, _ = scipy.linalg.qr(  # type: ignore
                    R.T, mode="economic", overwrite_a=True, check_finite=False
                )
                I, G = maxvol_square(
                    Q,
                    self.cross_strategy.maxiter_maxvol_square,
                    self.cross_strategy.tol_maxvol_square,  # type: ignore
                )
                self.I_g[k] = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])[I]
                self.mps[k + 1] = (G.T).reshape(r, s2, r_g)
                self.mps[k] = (U @ S).reshape(r_l, s1, r)
            else:
                self.mps[k] = (U @ S).reshape(r_l, s1, r)
                self.mps[k + 1] = V.reshape(r, s2, r_g)

    def cross_dmrg(self,
        callback: Callable | None = None,
    ) -> CrossResults:
        """
        Computes the MPS representation of a black-box function using the tensor cross-approximation (TCI)
        algorithm based on two-site optimizations in a DMRG-like manner.
        The black-box function can represent several different structures. See `black_box` for usage examples.

        Parameters
        ----------
        callback : Callable, optional
            A callable called on the MPS after each iteration.
            The output of the callback is included in a list 'callback_output' in CrossResults.
        Returns
        -------
        CrossResults
            A dataclass containing the MPS representation of the black-box function,
            among other useful information.
        """
        

        converged = False
        callback_output = []
        forward = True
        with make_logger(2) as logger:
            for i in range(self.cross_strategy.maxiter):
                # Forward sweep
                forward = True
                for k in range(self.sites - 1):
                    self._update_dmrg(k, forward)
                if callback:
                    callback_output.append(callback(self.mps, logger=logger))
                if converged := _check_convergence(self, i, self.cross_strategy, logger):
                    break
                # Backward sweep
                forward = False
                for k in reversed(range(self.sites - 1)):
                    self._update_dmrg(k, forward)
                if callback:
                    callback_output.append(callback(self.mps, logger=logger))
                if converged := _check_convergence(self, i, self.cross_strategy, logger):
                    break
            if not converged:
                logger("Maximum number of TT-Cross iterations reached")
        points = self.indices_to_points(forward)
        return CrossResults(
            mps=self.mps,
            points=points,
            evals=self.black_box.evals,
            callback_output=callback_output,
            cache=self.cache,
        )

    def optimize(self, is_min : bool = True,
                 max_func_evals : int = 10000,
                 forward : bool = True,
        callback: Callable | None = None,
    ) -> CrossResults:
        """
        Computes the MPS representation of a black-box function using the tensor cross-approximation (TCI)
        algorithm based on two-site optimizations in a DMRG-like manner.
        The black-box function can represent several different structures. See `black_box` for usage examples.

        Parameters
        ----------
        is_min : bool, optional
            If True the minimum is returned.
        max_func_evals : int, default=10000
            Maximum number of function evaluations.
        forward : bool, optional
            If True, first pass is forward.
        callback : Callable, optional
            A callable called on the MPS after each iteration.
            The output of the callback is included in a list 'callback_output' in CrossResults.
        Returns
        -------
        CrossResults
            A dataclass containing the MPS representation of the black-box function,
            among other useful information.
        """
        

        converged = False
        callback_output = []
        with make_logger(2) as logger:
            for i in range(self.cross_strategy.maxiter):
                stop = False
                # Forward sweep
                if forward:
                    for k in range(self.sites - 1):
                        self._update_dmrg_opt(k, forward, is_min)
                        if self.black_box.evals > max_func_evals:
                            stop=True
                            self.forward = forward
                            self.k = k + 1
                            break
                    forward = not forward
                    if callback:
                        callback_output.append(callback(self.mps, logger=logger))
                    if stop:
                        logger("Maximum number of evaluations reached")
                        break
                    #if converged := _check_convergence(self, i, self.cross_strategy, logger) or stop:
                        #break
                # Backward sweep
                else:
                    for k in reversed(range(self.sites - 1)):
                        self._update_dmrg_opt(k, forward, is_min)
                        if self.black_box.evals > max_func_evals:
                            stop=True
                            self.forward = forward
                            self.k = k - 1
                            break
                    forward = not forward
                    if callback:
                        callback_output.append(callback(self.mps, logger=logger))
                    #if converged := _check_convergence(self, i, self.cross_strategy, logger) or stop:
                        #break
                    if stop:
                        logger("Maximum number of evaluations reached")
                        break
            if not converged and not stop:
                logger("Maximum number of TT-Cross iterations reached")
        evals = self.black_box.evals
        #points = self.indices_to_points(forward)
        dtype = [('idx', 'U50'), ('F', 'f8'), ('evals', 'i4'), ('max_D', 'i4')]
        self.opt_trajectory = np.array(self.opt_trajectory, dtype)
        return CrossResults(
            mps=self.mps,
            evals=evals,
            callback_output=callback_output,
            cache=self.cache,
            opt_trajectory=self.opt_trajectory,
            i_opt=self.i_opt,
            y_opt=self.y_opt
        )
    
    def reoptimize(self, is_min : bool = True,
                 max_func_evals : int = 10000,
                 forward : bool = True,
        callback: Callable | None = None,
    ) -> CrossResults:
        """
        Computes the MPS representation of a black-box function using the tensor cross-approximation (TCI)
        algorithm based on two-site optimizations in a DMRG-like manner.
        The black-box function can represent several different structures. See `black_box` for usage examples.

        Parameters
        ----------
        is_min : bool, optional
            If True the minimum is returned.
        max_func_evals : int, default=10000
            Maximum number of function evaluations.
        forward : bool, optional
            If True, first pass is forward.
        callback : Callable, optional
            A callable called on the MPS after each iteration.
            The output of the callback is included in a list 'callback_output' in CrossResults.
        Returns
        -------
        CrossResults
            A dataclass containing the MPS representation of the black-box function,
            among other useful information.
        """
        self.refresh()
        converged = False
        callback_output = []
        forward = self.forward
        with make_logger(2) as logger:
            for i in range(self.cross_strategy.maxiter):
                stop = False
                # Forward sweep
                if forward:
                    if i == 0:
                        start_k = self.k 
                    else:
                        start_k = 0
                    for k in range(start_k, self.sites - 1):
                        self._update_dmrg_opt(k, forward, is_min)
                        if self.black_box.evals > max_func_evals:
                            stop=True
                            self.forward = forward
                            self.k = k + 1
                            break
                    forward = not forward
                    if callback:
                        callback_output.append(callback(self.mps, logger=logger))
                    if stop:
                        logger("Maximum number of evaluations reached")
                        break
                    #if converged := _check_convergence(self, i, self.cross_strategy, logger) or stop:
                        #break
                # Backward sweep
                else:
                    if i == 0:
                        start_k = self.k + 1
                    else:
                        start_k = self.sites - 1
                    for k in reversed(range(start_k)):
                        self._update_dmrg_opt(k, forward, is_min)
                        if self.black_box.evals > max_func_evals:
                            stop=True
                            self.forward = forward
                            self.k = k - 1
                            break
                    forward = not forward
                    if callback:
                        callback_output.append(callback(self.mps, logger=logger))
                    #if converged := _check_convergence(self, i, self.cross_strategy, logger) or stop:
                        #break
                    if stop:
                        logger("Maximum number of evaluations reached")
                        break
            if not converged and not stop:
                logger("Maximum number of TT-Cross iterations reached")

        evals = self.black_box.evals
        points = self.indices_to_points(forward)
        dtype = [('idx', 'U50'), ('F', 'f8'), ('evals', 'i4'), ('max_D', 'i4')]
        self.opt_trajectory = np.array(self.opt_trajectory, dtype)
        return CrossResults(
            mps=self.mps,
            points=points,
            evals=evals,
            callback_output=callback_output,
            cache=self.cache,
            opt_trajectory=self.opt_trajectory,
            i_opt=self.i_opt,
            y_opt=self.y_opt
        )
    

