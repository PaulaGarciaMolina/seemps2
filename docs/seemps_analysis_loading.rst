.. currentmodule:: seemps

.. _analysis_loading:

****************
Function Loading
****************

The SeeMPS library provides several methods to load univariate and multivariate functions in MPS and MPO structures. In the following, the most important are listed.

Tensorized operations
---------------------
These methods are useful to construct MPS corresponding to domain discretizations, and compose them using tensor products and sums to construct multivariate domains.

.. autosummary::
    :toctree: generated/
    
    ~seemps.analysis.mesh.RegularInterval
    ~seemps.analysis.mesh.ChebyshevInterval
    ~seemps.analysis.factories.mps_interval
    ~seemps.analysis.factories.mps_tensor_product
    ~seemps.analysis.factories.mps_tensor_sum

Tensor cross-interpolation (TT-Cross)
-------------------------------------
These methods are useful to compose MPS or MPO representations of black-box functions using tensor-train cross-interpolation (TT-Cross). See :doc:`algorithms/tt-cross`

.. autosummary::
    :toctree: generated/
    
    ~seemps.analysis.cross.black_box.BlackBoxLoadMPS
    ~seemps.analysis.cross.black_box.BlackBoxLoadTT
    ~seemps.analysis.cross.black_box.BlackBoxLoadMPO
    ~seemps.analysis.cross.black_box.BlackBoxComposeMPS
    ~seemps.analysis.cross.cross_maxvol
    ~seemps.analysis.cross.cross_dmrg
    ~seemps.analysis.cross.cross_greedy
    
Chebyshev expansions
--------------------
These methods are useful to compose univariate function on generic initial MPS or MPO and compute MPS approximations of functions.
See :doc:`algorithms/chebyshev`.

.. autosummary::
    :toctree: generated/
    
    ~seemps.analysis.chebyshev.cheb2mps
    ~seemps.analysis.chebyshev.cheb2mpo
    ~seemps.analysis.chebyshev.interpolation_coefficients
    ~seemps.analysis.chebyshev.projection_coefficients
    ~seemps.analysis.chebyshev.estimate_order


Multiscale interpolative constructions
--------------------------------------
These methods are useful to construct polynomial interpolants of univariate functions in MPS using the Lagrange interpolation framework.
See :doc:`algorithms/lagrange`.

.. autosummary::
    :toctree: generated/
    
    ~seemps.analysis.lagrange.lagrange_basic
    ~seemps.analysis.lagrange.lagrange_rank_revealing
    ~seemps.analysis.lagrange.lagrange_local_rank_revealing

Generic polynomial constructions
--------------------------------
These methods are useful to construct generic polynomials in the monomial basis from a collection of coefficients.

.. autosummary::
    :toctree: generated/
    
    ~seemps.analysis.polynomials.mps_from_polynomial