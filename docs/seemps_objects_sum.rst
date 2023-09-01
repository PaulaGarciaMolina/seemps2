.. current_module: seemps

.. _mps_sum_class:

***************
MPS combination
***************

Matrix-product states can be operated as normal quantum states, adding,
rescaling or subtracting them, as you would normally do with vectors. However,
when doing so, SeeMPS never really transform the states. Instead, it keeps
track of the operations in a :class:`~seemps.state.MPSSum` structure.

.. admonitions: python

For instance, you could combine two MPS states as follows

    >>> mps1 = random_mps(2, 10)
    >>> mps2 = product_state([1.0, 0.0], 10)
    >>> mps3 = 0.5 * mps1 - 0.3 * mps2
    >>> print(mps3)
    <seemps.state.mps.MPSSum object at 0x00000117FACDF490>

The result in `mps3` is a temporary object with the weights

    >>> mps3.weights
    [1, -1]

and `mps3.states[0]` and `mps3.states[1]` contain rescaled versions of `mps1`
and `mps2`.

In addition to creating the sums implicitly, you can also create them explicitly
using the classes' constructor.

.. autosummary::
    :toctree: generated/

    ~seemps.state.MPSSum

Conversions
-----------

An :class:`MPSSum` can be approximated by a matrix-product state through an
algorithm known as "simplification", explained in
:doc:`the algorithms section <seemps_algorithms>`. It can also be converted
to standard wavefunctions.

.. autosummary::
    :toctree: generated/

    ~seemps.state.MPSSum.toMPS
    ~seemps.state.MPSSum.to_vector
