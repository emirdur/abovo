API Reference
=============

.. currentmodule:: abovo

.. automodule:: abovo
   :noindex:

Matrix
------

.. autoclass:: abovo.Matrix
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:
   
DenseLayer
----------

.. autoclass:: abovo.DenseLayer
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

Sequential
----------

.. autoclass:: abovo.Sequential
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

Enums
-----

.. data:: LossType
   :annotation: Enum for loss functions
   
   * MSE = 0
   * CROSS_ENTROPY = 1

.. data:: ActivationType
   :annotation: Enum for activation functions
   
   * RELU = 0
   * LEAKY_RELU = 1
   * SIGMOID = 2
   * SOFTMAX = 3

.. data:: MatMulType
   :annotation: Enum for matrix multiplication implementations
   
   * NAIVE = 0
   * BLOCKED = 1
   * SIMD = 2
   * SIMD_MT = 3
   * METAL_GPU = 4