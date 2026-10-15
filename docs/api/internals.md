# Internals

These objects are not part of the stable public API: they live in private modules and may change.
They are documented here because you need them when extending the library, for example when writing a
new attribution method or surrogate (see {doc}`../user_guide/custom_surrogate`).

## Attribution base class

```{eval-rst}
.. autoclass:: causalig.attr._core.attribution.Attribution
   :members: attribute
   :private-members: _format_input, _format_output, _validate_inputs, _convergence_delta
```

## Surrogate contract and Jacobians

```{eval-rst}
.. autoclass:: causalig.attr._utils.surrogate.ModelInterface
   :members: forward

.. autodata:: causalig.attr._utils.surrogate.MASK_VAL
   :annotation: = 0.0

.. autofunction:: causalig.attr._utils.surrogate.compute_surrogate_gradients
```

## Path sampling

```{eval-rst}
.. autoclass:: causalig.attr._utils.path.PathSampler
   :members: sample

.. autoclass:: causalig.attr._utils.path.LinearPathSampler
   :members: sample, sample_pairwise_stochastic

.. autofunction:: causalig.attr._utils.path.linear_path_samples
```
