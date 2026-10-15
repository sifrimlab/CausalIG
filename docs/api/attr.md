# `causalig.attr`

```{eval-rst}
.. module:: causalig.attr
```

Attribution methods. {doc}`../user_guide/arguments` explains the shared arguments in depth, and
{doc}`../user_guide/methods` explains when to use which method.

## Causal gradient methods

```{eval-rst}
.. autoclass:: causalig.attr.CausalIntegratedGradients
   :members: attribute

.. autoclass:: causalig.attr.CausalExpectedGradientsMultiStep
   :members: attribute

.. autoclass:: causalig.attr.CausalExpectedGradients
   :members: attribute
```

## Shapley-value baselines

These require the `shap` extra (`pip install causalig[shap]`). They are imported lazily: accessing
`causalig.attr.CausalSHAP` is what imports `shap`.

```{eval-rst}
.. autoclass:: causalig.attr.CausalSHAP
   :members: attribute

.. autoclass:: causalig.attr.ConditionalSHAP
   :members: attribute
```

### Maskers and samplers

The building blocks the SHAP baselines use. They are not exported from `causalig.attr`; import them from
their modules if you need them.

```{eval-rst}
.. autoclass:: causalig.attr._core.causal_shap.CausalMasker
   :members: __call__

.. autofunction:: causalig.attr._core.causal_shap.gaussian_noise

.. autoclass:: causalig.attr._core.conditional_shap.ConditionalMasker
   :members: __call__

.. autoclass:: causalig.attr._core.conditional_shap.GaussianConditionalSampler
   :members: __call__

.. autofunction:: causalig.attr._core.conditional_shap.gaussian_conditional_sampler
```
