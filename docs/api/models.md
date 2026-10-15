# `causalig.models`

```{eval-rst}
.. module:: causalig.models
```

Surrogate models, their training utilities, and a differentiable structural causal model.
{doc}`../user_guide/surrogates` is the narrative guide, and {doc}`../user_guide/custom_surrogate` explains
how to write your own.

## Surrogates

```{eval-rst}
.. autoclass:: causalig.models.MaskedGlobalLinear
   :members: forward, compute_input_jacobian

.. autoclass:: causalig.models.surrogate.BasisExpansion
   :members: forward, derivative
```

## Training

```{eval-rst}
.. autofunction:: causalig.models.train_surrogate_model
```

## Structural causal models

```{eval-rst}
.. autoclass:: causalig.models.DifferentiableSCM
   :members: forward, forward_do

.. autofunction:: causalig.models.create_autoencoder_from_scm
```
