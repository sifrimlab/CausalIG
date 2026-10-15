# API reference

Generated from the docstrings in the source. The public API is what the three subpackages export:

| Module | Contents |
|---|---|
| {doc}`causalig.attr <attr>` | `CausalIntegratedGradients`, `CausalExpectedGradientsMultiStep`, `CausalExpectedGradients`, and the lazily imported `CausalSHAP` and `ConditionalSHAP`. |
| {doc}`causalig.models <models>` | `MaskedGlobalLinear`, `train_surrogate_model`, `DifferentiableSCM`, `create_autoencoder_from_scm`. |
| {doc}`causalig.utils <utils>` | DAG helpers and experiment logging. |
| {doc}`Internals <internals>` | The attribution base class, the surrogate contract and Jacobian computation, and path sampling. These are useful when extending the library. |

The top-level package re-exports the three gradient methods:

```python
import causalig

causalig.__version__
causalig.CausalIntegratedGradients        # same object as causalig.attr.CausalIntegratedGradients
```

```{toctree}
:maxdepth: 2

attr
models
utils
internals
```
