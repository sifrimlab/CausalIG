# CausalIG

**Causal Integrated Gradients (CIG)** and **Causal Expected Gradients (CEG)**: gradient-based feature
attribution that follows the causal structure of your inputs.

Standard attribution methods such as Integrated Gradients and SHAP treat input features as independent.
When features are causally related, these methods credit whichever feature the model happens to read,
usually a downstream correlate, rather than the upstream driver that actually moved it. A typical example
is a target gene that gets high importance while the transcription factor regulating it gets almost none.

CIG and CEG fix this by integrating gradients along a path that respects a causal graph over the
inputs. Each feature's attribution splits into two parts:

- a **direct effect**: the part flowing from the feature's own variation straight into the model;
- an **indirect effect**: the part mediated through the features it causally influences.

The attributions keep Integrated Gradients' completeness property (they sum to the change in model output)
and cost a fixed number of gradient evaluations. The conditional and causal Shapley baselines need
coalition sampling instead, which becomes prohibitive at high dimension.

## What you need

| Ingredient | What it is | What CausalIG provides |
|---|---|---|
| **A differentiable model** | The model you want to explain: any callable mapping `(batch, d)` to `(batch, C)` or `(batch,)`. | Nothing to do: it is used as a black box through autograd. |
| **A causal graph** | The *structure* of causal relations between the `d` input features, as a `(d, d)` adjacency matrix (`adjacency_matrix[i, j] = 1` means feature `i` causes feature `j`). It comes from prior knowledge such as domain expertise, [OmniPath](https://omnipathdb.org/) or [SCENIC](https://scenic.aertslab.org/). | Edges only, no weights or functional forms. |
| **A surrogate model** | A model that predicts each feature from its causal parents. It learns *how* the mechanisms behind the edges behave. | {class}`~causalig.models.MaskedGlobalLinear` plus a trainer, or {doc}`your own <user_guide/custom_surrogate>`. |

## Where to go next

- **New here?** {doc}`installation`, then the {doc}`quickstart` runs CIG and CEG end to end on a toy
  regulatory graph in under a minute on a CPU.
- **How does it work?** {doc}`user_guide/concepts` derives the direct/indirect split, shows a worked
  example and explains what the surrogate is for.
- **Which method and which arguments?** {doc}`user_guide/methods` compares CIG, CEG and the SHAP
  baselines; {doc}`user_guide/arguments` documents every argument of `attribute()`, the return shapes
  and the memory knobs.
- **Surrogates:** {doc}`user_guide/surrogates` covers the built-in `MaskedGlobalLinear` and its trainer;
  {doc}`user_guide/custom_surrogate` walks through writing, validating and contributing your own.
- **Reproducing the paper:** {doc}`experiments`.
- **Everything else:** the {doc}`API reference <api/index>`.

## Citation

> J. Michiels\*, A. Ismail\*, M. De Vos, A. Sifrim. *Causal Integrated Gradients: Scalable Feature
> Attribution Along Causal Paths.* arXiv preprint, 2026. [arXiv:2610.22126](https://arxiv.org/abs/2610.22126) (\* equal contribution)

```bibtex
@article{michiels2026causalig,
  title   = {Causal Integrated Gradients: Scalable Feature Attribution Along Causal Paths},
  author  = {Michiels, Joran and Ismail, Anis and De Vos, Maarten and Sifrim, Alejandro},
  journal = {arXiv preprint arXiv:2610.22126},
  year    = {2026},
}
```

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
```

```{toctree}
:hidden:
:caption: User guide

user_guide/concepts
user_guide/methods
user_guide/arguments
user_guide/surrogates
user_guide/custom_surrogate
experiments
```

```{toctree}
:hidden:
:caption: Reference

api/index
development
```
