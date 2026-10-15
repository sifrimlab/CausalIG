# Development

## Repository layout

```text
causalig/                         # the pip-installable library (torch + numpy only)
├── attr/                         # attribution methods (public: causalig.attr)
│   ├── _core/
│   │   ├── attribution.py                   # Attribution base class
│   │   ├── causal_integrated_gradients.py   # CIG
│   │   ├── causal_expected_gradients.py     # CEG / CEG-MultiStep (shared gradient core)
│   │   ├── causal_shap.py                   # CausalSHAP          ([shap] extra)
│   │   └── conditional_shap.py              # ConditionalSHAP     ([shap] extra)
│   └── _utils/
│       ├── path.py               # path sampling (linear grid, stratified)
│       └── surrogate.py          # surrogate contract + surrogate Jacobian computation
├── models/                       # public: causalig.models
│   ├── surrogate.py              # MaskedGlobalLinear, BasisExpansion, trainer
│   └── scm.py                    # DifferentiableSCM
└── utils/                        # public: causalig.utils
    ├── graph.py                  # DAG utilities
    └── logging_utils.py          # experiment logger setup

experiments/                      # paper benchmark (not installed; needs [experiments])
scripts/                          # run_* / plot_* / print_* entry points
tests/                            # pytest suite
docs/                             # this site (Sphinx + MyST)
```

## Tests and linting

```bash
pip install -e ".[dev,experiments]"
pytest                 # full suite
ruff check .           # lint
black --check .        # formatting
```

The same three commands run in CI (`.github/workflows/ci.yml`) on Python 3.10–3.13.

## Building the documentation

The site is built with [Sphinx](https://www.sphinx-doc.org/), the
[Read the Docs theme](https://sphinx-rtd-theme.readthedocs.io/) and
[MyST](https://myst-parser.readthedocs.io/), so pages are Markdown. API pages are generated from
docstrings, which use Google style (`Args:`, `Returns:`).

```bash
pip install -e .
pip install -r docs/requirements.txt
make -C docs html      # output in docs/_build/html/index.html
make -C docs strict    # fail on any warning, as Read the Docs and CI do
```

To add a page, create a `.md` file under `docs/` and list it in a `toctree` in `docs/index.md`. To
document a new public object, add an `autoclass` or `autofunction` entry to the matching page in
`docs/api/`.

## Deploying to Read the Docs

The repository is ready to build on [Read the Docs](https://readthedocs.org/). `.readthedocs.yaml` at
the root pins the build:

- Ubuntu 24.04 and Python 3.12;
- the CPU-only torch wheel, which avoids downloading the CUDA build;
- `docs/requirements.txt`;
- the package itself, installed with `pip install .`;
- a Sphinx build of `docs/conf.py` with warnings treated as errors.

`shap` is mocked rather than installed.

One-time setup:

1. Sign in to Read the Docs with GitHub and choose **Import a Project**, then pick `sifrimlab/CausalIG`.
   Org repositories may need an admin to grant Read the Docs access.
2. Leave the configuration-file path at its default, `.readthedocs.yaml`, and trigger a build.
3. In **Settings**, choose the default branch and version (`latest` tracks the default branch). Enable
   **Build pull requests** to get a preview for every PR.

Every push to the default branch then rebuilds the site. Tagging a release (for example `v0.1.0`) and
activating that version gives a frozen copy of the docs for that release.
