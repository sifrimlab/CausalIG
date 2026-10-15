# Installation

`causalig` needs Python ≥ 3.10. Its only hard dependencies are **PyTorch ≥ 2.0** and **NumPy**.

## From a clone (recommended for now)

```bash
git clone https://github.com/sifrimlab/CausalIG.git
cd CausalIG
pip install -e .
```

Or install straight from GitHub without cloning:

```bash
pip install "causalig @ git+https://github.com/sifrimlab/CausalIG.git"
```

:::{tip}
On Linux, `pip install torch` pulls the CUDA build by default, which is several GB. On a CPU-only machine,
install the CPU wheel *before* installing `causalig`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
:::

## With conda

The repository ships an `environment.yml` that creates a `causalig` environment:

```bash
conda env create -f environment.yml
conda activate causalig
pip install -e ".[shap,experiments]"   # adds the extras needed for the paper's experiments
```

## Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `shap` | `shap` | The Shapley-value baselines {class}`~causalig.attr.CausalSHAP` and {class}`~causalig.attr.ConditionalSHAP`. |
| `experiments` | `captum`, `sempler`, `matplotlib`, `scipy` | Running `experiments/` and `scripts/`, which reproduce the paper's synthetic benchmark (see {doc}`experiments`). |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `black`, `shap` | Running the test suite and linters (see {doc}`development`). |

```bash
pip install -e ".[shap]"
pip install -e ".[experiments]"
pip install -e ".[dev]"
```

`import causalig` never imports `shap`: the SHAP baselines are loaded lazily the first time you access
`causalig.attr.CausalSHAP` or `causalig.attr.ConditionalSHAP`.

## Check the install

```bash
python -c "import causalig; print(causalig.__version__)"
```
