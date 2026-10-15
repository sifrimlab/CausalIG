"""Sphinx configuration for the CausalIG documentation."""

import os
import sys

# Make `causalig` importable from a plain clone (Read the Docs also pip-installs
# it, see ../.readthedocs.yaml).
sys.path.insert(0, os.path.abspath(".."))

import causalig  # noqa: E402

# -- Project information ------------------------------------------------------

project = "CausalIG"
author = "Joran Michiels, Anis Ismail"
copyright = f"2026, {author}"
version = release = causalig.__version__

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "requirements.txt"]

# Docstrings in this codebase use single backticks for code, so render them as
# (cross-referenced where possible) code rather than italics.
default_role = "py:obj"

# -- MyST (Markdown) ----------------------------------------------------------

myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath", "deflist"]
myst_heading_anchors = 3

# -- autodoc / napoleon -------------------------------------------------------

# `shap` is an optional extra; mock it so the SHAP baselines can be documented
# without installing it on the docs builder.
autodoc_mock_imports = ["shap"]
autodoc_member_order = "bysource"
# Keep torch.nn.Module.forward's generic docstring off undocumented overrides.
autodoc_inherit_docstrings = False
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
python_use_unqualified_type_names = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- HTML output --------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = f"CausalIG {release}"
html_logo = "logo_transparent.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "logo_only": True,
    "navigation_depth": 3,
    "collapse_navigation": False,
}
html_context = {
    "display_github": True,
    "github_user": "sifrimlab",
    "github_repo": "CausalIG",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
