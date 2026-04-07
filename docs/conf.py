import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _install_doc_stubs() -> None:
    """Install tiny import-time stubs for optional runtime dependencies.

    The package imports NetKet and JAX at module import time. For documentation
    builds we only need enough structure for autodoc to import modules and read
    docstrings, not the full runtime stack.
    """

    if "jax" not in sys.modules:
        jax = types.ModuleType("jax")
        jax_numpy = types.ModuleType("jax.numpy")
        jax_tree_util = types.ModuleType("jax.tree_util")

        def register_pytree_node_class(cls):
            return cls

        jax_tree_util.register_pytree_node_class = register_pytree_node_class
        jax.tree_util = jax_tree_util
        jax.numpy = jax_numpy
        jax.jit = lambda fn=None, *a, **k: fn
        jax.vmap = lambda fn=None, *a, **k: fn
        jax.lax = types.SimpleNamespace(cond=lambda pred, t, f, *ops: t(*ops) if pred else f(*ops))
        sys.modules["jax"] = jax
        sys.modules["jax.numpy"] = jax_numpy
        sys.modules["jax.tree_util"] = jax_tree_util

    if "netket" not in sys.modules:
        netket = types.ModuleType("netket")
        hilbert = types.ModuleType("netket.hilbert")
        operator = types.ModuleType("netket.operator")
        operator_sum = types.ModuleType("netket.operator._sum")
        operator_prod = types.ModuleType("netket.operator._prod")

        class DiscreteHilbert:
            def __init__(self, size=0):
                self.size = size
                self.local_states = None

        class AbstractOperator:
            def __init__(self, hilbert=None):
                self.hilbert = hilbert

            def __add__(self, other):
                return NotImplemented

            def __radd__(self, other):
                return NotImplemented

            def __matmul__(self, other):
                return NotImplemented

            def __rmatmul__(self, other):
                return NotImplemented

        class DiscreteJaxOperator(AbstractOperator):
            pass

        class SumOperator(AbstractOperator):
            def __init__(self, left=None, right=None):
                self.left = left
                self.right = right

        class ProductOperator(AbstractOperator):
            def __init__(self, left=None, right=None):
                self.left = left
                self.right = right

        hilbert.DiscreteHilbert = DiscreteHilbert
        operator.AbstractOperator = AbstractOperator
        operator.DiscreteJaxOperator = DiscreteJaxOperator
        operator_sum.SumOperator = SumOperator
        operator_prod.ProductOperator = ProductOperator

        netket.hilbert = hilbert
        netket.operator = operator

        sys.modules["netket"] = netket
        sys.modules["netket.hilbert"] = hilbert
        sys.modules["netket.operator"] = operator
        sys.modules["netket.operator._sum"] = operator_sum
        sys.modules["netket.operator._prod"] = operator_prod


_install_doc_stubs()

project = "nkDSL"
author = "The nkDSL Authors"
copyright = "2026, The nkDSL Authors"
release = "0.0.1"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_nb",
]

html_math_renderer = "mathjax"

autosummary_generate = True
autosummary_imported_members = True
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
add_module_names = False
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

templates_path = ["_templates"]
html_static_path = ["_static"]
html_theme = "furo"
html_title = "nkDSL documentation"
# html_logo = "_static/nkDSL.png"
html_favicon = None
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#304180",
        "color-brand-content": "#893168",
    },
    "sidebar_hide_name": False,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "netket": ("https://netket.readthedocs.io/", None),
}

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
nb_execution_mode = "off"
nb_execution_timeout = 600
nb_execution_raise_on_error = True
nb_number_source_lines = False
