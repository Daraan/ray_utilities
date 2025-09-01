# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from unittest.mock import MagicMock

from sphinx.ext.autodoc.mock import _MockModule
from sphinx_autodoc_typehints import mock as autodoc_mock  # pyright: ignore[reportPrivateImportUsage]

# Add the project root to Python path
ROOT = os.path.abspath("../../")
print("cwd:", os.getcwd())
print("root path:", ROOT)
print("ROOT contents:", os.listdir(ROOT))
print("doc contents:", os.listdir(os.path.join(ROOT, "docs")))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.abspath("./"))
import _ray_metrics

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ray_utilities"
copyright = "2025, Daniel Sperber"
author = "Daniel Sperber"


# Mock problematic dependencies that might not be available during doc building
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):  # pyright: ignore[reportIncompatibleMethodOverride]
        return MagicMock()

    @classmethod
    def __ror__(cls, other):
        return object.__ror__(other)

    @classmethod
    def __or__(cls, other):
        return object.__ror__(other)


MOCK_MODULES = [
    "comet_ml",
    "wandb",
    "optuna",
    "jax",
    "flax",
    "distrax",
    "optax",
    "cv2",
    "tree",
    "dotenv",
    "tqdm",
    # "colorlog",
    # "typed_argument_parser",
    # "tap",
    "ray",
    "torch",
    "tensorflow",
    "tensorflow_probability",
    "numpy",
    "debugpy",
    "scipy",
    "pandas",
    "ray_utilities.connectors.exact_samples_to_learner",
    "gymnasium",
    "gym",
    "pyarrow",
    "pyarrow.fs",
    "git",
]
if False:
    # mock it myself
    for mod_name in MOCK_MODULES[:0]:
        sys.modules[mod_name] = Mock()

    sys.modules["ray.rllib.utils.annotations"] = Mock()
    sys.modules["ray.rllib.utils.annotations"].override = lambda _: lambda x: x


class Dummy: ...  # fmt: skip


class Dummy2: ...  # fmt: skip


class TuneMock(_MockModule):
    Trainable = Dummy


tune_mock = TuneMock("ray.tune")


class RayMock(_MockModule):
    tune = tune_mock


autodoc_mock(["ray", "gym", "gymnasium"])
sys.modules["ray"] = RayMock("ray")
sys.modules["gym"] = _MockModule("gym")
sys.modules["gymnasium"] = _MockModule("gymnasium")

sys.modules["ray.rllib.utils.metrics"] = _ray_metrics
sys.modules["ray"].__version__ = "2.48.0+mocked"  # pyright: ignore[reportAttributeAccessIssue]
sys.modules["gym"].__version__ = "0.26.0+mocked"  # pyright: ignore[reportAttributeAccessIssue]
sys.modules["gymnasium"].__version__ = "1.0.0+mocked"  # pyright: ignore[reportAttributeAccessIssue]
sys.modules["ray.rllib.utils.checkpoints"] = _MockModule("ray.rllib.utils.checkpoints")
sys.modules["ray.tune"] = _MockModule("ray.tune")
sys.modules["ray.tune.trainable"] = _MockModule("ray.tune.trainable")


sys.modules["ray.tune"] = tune_mock

sys.modules["ray.tune.trainable"].Trainable = Dummy
sys.modules["ray.rllib.utils.checkpoints"].Checkpointable = Dummy2
sys.modules["ray.rllib.core.rl_module"] = _MockModule("ray.rllib.core.rl_module")
sys.modules["ray.rllib.core.rl_module"].RLModule = Dummy
sys.modules["ray.rllib.core.models.catalog"] = _MockModule("ray.rllib.core.models.catalog")
sys.modules["ray.rllib.core.models.catalog"].Catalog = Dummy
sys.modules["ray.rllib.utils.metrics.stats"] = _MockModule("ray.rllib.utils.metrics.stats")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Ray Utilities"
copyright = "2024, Ray Utilities"
author = "Ray Utilities"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",  # Automatic documentation from docstrings
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "sphinx.ext.intersphinx",  # Cross-reference other projects
    "sphinx.ext.todo",  # TODO directives
    "sphinx.ext.coverage",  # Documentation coverage reports
    "sphinx.ext.imgmath",  # Math equations as images
    "sphinx.ext.githubpages",  # GitHub Pages deployment support
    # Third-party extensions for enhanced documentation
    "sphinx_autodoc_typehints",  # Better type annotation handling
    "sphinx_copybutton",  # Copy button for code blocks
    "myst_parser",  # Markdown support alongside RST
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "testing_utils.py",
    "ray_utilities/connectors/exact_samples_to_learner.py",
    "test/*",
    "setup.py",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are placed after the default static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Theme customization
html_theme_options = {
    "analytics_id": "",  # Add Google Analytics tracking ID if needed
    "analytics_anonymize_ip": False,
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add custom CSS and JS files
html_css_files = []
html_js_files = []

# HTML context for template customization
html_context = {
    "display_github": True,
    "github_user": "Daraan",
    "github_repo": "ray_utilities",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Additional HTML options
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Exclude experimental modules from autodoc
autodoc_mock_imports = MOCK_MODULES

autosummary_generate = True

# Intersphinx mapping for cross-references to external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "ray": ("https://docs.ray.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# MyST parser configuration for Markdown support
myst_enable_extensions = [
    "deflist",  # Definition lists
    "tasklist",  # Task lists
    "colon_fence",  # Colon code fences
]

# Todo extension configuration  # noqa: FIX002
todo_include_todos = True
todo_emit_warnings = True

# Coverage extension configuration
coverage_show_missing_items = True

# Type hints configuration
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True
typehints_use_signature = True
typehints_use_signature_return = True


# For ReadTheDocs compatibility
master_doc = "index"
