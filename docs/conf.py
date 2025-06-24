# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "Glimmer-ST"
author = "Qiyu Gong"
version = "0.1.0"
copyright = f"{datetime.now().year}, {author}"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- MyST configuration ------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".ipynb": "myst-nb",
}
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
myst_heading_anchors = 3

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# -- Bibliography ------------------------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
]

html_theme_options = {
    "repository_url": "https://github.com/thechenlab/Glimmer",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
}

html_show_sphinx = False
html_show_sourcelink = False

# -- Other settings ----------------------------------------------------------
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
master_doc = "index"
pygments_style = "tango"
pygments_dark_style = "monokai"
