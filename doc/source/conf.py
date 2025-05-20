# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os

# before pymiediff import:
# lower tqdm update rate to avoid progress bar "spam"
os.environ["TQDM_MININTERVAL"] = "5"

import warnings
import pymiediff

project = "pymiediff-doc"
copyright = "2025, O. K. Jackson, P. R. Wiecha"
author = "O. K. Jackson, P. R. Wiecha"

# The short X.Y version.
version = pymiediff.__version__
# The full version, including alpha/beta/rc tags.
release = version


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "pyvista.ext.plot_directive",
]


exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
pygments_style = "sphinx"

# The master toctree document.
master_doc = "index"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
# html_theme = "pydata_sphinx_theme"
html_theme = "sphinx_book_theme"
html_theme_options = {
    "logo": {
        "alt_text": "pyMieDiff - Home",
        "text": "Mie + autodiff",
        "image_light": "_static/pymiediff_logo-light.png",
        "image_dark": "_static/pymiediff_logo-dark.png",
    }
}

# -- options for autodoc -------------------------------------------------
autosummary_generate = True
templates_path = ['_templates']


# -- Options for automatic gallery -------------------------------------------------
# https://sphinx-gallery.github.io/stable/configuration.html
image_scrapers = ("matplotlib")
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "auto_gallery",  # path to where to save gallery generated output
    "filename_pattern": "/ex_",  # prefix of files to execute for image generation
    "ignore_pattern": "dev_",  # prefix / suffix of ignored files
    "remove_config_comments": True,  # remove sphinx config comments
    "image_scrapers": image_scrapers,
    "reset_modules": ("matplotlib"),
    "compress_images": ("images", "thumbnails"),
    "within_subsection_order": "FileNameSortKey",  # example order
    "show_memory": True,
    # "parallel": 4,  # faster evaluation, but no memory profiling
}

# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)
