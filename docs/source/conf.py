# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ultrasonic_ml'
copyright = '2026, Xinqiao Zhang'
author = 'Xinqiao Zhang'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os, sys
sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'myst_nb',
]

autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

html_theme = 'furo'

# myst-nb settings
nb_execution_mode = "off"  # "auto" or "cache" if you want notebooks re-run on build


import shutil
from pathlib import Path

def remove_api_dir(app, config):
    api_dir = Path(app.srcdir) / "api"
    if api_dir.exists():
        shutil.rmtree(api_dir)

def setup(app):
    app.connect("config-inited", remove_api_dir)
    
    
    