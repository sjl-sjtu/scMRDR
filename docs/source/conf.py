# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'scMRDR'
copyright = '2025, Jianle Sun'
author = 'Jianle Sun'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []

# templates_path = ['_templates']
# exclude_patterns = []


import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))

extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = []
autoclass_content = 'both'
napoleon_use_rtype = False
napoleon_custom_sections = [('Returns', 'params_style')]

autodoc_default_options = {
    'members': True,            
    'undoc-members': True,      
    'show-inheritance': True,   
    'private-members': True,    
}

autodoc_default_flags = [
    'members',
    'undoc-members',
    'private-members',
    'special-members',
    'inherited-members',
    'show-inheritance'
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

master_doc = 'index'
source_suffix = ['.rst']
