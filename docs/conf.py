import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from conftest import *

import abovo

project = 'abovo'
author = 'Emir D'
release = '0.1.2'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints'
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

nitpicky = False
suppress_warnings = ['autodoc.duplicate_object_description']

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']