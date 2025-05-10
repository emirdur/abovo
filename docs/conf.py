import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from mock_modules import Matrix, DenseLayer, Sequential, LossType, ActivationType
sys.modules['_abovo'] = type('_abovo', (), {
    'Matrix': Matrix,
    'DenseLayer': DenseLayer,
    'Sequential': Sequential,
    'LossType': LossType,
    'ActivationType': ActivationType,
})

project = 'abovo'
author = 'Emir D'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
