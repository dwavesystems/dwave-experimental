# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.ifconfig',
]

autosummary_generate = True

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
    }

# The master toctree document.
master_doc = 'index'

add_module_names = False
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.rst']
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "external_links": [
        {
            "url": "https://docs.dwavequantum.com/en/latest/index.html",
            "name": "Documentation",
        },
    ],
    "logo": {
        "image_light": "_static/DWave.svg",
        "image_dark": "_static/DWaveWhite.svg",
        "link": "https://docs.dwavequantum.com/en/latest/index.html",
    },
    "collapse_navigation": True,
    "show_prev_next": False,
}
html_sidebars = {"**": ["sidebar-nav-bs"]}  # remove ads

html_static_path = ['_static']

def setup(app):
   app.add_css_file('theme_overrides.css')

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'networkx': ('https://networkx.github.io/documentation/stable/', None),
    'dwave': ('https://docs.dwavequantum.com/en/latest/', None),
}
# global substitutions
rst_epilog = """
.. |copy| unicode:: U+000A9 .. COPYRIGHT SIGN
.. |deg| unicode:: U+00B0
.. |nbsp| unicode:: U+00A0    .. non-breaking space
.. |nb-| unicode:: U+2011  .. Non-breaking hyphen (e.g., "D |nb-| Wave")
    :trim:
.. |reg| unicode:: U+000AE .. REGISTERED SIGN
.. |tm| unicode::  U+2122
.. |Darr| unicode:: U+02193 .. DOWNWARDS ARROW from docutils/parsers/rst/include/isonum.txt
.. |Uarr| unicode:: U+02191 .. UPWARDS ARROW from docutils/parsers/rst/include/isonum.txt

.. |array-like| replace:: array-like    .. used in dwave-optimization
.. _array-like: https://numpy.org/devdocs/glossary.html#term-array_like

.. |adv2| unicode:: Advantage2
.. |adv2_tm| unicode:: Advantage2 U+2122
.. |cloud| unicode:: Leap
.. _cloud: https://cloud.dwavesys.com/leap
.. |cloud_tm| unicode:: Leap U+2122
.. _cloud_tm: https://cloud.dwavesys.com/leap
.. |dwave_2kq| unicode:: D U+2011 Wave U+00A0 2000Q
.. |dwave_2kq_tm| unicode:: D U+2011 Wave U+00A0 2000Q U+2122
.. |dwave_5kq| unicode:: Advantage
.. |dwave_5kq_tm| unicode:: Advantage U+2122
.. |dwave_short| unicode:: D U+2011 Wave
.. _dwave_short: https://dwavequantum.com
.. |dwave_short_tm| unicode:: D U+2011 Wave U+2122 U+0020
.. |dwave_system| unicode:: D-Wave U+00A0 System
.. |dwave_systems_inc| unicode:: D U+2011 Wave U+00A0 Quantum U+00A0 Inc.

.. |support_email| unicode:: D U+2011 Wave U+00A0 Customer U+00A0 Support
.. _support_email: support@dwavesys.com

.. |ocean_tm| unicode:: Ocean U+2122
.. |ocean_sdk| replace:: Ocean software
.. _ocean_sdk: https://github.com/dwavesystems/dwave-ocean-sdk
"""