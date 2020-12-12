import os
import sys

# enable autodoc to load local modules
sys.path.insert(0, os.path.abspath("."))

project = "RhymeCheck"
copyright = "2020, Sourab Jha, Shubham Mishra, Kartavya Kothari"
author = "Sourab Jha, Shubham Mishra, Kartavya Kothari"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.intersphinx"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
html_theme = "alabaster"
html_static_path = ["_static"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None)
}
html_theme_options = {"nosidebar": True}
