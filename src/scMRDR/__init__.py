"""
A package to integrate unpaired multi-omics single-cell data via single-cell Multi-omics Regularized Disentangled Representations.
"""

__version__ = "1.0.0"
__author__ = "Jianle Sun"

import os
import glob

modules = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
__all__ = [os.path.basename(f)[:-3] for f in modules if not f.endswith("__init__.py")]
