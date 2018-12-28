# coding: utf-8
# pylint: disable=invalid-name, no-member, trailing-comma-tuple, bad-mcs-classmethod-argument
"""ctypes library of mxnet and helper functions."""
from __future__ import absolute_import

import numpy as np

__all__ = ['MXNetError']
#----------------------------
# library loading
#----------------------------

# pylint: disable=pointless-statement
try:
    basestring
    long
except NameError:
    basestring = str
    long = int
# pylint: enable=pointless-statement

numeric_types = (float, int, long, np.generic)