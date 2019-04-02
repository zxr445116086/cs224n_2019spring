#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
sanity_check.py: sanity checks for assignment 5
Usage:
    sanity_check.py highway
    sanity_check.py 1f
    sanity_check.py 1j
    sanity_check.py 2a
    sanity_check.py 2b
    sanity_check.py 2c
    sanity_check.py 2d
"""

from docopt import docopt
from highway import Highway
from sanity_check import DummyVocab

import torch
import torch.nn as nn
import torch.mm.utils

#----------
# CONSTANTS
#----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0