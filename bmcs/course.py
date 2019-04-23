'''
Created on Dec 23, 2016

@author: rch

This module serves for packaging of models for scripts doing parametric
studies presented in the associated wiki pages.

The scripts are grouped into lectures.
'''

from .scripts.part1_debonding import lecture02_bond \
    as lecture02
from .scripts.part1_debonding import lecture03_pullout \
    as lecture03
from .scripts.part1_debonding import lecture04_anchorage \
    as lecture04
from .scripts.part1_debonding import lecture05_fracture \
    as lecture05
