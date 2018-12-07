'''
Created on Dec 23, 2016

@author: rch

This module serves for packaging of models for scripts doing parametric
studies presented in the associated wiki pages.

The scripts are grouped into lectures.
'''

from . import scripts.part1_debonding.lecture02_bond \
    as lecture02
from . import scripts.part1_debonding.lecture03_pullout \
    as lecture03
from . import scripts.part1_debonding.lecture04_anchorage \
    as lecture04
from . import scripts.part1_debonding.lecture05_fracture \
    as lecture05
