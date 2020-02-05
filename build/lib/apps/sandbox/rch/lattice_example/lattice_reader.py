'''
Created on Nov 25, 2019

@author: rch
'''

import os

import numpy as np
home_dir = os.path.expanduser('~')


ldir = os.path.join(home_dir, 'simdb', 'simdata', 'lattice_example')
nodes = np.loadtxt(os.path.join(ldir, 'nodes.inp'),
                   skiprows=1, usecols=(1, 2, 3))
vertices = np.loadtxt(os.path.join(ldir, 'vertices.inp'),
                      skiprows=1, usecols=(1, 2, 3))
elements = np.loadtxt(os.path.join(ldir, 'mechElems.inp'),
                      skiprows=1, usecols=(1, 2, 3), dtype=np.int_)
print(nodes)
print(vertices)
print(elements)
print(nodes[elements])
