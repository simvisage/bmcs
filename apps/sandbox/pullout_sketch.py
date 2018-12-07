'''
Created on Dec 6, 2018

@author: rch
'''


from bmcs.pullout.pullout_multilinear import \
    PullOutModel
import matplotlib.patches as patches

import numpy as np
import pylab as p

if __name__ == '__main__':
    po = PullOutModel()

    L = po.geometry.L_x
    H = 0.1 * L
    print('L', L)
    print('H', H)
    rect = patches.Rectangle(
        (0, 0), L, H, linewidth=1, edgecolor='black',
        facecolor='lightgray')
    ax = p.subplot(111)
    ax.add_patch(rect)

    p.show()
