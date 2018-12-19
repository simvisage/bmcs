'''
Created on Dec 6, 2018

@author: rch
'''

from bmcs.pullout.pullout_multilinear import \
    PullOutModel

import numpy as np
import pylab as p

if __name__ == '__main__':
    po = PullOutModel()

    for L in [10, 20, 30, 40]:
        print ''
        po.geometry.L_x = L
        po.run()

        P_t = po.get_P_t()
        w_0, w_L = po.get_w_t()

        m_idx = np.argmax(P_t)
        P_max = P_t[m_idx]
        w_at_max = w_L[m_idx]

        print P_max, w_at_max
        p.plot(w_L, P_t)
    p.show()
