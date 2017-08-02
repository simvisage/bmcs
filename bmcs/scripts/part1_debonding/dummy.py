from bmcs.course import lecture04 as lec
import pylab

import numpy as np
po = lec.get_pullout_model_carbon_concrete(w_max=5.0)
p_array = np.array([5, 10, 15], dtype=np.float_)
for p in p_array:
    po.cross_section.P_b = p
    po.run()
    P = po.get_P_t()
    w0, wL = po.get_w_t()
    pylab.plot(wL, P, label='p=%d [mm]' % p)
pylab.legend(loc=2)
pylab.show()
