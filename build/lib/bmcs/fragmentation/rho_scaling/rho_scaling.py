'''
Created on Jul 31, 2017

@author: rch

The task solved within this package addresses the question of how
does the response of a tensile test with one fragmenting component,
(multiple cracking/rupture). Assuming there is a function with
a constant derivative in the final stage

'''

from bmcs.fragmentation.scm import SCM
from traits.api import Property

import numpy as np
import pylab as p


if __name__ == '__main__':
    from view.window import BMCSWindow

    p.figure(figsize=(12, 6))
    axc = p.subplot(121)
    axt = p.subplot(122)
    sig_mu = 4.0
    sig_fu = 1000.0
    rho1 = 0.01
    scm = SCM(sigma_mu=sig_mu, sigma_fu=sig_fu, rho=rho1)
    eps1, sig1 = scm.sig_eps
    axc.plot(eps1, sig1, label='rho = %5.2f' % rho1)
    axt.plot(eps1, sig1 / rho1, label='rho = %5.2f' % rho1)

    rho2 = 2 * rho1
    scm = SCM(sigma_mu=sig_mu, sigma_fu=sig_fu, rho=rho2)
    eps2, sig2 = scm.sig_eps
    axc.plot(eps2, sig2, label='rho = %5.2f' % rho2)
    axt.plot(eps2, sig2 / rho2, label='rho = %5.2f' % rho2)

    axt.set_xlabel('composite strain [-]')
    axc.set_xlabel('composite strain [-]')
    axt.set_ylabel('reinforcement stress [MPa]')
    axc.set_ylabel('composite stress [MPa]')
    axc.legend(loc=2)
    axt.legend(loc=2)
    p.show()
