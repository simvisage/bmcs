#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Sep 29, 2011 by: rch


from etsproxy.traits.api import Float, Str, implements
from etsproxy.traits.ui.ui_traits import Image
from math import pi
from numpy import sqrt, minimum, maximum
from stats.spirrid import SPIRRID, RV, make_ogrid, Heaviside
from stats.spirrid.i_rf import IRF
from stats.spirrid.rf import RF
import etsproxy.mayavi.mlab as m
import numpy as np
import os.path
import string
import tempfile

if __name__ == '__main__':

    class CBClampedFiber(RF):
        '''
        Crack bridged by a short fiber with constant
        frictional interface to the matrix; clamped fiber end
        '''

        implements(IRF)

        title = Str('crack bridge - clamped fiber with constant friction')
        image = Image('pics/cb_short_fiber.jpg')

        xi = Float(0.0179, auto_set = False, enter_set = True, input = True,
                    distr = ['weibull_min', 'uniform'])

        tau = Float(2.5, auto_set = False, enter_set = True, input = True,
                    distr = ['uniform', 'norm'])

        l = Float(0.0, auto_set = False, enter_set = True, input = True,
                  distr = ['uniform'], desc = 'free length')

        D_f = Float(26e-3, auto_set = False, input = True,
                  enter_set = True, distr = ['uniform', 'weibull_min'])

        E_f = Float(72.0e3, auto_set = False, enter_set = True, input = True,
                      distr = ['uniform'])

        theta = Float(0.01, auto_set = False, enter_set = True, input = True,
                      distr = ['uniform', 'norm'], desc = 'slack')

        phi = Float(1., auto_set = False, enter_set = True, input = True,
                      distr = ['uniform', 'norm'], desc = 'bond quality')

        Ll = Float(1., auto_set = False, enter_set = True, input = True,
                  distr = ['uniform'], desc = 'embedded length - left')

        Lr = Float(.5, auto_set = False, enter_set = True, input = True,
                  distr = ['uniform'], desc = 'embedded length - right')

        w = Float(auto_set = False, enter_set = True, input = True,
                   desc = 'crack width',
                   ctrl_range = (0, 0.01, 100))

        x_label = Str('crack opening [mm]')
        y_label = Str('force [N]')

        C_code = Str('')

        # TODO: case where Lmin is zero - gives a double sided pullout
        # should be one sided though
        def __call__(self, w, tau, l, D_f, E_f, theta, xi, phi, Ll, Lr):

            A = pi * D_f ** 2 / 4.
            Lmin = minimum(Ll, Lr)
            Lmax = maximum(Ll, Lr)

            Lmin = maximum(Lmin - l / 2., 0)
            Lmax = maximum(Lmax - l / 2., 0)

            l = minimum(Lr + Ll, l)

            l = l * (1 + theta)
            w = w - theta * l

            T = tau * phi * D_f * pi

            # double sided debonding
            l0 = l / 2.
            q0 = (-l0 * T + sqrt((l0 * T) ** 2 + w * Heaviside(w) * E_f * A * T))

            # displacement at which the debonding to the closer clamp is finished
            # the closer distance is min(L1,L2)

            w0 = Lmin * T * (Lmin + 2 * l0) / E_f / A

            # debonding from one side; the other side is clamped
            # equal to L1*T + one sided pullout with embedded length Lmax - Lmin and free length 2*L1 + l

            # force at w0
            Q0 = Lmin * T
            l1 = 2 * Lmin + l
            q1 = (-(l1) * T + sqrt((l1 * T) ** 2 +
                2 * (w - w0) * Heaviside(w - w0) * E_f * A * T)) + Q0

            # displacement at debonding finished at both sides
            # equals a force of T*(larger clamp distance)


            # displacement, at which both sides are debonded
            w1 = w0 + (Lmax - Lmin) * T * ((Lmax - Lmin) + 2 * (l + 2 * Lmin)) / 2 / E_f / A
            # linear elastic response with stiffness EA/(clamp distance)
            q2 = E_f * A * (w - w1) / (Lmin + Lmax + l) + (Lmax) * T

            q0 = q0 * Heaviside(w0 - w)
            q1 = q1 * Heaviside(w - w0) * Heaviside(w1 - w)
            q2 = q2 * Heaviside(w - w1)

            q = q0 + q1 + q2

            # include breaking strain
            q = q * Heaviside(A * E_f * xi - q)
            #return q0, q1, q2 * Heaviside( A * E_f * xi - q2 ), w0 + theta * l, w1 + theta * l
            return q

    class CBClampedFiberSP(CBClampedFiber):
        '''
        stress profile for a crack bridged by a short fiber with constant
        frictional interface to the matrix; clamped fiber end
        '''

        x = Float(0.0, auto_set = False, enter_set = True, input = True,
                  distr = ['uniform'], desc = 'distance from crack')

        x_label = Str('position [mm]')
        y_label = Str('force [N]')

        C_code = Str('')

        def __call__(self, w, x, tau, l, D_f, E_f, theta, xi, phi, Ll, Lr):

            T = tau * phi * D_f * pi

            q = super(CBClampedFiberSP, self).__call__(w, tau, l, D_f, E_f, theta, xi, phi, Ll, Lr)
            q_x = q * Heaviside(l / 2. - abs(x)) + (q - T * (abs(x) - l / 2.)) * Heaviside(abs(x) - l / 2.)
            q_x = q_x * Heaviside(x + Ll) * Heaviside(Lr - x)
            q_x = q_x * Heaviside(q_x)

            return q_x

    q = CBClampedFiberSP()

    s = SPIRRID(q = q,
                sampling_type = 'LHS',
                evars = dict(w = np.linspace(0.0, 0.4, 50),
                             x = np.linspace(-20.1, 20.5, 100),
                             Lr = np.linspace(0.1, 20.0, 50)
                             ),
                tvars = dict(tau = RV('uniform', 0.7, 1.0),
                             l = RV('uniform', 5.0, 10.0),
                             D_f = 26e-3,
                             E_f = 72e3,
                             theta = 0.0,
                             xi = RV('weibull_min', scale = 0.017, shape = 8, n_int = 10),
                             phi = 1.0,
                             Ll = 50.0,
#                              Lr = 1.0
                             ),
                n_int = 5)

    e_arr = make_ogrid(s.evar_lst)
    n_e_arr = [ e / np.max(np.fabs(e)) for e in e_arr ]

    max_mu_q = np.max(np.fabs(s.mu_q_arr))
    n_mu_q_arr = s.mu_q_arr / max_mu_q
    n_std_q_arr = np.sqrt(s.var_q_arr) / max_mu_q

    #===========================================================================
    # Prepare plotting 
    #===========================================================================
    tdir = tempfile.mkdtemp()
    n_img = n_mu_q_arr.shape[0]
    fnames = [os.path.join(tdir, 'x%02d.jpg' % i) for i in range(n_img) ]

    f = m.figure(1, size = (1000, 500), fgcolor = (0, 0, 0),
                 bgcolor = (1., 1., 1.))

    s = m.surf(n_e_arr[1], n_e_arr[2], n_mu_q_arr[0, :, :])
    ms = s.mlab_source

    m.axes(s, color = (.7, .7, .7),
           extent = (-1, 1, 0, 1, 0, 1),
           ranges = (-0.21, 0.21, 0.1, 20, 0, max_mu_q),
           xlabel = 'x[mm]', ylabel = 'Lr[mm]',
           zlabel = 'f[N]',)
    m.view(-60.0, 70.0, focalpoint = [0., 0.45, 0.45])

    m.savefig(fnames[0])

    for i, fname in enumerate(fnames[1:]):
        ms.scalars = n_mu_q_arr[i, :, :]
        m.savefig(fname)

    images = string.join(fnames, ' ')
    destination = os.path.join('fig', 'fiber_cb_8p_anim.gif')

    import platform
    if platform.system() == 'Linux':
        os.system('convert ' + images + ' ' + destination)
    else:
        raise NotImplementedError('film production available only on linux')

    m.show()
