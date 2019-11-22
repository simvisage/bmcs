'''
Created on Feb 9, 2010

@author: rostislav
'''

from traits.api import \
    Instance, Array, Float, Tuple, provides, \
    cached_property, Property, Range

import numpy as np
from util.traits.either_type import EitherType

from .boundary_conditions import FiniteEmbeddedLength, InfiniteEmbeddedLength, \
    ClampedFibre
from .energy_criterion import EnergyCriterion
from .parameters import Geometry, Material, Plot
from .resp_func import ResponseFunctionBase, IResponseFunction
from .stress_criterion import StressCriterion


@provides(IResponseFunction)
class RespFunc(ResponseFunctionBase):

    listener_string = 'boundary.+modified,'\
        'approach.+modified,'\
        'material.+modified,'\
        'plot.+modified,'\
        'geometry.+modified,'\
        'boundary.type.+modified'

    def __init__(self, **kw):
        super(RespFunc, self).__init__(**kw)
        self.material.add_listeners = self.add_listeners
        self.material.remove_listeners = self.remove_listeners
        self.material.get_value = self.get_value
        self._boundary_changed()

    boundary = EitherType(names=['infinite length', 'finite length',
                                 'clamped fibre end'],
                          klasses=[InfiniteEmbeddedLength, FiniteEmbeddedLength, ClampedFibre])

    def _boundary_changed(self):
        self.remove_listeners()
        self.geometry = self.boundary.geometry
        self.plot = self.boundary.plot
        self.get_value()
        self.add_listeners()

    def update_parameters(self):
        self.remove_listeners()
        self.approach.material = self.material
        self.boundary.type.geometry = self.geometry
        self.boundary.type.plot = self.plot
        self.approach.geometry = self.geometry
        self.approach.plot = self.plot
        self.add_listeners()

    approach = EitherType(names=['stress criterion', 'energy criterion'],
                          klasses=[StressCriterion, EnergyCriterion, ])

    geometry = Instance(Geometry)

    def _geometry_default(self):
        return Geometry()

    material = Instance(Material)

    def _material_default(self):
        return Material()

    plot = Instance(Plot)

    def _plot_default(self):
        return Plot()

    values = Tuple(Array, Array)

    def get_value(self):
        self.update_parameters()
        self.remove_listeners()
        if self.boundary.type.BC == 'double-sided pull-out with infinite embedded length':
            self.approach.clamp = False
            l = self.geometry.l
            self.geometry.l = 0.0
            u_plot = self.plot.u_plot
            self.plot.u_plot = self.plot.w_plot / 2.0
            self.approach.bool_infinite = True
            x, y = self.approach.get_value()
            self.approach.bool_infinite = False
            x *= 2.0
            self.values = (x, y)
            self.geometry.l = l
            self.plot.u_plot = u_plot
        elif self.boundary.type.BC == 'double-sided pull-out with finite embedded length':
            # todo: not complete
            self.approach.clamp = False
            l = self.geometry.l
            L = self.geometry.L
            self.geometry.l = 0.0
            u_plot = self.plot.u_plot
            self.geometry.L = self.boundary.type.Le
            self.approach.bool_finite = True
            x1, y1 = self.approach.get_value()
            self.geometry.L = self.boundary.geometry.Lf - self.boundary.type.Le
            x2, y2 = self.approach.get_value()
            self.values = self.approach.get_value()
            self.approach.bool_finite = False
            self.geometry.l = l
            self.geometry.L = L
        elif self.boundary.type.BC == 'double-sided pull-out with clamped fibre end':
            # todo: not complete
            self.approach.clamp = True
            self.approach.bool_clamp = True
            self.values = self.approach.get_value()
            self.approach.bool_clamp = False
        elif self.boundary.type.BC == 'one-sided pull-out with clamped fibre end':
            self.approach.clamp = True
            self.approach.bool_clamp = True
            self.approach.L = self.boundary.type.Le
            self.values = self.approach.get_value()
            self.approach.bool_clamp = False
        elif self.boundary.type.BC == 'one-sided pull-out with finite embedded length':
            self.approach.clamp = False
            self.approach.bool_finite = True
            self.values = self.approach.get_value()
            self.approach.bool_finite = False
        elif self.boundary.type.BC == 'one-sided pull-out with infinite embedded length':
            self.approach.clamp = False
            self.approach.bool_infinite = True
            self.values = self.approach.get_value()
            self.approach.bool_infinite = False
        else:
            # place holder
            self.values = self.approach.get_value()
        self.add_listeners()


class OneSidedInfinite(RespFunc):

    param_names = ['qf', 'qy', 'Ef', 'l', 'Af', 'k']
    title = 'one sided infinite'
    qf = Float(1220.0)
    qy = Float(1430.0)
    Ef = Float(210.e9)
    l = Float(0.001)
    k = Float(20e9)
    Af = Float(3.14e-8)

    w = Property(depends_on='k, Ef, Af')

    @cached_property
    def _get_w(self):
        return np.sqrt(self.k / self.Ef / self.Af)

    def __call__(self, u, qf, qy, Ef, l, Af, k):
        w = self.w
        deb = Heaviside(u - qy / (Ef * Af * w) * (1 / w + l)) * ((-l * qf * w +
                                                                  np.sqrt(abs(l ** 2 * qf ** 2 * w ** 2 +
                                                                              qy ** 2 + 2. * u * w ** 2 * Ef * Af * qf - 2. * qy * qf))) / w)
        lin = Heaviside(-u + qy / (Ef * Af * w) * (1 / w + l)
                        ) * (Ef * Af * w * u / (w * l + 1))
        return lin + deb


def Heaviside(x):
    return (np.sign(x) + 1.0) / 2.0

# class OneSidedFinite(RespFunc):
#
#     param_names = ['qf', 'qy', 'Ef', 'Lf', 'Af', 'phi', 'z', 'k' ]
#     title = 'one sided finite'
#     qf = Float(1220.0)
#     qy = Float(12100)
#     Ef = Float(210e9)
#     Lf = Float(0.006)
#     z = Float(0.0)
#     phi = Range(0, np.pi / 2.)
#     k = Float(20e9)
#     Af = Float(3.14e-8)
#     rf = Float(0.001)
#     l = Float(0.0)
#     beta = Float(0.0)
#     f = Float(0.05)
#
#     a = Array
#     L = Property(depends_on = 'Lf, phi, z')
#     def _get_L(self):
#         return (self.Lf / 2. - self.z / np.cos(self.phi)) * \
#                Heaviside(self.Lf / 2. - self.z / np.cos(self.phi))
#
#
#    w = Property(depends_on = 'k, Ef, Af')
#    def _get_w(self):
#        return np.sqrt(self.k / self.Ef / self.Af)
#
#    def u_a_residuum(self, a, u, qf, qy, Ef, Lf, Af, phi, z, k, l):
#
#        w = self.w
#        L = self.L
#        P_deb = qf * a + qy / w * tanh(w * (L - a))
#        u_eval = (P_deb - qf * a) / Ef / Af / w / tanh(w * (L - a)) + \
#            (P_deb - .5 * qf * a) / Ef / Af * a + P_deb * l / Af / Ef
#        return u - u_eval
#
#    def get_a(self, u , qf, qy, Ef, Lf, Af, phi, z, k, l):
#        return ridder(self.u_a_residuum, 1e-12, self.get_umax[1], args = (u, qf, qy, Ef, Lf, Af, phi, z, k, l))
#
#    def get_u(self, P, a, qf):
#        ''' takes a- and P-array and returns u-array '''
#        Ef = self.Ef
#        Af = self.Af
#        w = self.w
#        L = self.L
#        u = (P - qf * a) / Ef / Af / w / tanh(w * (L - a)) + \
#             (P - .5 * qf * a) / Ef / Af * a + P * self.l / Af / Ef
#        return u
#
#    def u_L0_residuum(self, u, qf, L0):
#
#        L = self.L
#        Ef = self.Ef
#        Af = self.Af
#        l = self.l
#        #k = self.k
#        #w = np.sqrt( k / Ef / Af )
#
#        #a = linspace( 0, L - L / 1e10, 5 )
#        #a = self.a
#        #P_deb = qf * a + self.qy / w * tanh( w * ( L - a ) )
#        #u_deb = self.get_u( P_deb, a, qf )
#        #idxmax = argmax( u_deb )
#        #u_max = u_deb[idxmax]
#
#        P = qf * L0 * (1 + self.beta * (L - L0) / (2 * self.rf))
#        delta_u = P * L0 / (2. * Ef * Af)
#        delta_free_l = (l + L - L0) * P / (Ef * Af)
#        delta_l = L - L0
#        u_eval = delta_u + delta_free_l + delta_l
#        return u - u_eval
#
#    def continuous_finite(self, u, qf, qy, Ef, Lf, Af, phi, z, k, l):
#        ''' returns the u and P array for finite embedded length '''
#
#        L = self.L
#        w = self.w
#        #a = linspace( 0, L - L / 1e10, 100 )
#
#        a_func = frompyfunc(self.get_a, 10, 1)
#        # return root a(u)
#        self.a = array(a_func(u, qf, qy, Ef, Lf, Af, phi, z, k, l), dtype = float)
#        a = self.a
#
#        # P-u diagram including snap back
#        P_deb_full = qf * a + qy / w * tanh(w * (L - a))
#        u_deb_full = self.get_u(P_deb_full, a, qf)
#        #print  u_deb_full[0]
#        #idxmax = argmax( u_deb_full )
#        # P-u diagram snap back cutted
#        u_deb = u_deb_full#[0:idxmax + 1]
#        P_deb = P_deb_full#[0:idxmax + 1]
#        print 'a', a
#        print 'c', u_deb, P_deb
#        return u_deb, P_deb
#
#        # pull-out stage
#        # L0 is the embedded length of a pure frictional pull-out that
#        # corresponds to the displacement at the end of the debonding stage
#        L0 = brentq(self.u_L0_residuum, 1e-12, 2 * L, args = (u, qf), xtol = 0.00001)
#        # if L0 is not in interval (0,L), the load drops to zero
#        if round(L, 7) >= round(L0, 7) >= 0:
#            #lp = linspace( L0, 0, 5 )
#            lp = L0
#            P_pull = qf * lp * (1 + self.beta * (L - lp) / (2 * self.rf))
#            # displacement corresponding to the actual embedded length
#            delta_u = P_pull * lp / (2. * Ef * Af)
#            # displacement corresponding to the actual free length
#            delta_free_l = (self.l + L - lp) * P_pull / (Ef * Af)
#            # displacement corresponding to the free length increment
#            delta_l = L - lp
#            u_pull = delta_u + delta_free_l + delta_l
#            print 'a'
#            return  u_deb, u_pull, P_deb, P_pull
#        else:
#            u_pull = u_deb[-1]
#            P_pull = 0
#            print 'b'
#            return u_deb, u_pull, P_deb, P_pull
#
#    def get_u_max(self, qf, qy, Ef, Lf, Af, phi, z, k, l):
#        ''' takes a- and returns u_max'''
#
#
#        ndims = sum([isinstance(param, ndarray) for param in [qf, qy, Ef, Lf, Af, phi, z, k, l]])
#        w = self.w
#        L = self.L
#
#        ax = ndims * 'newaxis,'
#        ax_string = 'a_linspace[:,' + ax[:-1] + ']'
#        a_linspace = linspace(0, L - L / 1e15, 20)
#        a = eval(ax_string)
#
#        P = qf * a + qy / w * tanh(w * (L - a))
#        u_arr = (P - qf * a) / Ef / Af / w / tanh(w * (L - a)) + \
#             (P - .5 * qf * a) / Ef / Af * a + P * l / Af / Ef
#        u_lin_deb = qy / (Ef * Af * w) * (1 / w + l)
#        u_deb_pull = max(u_arr)
#        Heaviside(u_arr - u_lin_deb) * Heaviside(u_deb_pull - u_arr)
#        idx = argmax(u_arr)
#        #print 'u', u_arr
#        print u_arr[0, ]
#
#        #fce_P = interp1d( hstack( ( u_arr[..., 0:idx] ) ), hstack( ( P[..., 0:idx] ) ) )
#        return u_max, a
#
#    def __call__(self, u, qf, qy, Ef, Lf, Af, phi, z, k, l):
#        w = self.w
#        u_lin_deb = qy / (Ef * Af * w) * (1 / w + l)
#        u_deb_pull, a_deb_pull, fce_P_interp = self.get_u_max(u, qf, qy, Ef, Lf, Af, phi, z, k, l)
#        #deb = deb * Heaviside( u - u_lin_deb ) * Heaviside( u_deb_pull - u )
#        fce_P = frompyfunc(fce_P_interp, 1, 1)
#        deb = fce_P(u) * Heaviside(u - u_lin_deb) * Heaviside(u_deb_pull - u)
#        return deb
#
#
# if __name__ == "__main__":
#     rf = OneSidedFinite()
#     y = rf.get_u_max(qf=array([1220, 1230]), qy=array([[1220], [1225]]), l=0.001,
#                      Ef=210e9, Lf=0.006, z=0.0, phi=0, k=20e9, Af=3.14e-8)
#    x = linspace( 0, 6.12e-6, 100 )
#    plt.plot( x, y )
#    plt.show()


#    x = array( [0, 1, 5.] )
#    y = array( [3, 5, 9.] )
#    func = interp1d(x,y)
#
#    xx = array( [0, 4, 8.] )
#    yy = array( [1, 7, 9.] )
#    func2 = interp1d(xx,yy)
#
#    f_arr = array([func, func2])
#
#
#    def f( u, a ):
#        return a( u )
#
#    fcefrom = frompyfunc( f, 2, 1 )
#
#
#    print fcefrom( 2, f_arr )


#    a = linspace( -.0, 0.003, 200 )
#    qf = 1220
#    qy = 12100.0
#    l = 0.0
#    Ef = 210e9
#    Lf = 0.006
#    z = 0.0
#    phi = 0
#    k = 20e9
#    Af = 3.14e-8
#    P = qf * a + qy / rf.w * tanh( rf.w * ( rf.L - a ) )
#    b = 2e-6 - rf.get_u( P, a, qf )
#    #b = b * Heaviside( -b )
#    rf.get_u( P, a, qf )
#    plt.plot( a, b )
#    plt.show()
