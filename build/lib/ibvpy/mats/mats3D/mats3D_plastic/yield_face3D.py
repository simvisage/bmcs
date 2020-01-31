
from traits.api import \
    Float, HasStrictTraits, Instance, Int
from traitsui.api import View, Item, HSplit, Group

import mayavi.mlab as m
import numpy as np


ONE = np.ones((1,), dtype=np.float_)
DELTA = np.identity(3)

# Levi Civita symbol
EPS = np.zeros((3, 3, 3), dtype='f')
EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1


class YieldConditionJ2(HasStrictTraits):
    '''J2-plasticity
    '''
    n_D = Int(3)
    sig_y = Float(2.0)
    H = Float(10.0)

    def f(self, sig_ij, z=0):
        '''Given a tensor sig_ij return the yield condition
        '''
        I1 = np.einsum('...ii,...ii', sig_ij, DELTA)
        s_ij = sig_ij - np.einsum('...,ij->...ij', I1 / 3.0, DELTA)
        J2 = np.einsum('...ii,...ii', s_ij, s_ij) / 2.0

        return J2 - self.sig_y - self.H * z

    def get_df_dsig(self):
        pass


class YieldConditionDruckerPrager(HasStrictTraits):
    '''Drucker-Prager yield condition
    '''
    n_D = Int(3)
    alpha_F = Float(0.24)
    tau2_y = Float(2.6)
#     f_t = Float(3.0)
#     f_c = Float(30.0)

    def f(self, sig_ij):
        '''Given a tensor sig_ij return the yield condition
        '''
        I1 = np.einsum('...ii,...ii', sig_ij, DELTA)
        s_ij = sig_ij - np.einsum('...,ij->...ij', I1 / 3.0, DELTA)
        J2 = np.einsum('...ij,...ij', s_ij, s_ij) / 2.0

        #alpha_F = (self.f_c - self.f_t) / 3.0
        alpha_F = self.alpha_F
        tau2_y = self.tau2_y
        #tau2_y = self.f_c * self.f_t / 3.0

        return np.sqrt(J2) + alpha_F * I1 - tau2_y

    def get_df_dsig(self):
        pass

    view = View(Item('alpha_F', full_size=True, resizable=True),
                Item('tau2_y'))


class YieldConditionVonMises(HasStrictTraits):

    '''
    '''
    k = Float(3)

    def f(self, sig_ij):
        I1 = np.einsum('...ii,...ii', sig_ij, DELTA)
        s_ij = sig_ij - np.einsum('...,ij->...ij', I1 / 3.0, DELTA)
        J2 = np.einsum('...ij,...ij', s_ij, s_ij) / 2.0

        return J2 - self.k ** 2

    view = View(Item('k', full_size=True, resizable=True),
                )


class YieldConditionWillamWarnke(HasStrictTraits):

    '''the three parameter Willam-Warnke yield function,
    https://en.wikipedia.org/wiki/Willam-Warnke_yield_criterion
    '''
    sig_c = Float(10.)  # the uniaxial compressive strength
    sig_t = Float(3.)  # the uniaxial tensile strength
    sig_b = Float(17.)  # the equibiaxial compressive strength

    def f(self, sig_ij):
        I1 = np.einsum('...ii,...ii', sig_ij, DELTA)
        s_ij = sig_ij - np.einsum('...,ij->...ij', I1 / 3.0, DELTA)
        J2 = np.einsum('...ij,...ij', s_ij, s_ij) / 2.0
        J3 = np.einsum('...ij,...jk,...ki', s_ij, s_ij, s_ij) / 3.0
        # the Lode angle

        theta = np.arccos(1.5 * np.sqrt(3.0) * J3 / J2 ** (1.5)) / 3.0

        rc = np.sqrt(1.2) * self.sig_b * self.sig_t / \
            (3 * self.sig_b * self.sig_t +
             self.sig_c * (self.sig_b - self.sig_t))
        rt = np.sqrt(1.2) * self.sig_b * self.sig_t / \
            (self.sig_c * (2 * self.sig_b + self.sig_t))

        u = 2 * rc * (rc ** 2 - rt ** 2) * np.cos(theta)

        a = 4 * (rc ** 2 - rt ** 2) * np.cos(theta) ** 2 + \
            5 * rt ** 2 - 4 * rt * rc
        v = rc * (2 * rt - rc) * np.sqrt(a)

        w = 4 * (rc ** 2 - rt ** 2) * np.cos(theta) ** 2 + (rc - 2 * rt) ** 2

        r = (u + v) / w
        z = self.sig_b * self.sig_t / self.sig_c / (self.sig_b - self.sig_t)

        return (1. / (3. * z) * I1 / self.sig_c + np.sqrt(0.4) /
                r * np.sqrt(J2) / self.sig_c - 1.)

    view = View(Item('sig_c', full_size=True, resizable=True),
                Item('sig_t'),
                Item('sig_b'))


class YieldConditionRankine(HasStrictTraits):

    '''single-parameter Rankine failure surface
    '''
    sig_t = Float(3.)  # the uniaxial tensile strength

    def f(self, sig_ij):
        sig_evals, sig_evects = np.linalg.eigh(sig_ij)
        max_sig = np.max(sig_evals, axis=-1)
        f = max_sig - self.sig_t
        return f

    view = View(Item('sig_t', full_size=True, resizable=True),
                )


class YieldConditionAbaqus(HasStrictTraits):

    '''Lee et al (1988) - PLASTIC-DAMAGE MODEL FOR CYCLIC
     LOADING OF CONCRETE STRUCTURES'''

    sig_c = Float(-10.)  # the uniaxial compressive strength
    sig_t = Float(3.)  # the uniaxial tensile strength
    sig_b = Float(-17.)  # the equibiaxial compressive strength

    def f(self, sig_ij):
        I1 = np.einsum('...ii,...ii', sig_ij, DELTA)
        s_ij = sig_ij - np.einsum('...,ij->...ij', I1 / 3.0, DELTA)
        J2 = np.einsum('...ij,...ij', s_ij, s_ij) / 2.0

        alpha = (self.sig_b - self.sig_c) / (2. * self.sig_b - self.sig_c)
        beta = self.sig_c / self.sig_t * (alpha - 1.) - (1. + alpha)

        sig_i = np.einsum('...ii->...i', sig_ij)
        sig_max = np.amax(sig_i, axis=-1)

        F = 1. / (1. - alpha) * \
            (alpha * I1 + np.sqrt(3. * J2) + beta * sig_max)
        c = -self.sig_c  # cohesion

        return F - c

    view = View(Item('sig_c', full_size=True, resizable=True),
                Item('sig_t'),
                Item('sig_b'))


class YieldConditionExtendedLeonModel(HasStrictTraits):

    '''
    '''

    sig_c = Float(-10.)  # the uniaxial compressive strength
    sig_t = Float(3.)  # the uniaxial tensile strength
    sig_b = Float(-17.)  # the equibiaxial compressive strength

    def f(self, sig_ij):
        I1 = np.einsum('...ii,...ii', sig_ij, DELTA)
        s_ij = sig_ij - np.einsum('...,ij->...ij', I1 / 3.0, DELTA)
        J2 = np.einsum('...ij,...ij', s_ij, s_ij) / 2.0
        J3 = np.einsum('...ij,...jk,...ki', s_ij, s_ij, s_ij) / 3.0
        z = I1 / np.sqrt(3.0)
        r = np.sqrt(2.0 * J2)
        sin3theta = J3 / 2.0 * np.power(3.0 / J2, 1.5)
        theta = np.arcsin(sin3theta) / 3.0

        # -- here to go on

        return z, r, theta

    view = View(Item('sig_c', full_size=True, resizable=True),
                Item('sig_t'),
                Item('sig_b'))


def get_lut():
    opacity = 20.0
    lut = np.zeros((256, 4), dtype=Int)
    alpha = 255 * opacity / 100.0
    lut[:] = np.array([0, 0, 255, int(round(alpha))], dtype=Int)


def show_yield_faces():
    #     yc = YieldConditionJ2(sig_y=6.0)
    sig = np.array([[2, 3, 4],
                    [1, 3, 2],
                    [3, 4, 5]], dtype=np.float_)

    yc = YieldConditionExtendedLeonModel()
    f = yc.f(sig)
    print('f', f)

    # return

    min_sig = -20.0
    max_sig = 5.0
    n_sig = 100j
    sig_1, sig_2, sig_3 = np.mgrid[min_sig: max_sig: n_sig,
                                   min_sig: max_sig: n_sig,
                                   min_sig: max_sig: n_sig]

    sig_abcj = np.einsum('jabc->abcj', np.array([sig_1, sig_2, sig_3]))
    sig_abcij = np.einsum('abcj,jl->abcjl', sig_abcj, DELTA)

    yc = YieldConditionDruckerPrager(alpha_F=0.24, tau2_y=2.6)
    f = yc.f(sig_abcij)
    f_pipe = m.contour3d(
        sig_1, sig_2, sig_3, f, contours=[0.0], color=(1, 0, 0))
    f_pipe.module_manager.scalar_lut_manager.lut.table = get_lut()

#     yc = YieldConditionVonMises(k=10.)
    yc = YieldConditionWillamWarnke()
    f = yc.f(sig_abcij)
    f_pipe = m.contour3d(
        sig_1, sig_2, sig_3, f, contours=[0.0], color=(0, 1, 0))
    f_pipe.module_manager.scalar_lut_manager.lut.table = get_lut()

    yc = YieldConditionAbaqus()
    f = yc.f(sig_abcij)
    f_pipe = m.contour3d(
        sig_1, sig_2, sig_3, f, contours=[0.0], color=(0, 0, 1))
    f_pipe.module_manager.scalar_lut_manager.lut.table = get_lut()

    m.axes(f_pipe)
    m.show()


if __name__ == '__main__':
    show_yield_faces()
