
from traits.api import \
    Float, HasStrictTraits, Instance, Int

import mayavi.mlab as m
import numpy as np


ONE = np.ones((1,), dtype=np.float_)
DELTA = np.identity(3)

# Levi Civita symbol
EPS = np.zeros((3, 3, 3), dtype='f')
EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1


class YieldConditionJ2(HasStrictTraits):
    '''
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
    '''
    '''
    n_D = Int(3)
    f_t = Float(3.0)
    f_c = Float(30.0)

    def f(self, sig_ij, z=0):
        '''Given a tensor sig_ij return the yield condition
        '''
        I1 = np.einsum('...ii,...ii', sig_ij, DELTA)
        s_ij = sig_ij - np.einsum('...,ij->...ij', I1 / 3.0, DELTA)
        J2 = np.einsum('...ii,...ii', s_ij, s_ij) / 2.0

        alpha_F = (self.f_c - self.f_t) / 3.0
        tau2_y = self.f_c * self.f_t / 3.0

        return J2 + alpha_F * I1 - tau2_y

    def get_df_dsig(self):
        pass


def get_lut():
    opacity = 20.0
    lut = np.zeros((256, 4), dtype=Int)
    alpha = 255 * opacity / 100.0
    lut[:] = np.array([0, 0, 255, int(round(alpha))], dtype=Int)


if __name__ == '__main__':
    yc = YieldConditionJ2(sig_y=6.0)
    yc = YieldConditionDruckerPrager(f_t=3.0, f_c=30.0)
    sig = np.array([[2, 3, 4],
                    [1, 3, 2],
                    [3, 4, 5]], dtype=np.float_)
    min_sig = -50.0
    max_sig = 10.0
    n_sig = 30j
    sig_1, sig_2, sig_3 = np.mgrid[min_sig: max_sig: n_sig,
                                   min_sig: max_sig: n_sig,
                                   min_sig: max_sig: n_sig]

    sig_abcj = np.einsum('jabc->abcj', np.array([sig_1, sig_2, sig_3]))
    sig_abcij = np.einsum('abcj,jl->abcjl', sig_abcj, DELTA)
    f = yc.f(sig_abcij)

    f_pipe = m.contour3d(sig_1, sig_2, sig_3, f, contours=[0.0])
    f_pipe.module_manager.scalar_lut_manager.lut.table = get_lut()
    m.show()
