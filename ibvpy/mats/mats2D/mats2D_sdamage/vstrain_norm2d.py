
from traits.api import HasTraits
import numpy as np


class IStrainNorm2D(HasTraits):

    def get_eps_eq(self, eps_Emef, kappa_Em):
        raise NotImplementedError

    def get_deps_eq(self, eps_Emef):
        raise NotImplementedError


class Rankine(IStrainNorm2D):
    '''
    Computes principal strains and makes a norm of their positive part
    '''

    def get_eps_eq(self, eps_Emef, kappa_Em):

        eps_11 = eps_Emef[..., 0, 0]
        eps_22 = eps_Emef[..., 1, 1]
        eps_12 = eps_Emef[..., 0, 1]
        eps_eq_Em = (
            0.5 * (eps_11 + eps_22) +
            np.sqrt(((eps_11 - eps_22) / 2.0)**2.0 + eps_12**2.0)
        )
        e_Em = np.concatenate(
            (eps_eq_Em[:, :, None], kappa_Em[:, :, None]), axis=2
        )
        eps_eq = np.max(e_Em, axis=2)
        return eps_eq

    def get_deps_eq(self, eps_Emef):
        eps11 = eps_Emef[..., 0, 0]
        eps22 = eps_Emef[..., 1, 1]
        eps12 = eps_Emef[..., 0, 1]
        eps_11_22 = eps11 - eps22
        factor = 1. / (2. * np.sqrt(eps_11_22 * eps_11_22 +
                                    4.0 * eps12 * eps12))
        df_trial1 = factor * np.array([[eps11 - eps22, 4.0 * eps12],
                                       [4.0 * eps12, eps22 - eps11]])
        return (np.einsum('ab...->...ab', df_trial1) +
                0.5 * np.identity(2)[None, :, :])


if __name__ == '__main__':
    import sympy as sp

    eps_0, eps_f, kappa = sp.symbols(
        '\\varepsilon_0, \\varepsilon_\\mathrm{f}, \\kappa')
    f = 1 - (eps_0 / kappa) * sp.exp(- (kappa - eps_0) / (eps_f - eps_0))
    print f
    print sp.diff(f, kappa)

    s_11, s_22, s_12 = sp.symbols(
        '\\varepsilon_{11},\\varepsilon_{22},\\varepsilon_{12}')
    sig = sp.Matrix([[s_11, s_12],
                     [s_12, s_22]])
    sig_12 = sig.eigenvals()
    for sig_principal, item in sig_12.items():
        print sig_principal
        sig_diff_11 = sp.diff(sig_principal, s_11)
        sig_diff_12 = sp.diff(sig_principal, s_12)
        sig_diff_22 = sp.diff(sig_principal, s_22)
        print '11', sig_diff_11
        print '12', sig_diff_12
        print '22', sig_diff_22
