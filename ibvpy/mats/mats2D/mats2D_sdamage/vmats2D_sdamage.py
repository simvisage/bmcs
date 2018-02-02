
from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
from ibvpy.mats.mats_eval import IMATSEval
from traitsui.api import \
    Item, View, VSplit, Group, Spring
from util.traits.either_type import EitherType
import numpy as np
import traits.api as tr
from vstrain_norm2d import Energy, Euclidean, Mises, Rankine, Mazars


class MATS2D(tr.HasStrictTraits):
    # -----------------------------------------------------------------------------------------------------
    # Construct the fourth order elasticity tensor
    # for the plane stress case (shape: (2,2,2,2))
    # -----------------------------------------------------------------------------------------------------
    stress_state = tr.Enum("plane_stress", "plane_strain", input=True)

    #-------------------------------------------------------------------------
    # Material parameters
    #-------------------------------------------------------------------------

    E = tr.Float(34e+3,
                 label="E",
                 desc="Young's Modulus",
                 auto_set=False,
                 input=True)

    nu = tr.Float(0.2,
                  label='nu',
                  desc="Poison ratio",
                  auto_set=False,
                  input=True)

    # first Lame paramter

    def _get_lame_params(self):
        la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E / (2 + 2 * self.nu)
        return la, mu

    D_ab = tr.Property(tr.Array, depends_on='+input')
    '''Elasticity matrix (shape: (3,3))
    '''
    @tr.cached_property
    def _get_D_ab(self):
        if self.stress_state == 'plane_stress':
            return self._get_D_ab_plane_stress()
        elif self.stress_state == 'plane_strain':
            return self._get_D_ab_plane_strain()

    def _get_D_ab_plane_stress(self):
        '''
        Elastic Matrix - Plane Stress
        '''
        E = self.E
        nu = self.nu
        D_stress = np.zeros([3, 3])
        D_stress[0, 0] = E / (1.0 - nu * nu)
        D_stress[0, 1] = E / (1.0 - nu * nu) * nu
        D_stress[1, 0] = E / (1.0 - nu * nu) * nu
        D_stress[1, 1] = E / (1.0 - nu * nu)
        D_stress[2, 2] = E / (1.0 - nu * nu) * (1.0 / 2.0 - nu / 2.0)
        return D_stress

    def _get_D_ab_plane_strain(self):
        '''
        Elastic Matrix - Plane Strain
        '''
        E = self.E
        nu = self.nu
        D_strain = np.zeros([3, 3])
        D_strain[0, 0] = E * (1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)
        D_strain[0, 1] = E / (1.0 + nu) / (1.0 - 2.0 * nu) * nu
        D_strain[1, 0] = E / (1.0 + nu) / (1.0 - 2.0 * nu) * nu
        D_strain[1, 1] = E * (1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)
        D_strain[2, 2] = E * (1.0 - nu) / (1.0 + nu) / (2.0 - 2.0 * nu)
        return D_strain

    map2d_ijkl2a = tr.Array(np.int_, value=[[[[0, 0],
                                              [0, 0]],
                                             [[2, 2],
                                              [2, 2]]],
                                            [[[2, 2],
                                              [2, 2]],
                                             [[1, 1],
                                                [1, 1]]]])
    map2d_ijkl2b = tr.Array(np.int_, value=[[[[0, 2],
                                              [2, 1]],
                                             [[0, 2],
                                              [2, 1]]],
                                            [[[0, 2],
                                              [2, 1]],
                                             [[0, 2],
                                                [2, 1]]]])

    D_abef = tr.Property(tr.Array, depends_on='+input')

    @tr.cached_property
    def _get_D_abef(self):
        return self.D_ab[self.map2d_ijkl2a, self.map2d_ijkl2b]


# from scipy.linalg import eig, inv
#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------
class MATS2DScalarDamage(MATS2DEval, MATS2D):
    '''
    Isotropic damage model.
    '''

    tr.implements(IMATSEval)

    #-------------------------------------------------------------------------
    # Parameters of the numerical algorithm (integration)
    #-------------------------------------------------------------------------

    stiffness = tr.Enum("secant", "algorithmic",
                        input=True)

    epsilon_0 = tr.Float(5e-2,
                         label="eps_0",
                         desc="Strain at the onset of damage",
                         auto_set=False,
                         input=True)

    epsilon_f = tr.Float(191e-1,
                         label="eps_f",
                         desc="Slope of the damage function",
                         auto_set=False,
                         input=True)

    strain_norm = EitherType(klasses=[Rankine,
                                      Mazars,
                                      Euclidean,
                                      Energy,
                                      Mises,
                                      ], input=True)

    # This event can be used by the clients to trigger an action upon
    # the completed reconfiguration of the material model
    #
    changed = tr.Event

    #--------------------------------------------------------------------------
    # View specification
    #--------------------------------------------------------------------------

    view_traits = View(VSplit(Group(Item('E'),
                                    Item('nu'),
                                    Item('epsilon_0'),
                                    Item('epsilon_f'),
                                    Item('strain_norm')),
                              Group(Item('stress_state', style='custom'),
                                    Item('stiffness', style='custom'),
                                    Spring(resizable=True),
                                    label='Configuration parameters',
                                    show_border=True,
                                    ),
                              ),
                       resizable=True
                       )

    #--------------------------------------------------------------------------
    # Private initialization methods
    #--------------------------------------------------------------------------
    def get_state_array_size(self):
        '''
        Return number of number to be stored in state array
        @param sctx:spatial context
        '''
        return 2

    def init(self, s_Emg):
        '''
        Intialize state variables.
        '''
        s_Emg[:, :, :] = 0

    #--------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #--------------------------------------------------------------------------

    def get_corr_pred(self, eps_Emab_n1, deps_Emab, tn, tn1,
                      update_state, s_Emg):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        if update_state:

            eps_Emab_n = eps_Emab_n1 - deps_Emab

            kappa_Em, omega_Em, f_idx = self._get_state_variables(
                s_Emg, eps_Emab_n)

            s_Emg[:, :, 0] = kappa_Em
            s_Emg[:, :, 1] = omega_Em

        kappa_Em, omega_Em, f_idx = self._get_state_variables(
            s_Emg, eps_Emab_n1)

        phi_Em = (1.0 - omega_Em)

        D_Emabef = np.einsum('Em,abef->Emabef',
                             phi_Em, self.D_abef)

        # print 'D_Emabef', D_Emabef
        sigma_Emab = np.einsum('Em,Emabef,Emef->Emab',
                               phi_Em, D_Emabef, eps_Emab_n1)

        if self.stiffness == "algorithmic":
            D_red = self._get_alg_stiffness(
                kappa_Em[f_idx], eps_Emab_n1[f_idx])
            # print 'D_red',D_red
            D_Emabef[f_idx] -= D_red
            # print 'D_Emabef',D_Emabef[0, 0]

        return D_Emabef, sigma_Emab

    #--------------------------------------------------------------------------
    # Subsidiary methods realizing configurable features
    #--------------------------------------------------------------------------
    def _get_state_variables(self, s_Emg, eps_Emab):
        kappa_Em = np.copy(s_Emg[:, :, 0])
        omega_Em = np.copy(s_Emg[:, :, 1])
        eps_eq_Em = self.strain_norm.get_eps_eq(eps_Emab,
                                                kappa_Em)
        f_trial_Em = eps_eq_Em - self.epsilon_0
        f_idx = np.where(f_trial_Em > 0)
        kappa_Em[f_idx] = eps_eq_Em[f_idx]
        omega_Em[f_idx] = self._get_omega(eps_eq_Em[f_idx])
        return kappa_Em, omega_Em, f_idx

    def _get_omega(self, kappa_Em):
        '''
        Return new value of damage parameter
        @param kappa:
        '''

        omega_Em = np.zeros_like(kappa_Em)

        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f
        kappa_idx = np.where(kappa_Em >= epsilon_0)
        omega_Em[kappa_idx] = (1.0 - (epsilon_0 / kappa_Em[kappa_idx] *
                                      np.exp(-1.0 * (kappa_Em[kappa_idx] - epsilon_0) /
                                             (epsilon_f - epsilon_0))
                                      ))
        return omega_Em

    def _get_domega(self, kappa_Em):
        '''
        Return new value of damage parameter
        @param kappa:
        '''
        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f

        domega_Em = np.zeros_like(kappa_Em)
        kappa_idx = np.where(kappa_Em >= epsilon_0)

#         factor = epsilon_0 / (epsilon_f - epsilon_0)
#
#         domega_Em[kappa_idx] = (
#              factor * np.exp(-(kappa_Em[kappa_idx] - epsilon_0) /
#                              epsilon_f - epsilon_0)
#         )

        factor_1 = epsilon_0 / (kappa_Em[kappa_idx] * kappa_Em[kappa_idx])
        factor_2 = epsilon_0 / (kappa_Em[kappa_idx] * (epsilon_f - epsilon_0))
        domega_Em[kappa_idx] = (
            (factor_1 + factor_2) * np.exp(-(kappa_Em[kappa_idx] - epsilon_0) /
                                           (epsilon_f - epsilon_0))
        )

        f1 = 1 / kappa_Em[kappa_idx]
        f2 = 1 / kappa_Em[kappa_idx] + 1 / ((epsilon_f - epsilon_0))
        f3 = epsilon_0 * np.exp((epsilon_0 - kappa_Em[kappa_idx]) /
                                (epsilon_f - epsilon_0))
        domega2_Em = f1 * f2 * f3
        if not np.allclose(domega_Em[kappa_idx], domega2_Em, 1e-5, 1e-5):
            print domega_Em[kappa_idx]
            print domega2_Em
            raise ValueError
        # print 'DOMEGA', domega_Em
        return domega_Em

    def _get_alg_stiffness(self, kappa_Em, eps_Emab_n1):
        '''
        '''
        domega_Em = self._get_domega(kappa_Em)
        deps_eq_Emcd = self.strain_norm.get_deps_eq(eps_Emab_n1)
        # print 'DEPS', deps_eq_Emcd[0]
        return np.einsum('...,...cd,abcd,...cd->...abcd',
                         domega_Em, deps_eq_Emcd, self.D_abef, eps_Emab_n1)


if __name__ == '__main__':

    #-------------------------------------------------------------------------
    # Example
    #-------------------------------------------------------------------------

    mats = MATS2DScalarDamage(E=30000,
                              stiffness='algorithmic',
                              nu=0.0,
                              epsilon_0=0.3,
                              epsilon_f=0.5)

    eps_Emab = np.array([[[[0.8, 0],
                           [0, 0.0]]
                          ]], dtype=np.float_)
    s_Emg = np.array([[[0, 0],
                       ]], dtype=np.float_)

    # print mats.strain_norm.get_eps_eq(eps_Emab, s_Emg[:, :, 0])
    # print mats._get_state_variables(s_Emg, eps_Emab)
    D_Emabcd, sig_Emab = mats.get_corr_pred(
        eps_Emab, eps_Emab, 0, 0, False, s_Emg)
