from math import sqrt, fabs

from ibvpy.mats.mats2D.mats2D_tensor import map2d_sig_eng_to_mtx
from numpy import where, zeros, dot, diag, linalg
from traits.api import HasTraits, Float


#from numpy.dual import *
class IStrainNorm2D(HasTraits):

    def get_f_trial(self, epsilon, D_el, E, nu, kappa):
        raise NotImplementedError

    def get_dede(self, epsilon, D_el, E, nu):
        raise NotImplementedError

# from the maple sheet damage_model_2D_euclidean_local_Code


class Euclidean(IStrainNorm2D):

    P_I = diag([1., 1., 0.5])

    def get_f_trial(self, epsilon, D_el, E, nu, kappa):
        '''
        Returns equivalent strain - kappa
        @param epsilon:
        @param D_el:
        @param E:
        @param kappa:
        '''
        return float(sqrt(dot(dot(epsilon, self.P_I), epsilon.T)) - kappa)

    def get_dede(self, epsilon, D_el, E, nu):
        dede = zeros(3)
        t1 = pow(epsilon[0], 2.)
        t3 = pow(epsilon[1], 2.)
        t5 = pow(epsilon[2], 2.)
        t8 = sqrt(4. * t1 + 4. * t3 + 2. * t5)
        t9 = 1. / t8
        dede[0] = 2. * t9 * epsilon[0]
        dede[1] = 2. * t9 * epsilon[1]
        dede[2] = t9 * epsilon[2]
        return dede


# from the maple sheet damage_model_2D_euclidean_local_Code
class Energy(IStrainNorm2D):

    def get_f_trial(self, epsilon, D_el, E, nu, kappa):
        # print "time %8.2f sec"%diff
        return sqrt(1. / E * dot(dot(epsilon, D_el), epsilon.T)) - kappa

    def get_dede(self, epsilon, D_el, E, nu):
        dede = zeros(3)
        t1 = 1. / E
        t2 = t1 * epsilon[0]
        t3 = t2 * D_el[0][0]
        t4 = t1 * epsilon[1]
        t5 = t4 * D_el[1][0]
        t8 = t2 * D_el[0][1]
        t9 = t4 * D_el[1][1]
        t12 = pow(epsilon[2], 2.)
        t16 = sqrt(epsilon[0] * (t3 + t5) + epsilon[1]
                   * (t8 + t9) + t12 * t1 * D_el[2][2])
        t17 = 1. / t16
        dede[0] = t17 * (2. * t3 + t5 + t4 * D_el[0][1]) / 2.
        dede[1] = t17 * (t2 * D_el[1][0] + t8 + 2. * t9) / 2.
        dede[2] = t17 * t1 * epsilon[2] * D_el[2][2]
        return dede


# from the maple sheet damage_model_2D_von_Mises_local_Code
class Mises(IStrainNorm2D):
    k = Float(10.,
              label="k",
              desc="Shape Parameter")

    def get_f_trial(self, epsilon, D_el, E, nu, kappa):
        t1 = self.k - 1.
        t2 = (epsilon[0] + epsilon[1])
        t4 = 1. / self.k
        t6 = 1. - 2. * nu
        t10 = t1 * t1
        t11 = t6 * t6
        t14 = (t2 * t2)
        t16 = pow(epsilon[0], 2.)
        t18 = pow(epsilon[1], 2.)
        t20 = pow(epsilon[2], 2.)
        t26 = pow((1 + nu), 2.)
        t31 = sqrt((t10 / t11 * t14) + 12. * self.k *
                   (t16 / 2. + t18 / 2. + t20 / 4. - t14 / 6.) / t26)
        t34 = (t1 * t2 * t4 / t6) / 2. + t4 * t31 / 2.
        return t34 - kappa

    def get_dede(self, epsilon, D_el, E, nu):
        dede = zeros(3)
        t1 = self.k - 1.
        t2 = 1. / self.k
        t5 = 1. - 2. * nu
        t8 = (t1 * t2 / t5) / 2.
        t9 = t1 * t1
        t10 = t5 * t5
        t12 = t9 / t10
        t13 = (epsilon[0] + epsilon[1])
        t14 = t13 * t13
        t16 = pow(epsilon[0], 2.)
        t18 = pow(epsilon[1], 2.)
        t20 = pow(epsilon[2], 2.)
        t26 = pow((1. + nu), 2.)
        t27 = 1. / t26
        t31 = sqrt((t12 * t14) + 12. * self.k *
                   (t16 / 2. + t18 / 2. + t20 / 4. - t14 / 6.) * t27)
        t32 = 1. / t31
        t33 = t2 * t32
        t35 = 2. * t12 * t13
        dede[0] = t8 + t33 * (t35 + 12. * self.k * (2. /
                                                    3. * epsilon[0] - epsilon[1] / 3.) * t27) / 4.
        dede[1] = t8 + t33 * (t35 + 12. * self.k * (2. /
                                                    3. * epsilon[1] - epsilon[0] / 3.) * t27) / 4.
        dede[2] = 3. / 2. * t32 * epsilon[2] * t27
        return dede


class Rankine(IStrainNorm2D):
    '''
    computes main stresses and makes a norm of their positive part
    '''

    def get_f_trial(self, epsilon, D_el, E, nu, kappa):
        sigma_I = linalg.eigh(map2d_sig_eng_to_mtx(
            dot(D_el, epsilon)))[0]  # main stresses
        # positive part and norm
        eps_eqv = linalg.norm(where(sigma_I >= 0., sigma_I, zeros(2))) / E
        return eps_eqv - kappa

    def get_dede(self, epsilon, D_el, E, nu):
        dede = zeros(3)
        t1 = D_el[0][0] / 2.
        t2 = D_el[1][0] / 2.
        t3 = pow(D_el[0][0], 2.)
        t4 = pow(epsilon[0], 2.)
        t6 = D_el[0][0] * epsilon[0]
        t7 = D_el[0][1] * epsilon[1]
        t13 = D_el[1][1] * epsilon[1]
        t16 = pow(D_el[0][1], 2.)
        t17 = pow(epsilon[1], 2.)
        t19 = D_el[1][0] * epsilon[0]
        t25 = pow(D_el[1][0], 2.)
        t29 = pow(D_el[1][1], 2.)
        t31 = pow(D_el[2][2], 2.)
        t32 = pow(epsilon[2], 2.)
        t35 = t3 * t4 + 2 * t6 * t7 - 2. * D_el[0][0] * t4 * D_el[1][0] - 2. * t6 * t13 + t16 * t17 - 2. * \
            t7 * t19 - 2. * D_el[0][1] * t17 * D_el[1][1] + t25 * \
            t4 + 2. * t19 * t13 + t29 * t17 + 4. * t31 * t32
        t36 = sqrt(t35)
        t37 = 1. / t36
        t57 = t37 * (2. * t3 * epsilon[0] + 2. * D_el[0][0] * D_el[0][1] * epsilon[1] - 4. * t6 * D_el[1][0] - 2. * D_el[0][0] *
                     D_el[1][1] * epsilon[1] - 2. * t7 * D_el[1][0] + 2. * t25 * epsilon[0] + 2. * D_el[1][0] * D_el[1][1] * epsilon[1]) / 4.
        t59 = fabs(t6 / 2. + t7 / 2. + t19 / 2. + t13 / 2. + t36 / 2.) / \
            (t6 / 2. + t7 / 2. + t19 / 2. + t13 / 2. + t36 / 2.)
        t63 = 1. / E
        t66 = D_el[0][1] / 2.
        t67 = D_el[1][1] / 2.
        t85 = t37 * (2. * t6 * D_el[0][1] - 2. * t6 * D_el[1][1] + 2. * t16 * epsilon[1] - 2. * D_el[0][1] *
                     D_el[1][0] * epsilon[0] - 4. * t7 * D_el[1][1] + 2. * t19 * D_el[1][1] + 2. * t29 * epsilon[1]) / 4.
        dede[0] = (t1 + t2 + t57 + t59 * (t1 + t2 + t57)) * t63 / 2.
        dede[1] = (t66 + t67 + t85 + t59 * (t66 + t67 + t85)) * t63 / 2.
        dede[2] = (t37 * t31 * epsilon[2] + t59 * t37 * t31 * epsilon[2]) * t63
        return dede


class Mazars(IStrainNorm2D):

    def get_f_trial(self, epsilon, D_el, E, nu, kappa):
        epsilon_pp = where(epsilon >= 0., epsilon, zeros(3))
        return sqrt(1. / E * dot(dot(epsilon_pp, D_el), epsilon_pp.T)) - kappa

    def get_dede(self, epsilon, D_el, E, nu):
        dede = zeros(3)
        t1 = epsilon[0] / 4.
        t2 = epsilon[1] / 4.
        t3 = pow(epsilon[0], 2.)
        t6 = pow(epsilon[1], 2.)
        t7 = pow(epsilon[2], 2.)
        t10 = sqrt(t3 - 2. * epsilon[0] * epsilon[1] + t6 + 4. * t7)
        t11 = t10 / 4.
        t12 = epsilon[0] + epsilon[1] + t10
        t13 = fabs(t12 / 2.)
        t15 = t1 + t2 + t11 + t13 / 2.
        t16 = 0.1e1 / E
        t17 = t16 * t15
        t19 = epsilon[0] + epsilon[1] - t10
        t20 = fabs(t19 / 2.)
        t22 = t1 + t2 - t11 + t20 / 2.
        t23 = t16 * t22
        t25 = t17 * D_el[0][0] + t23 * D_el[1][0]
        t29 = t17 * D_el[0][1] + t23 * D_el[1][1]
        t32 = sqrt(t15 * t25 + t22 * t29)
        t33 = 1. / t32
        t34 = 1. / t10
        t35 = epsilon[0] - epsilon[1]
        t36 = 2. * t34 * t35
        t37 = t36 / 0.8e1
        t38 = fabs(t12 / 2.) / (t12 / 2.)
        t39 = t36 / 4.
        t43 = 1. / 4. + t37 + t38 * (1. / 2. + t39) / 2.
        t45 = t16 * t43
        t47 = fabs(t19 / 2.) / (t19 / 2.)
        t51 = 1. / 4. - t37 + t47 * (1. / 2. - t39) / 2.
        t52 = t16 * t51
        t64 = -2. * t34 * t35
        t65 = t64 / 8.
        t66 = t64 / 4.
        t70 = 1. / 4. + t65 + t38 * (1. / 2. + t66) / 2.
        t72 = t16 * t70
        t77 = 1. / 4. - t65 + t47 * (1. / 2. - t66) / 2.
        t78 = t16 * t77
        t90 = t34 * epsilon[2]
        t93 = t90 + t38 * t34 * epsilon[2]
        t95 = t16 * t93
        t99 = -t90 - t47 * t34 * epsilon[2]
        t100 = t16 * t99
        dede[0] = t33 * (t43 * t25 + t15 * (t45 * D_el[0][0] + t52 * D_el[1][0]) +
                         t51 * t29 + t22 * (t45 * D_el[0][1] + t52 * D_el[1][1])) / 2.
        dede[1] = t33 * (t70 * t25 + t15 * (t72 * D_el[0][0] + t78 * D_el[1][0]) +
                         t77 * t29 + t22 * (t72 * D_el[0][1] + t78 * D_el[1][1])) / 2.
        dede[2] = t33 * (t93 * t25 + t15 * (t95 * D_el[0][0] + t100 * D_el[1][0]) +
                         t99 * t29 + t22 * (t95 * D_el[0][1] + t100 * D_el[1][1])) / 2.
        return dede
