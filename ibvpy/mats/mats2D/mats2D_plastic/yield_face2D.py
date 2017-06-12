from numpy import zeros, dot, diag, array
from traits.api import HasTraits, Float, Array
from traitsui.api import \
    Item, View, HSplit, VSplit, VGroup, Group, Spring

import numpy as np


class IYieldFace2D(HasTraits):

    def get_f_trial(self, xi_trial, q_1):
        raise NotImplementedError


class YieldFaceCPA(IYieldFace2D):
    '''
    Cutting Plane Algorithm
    '''

    def get_diff1s(self, epsilon_n1, E, nu, sctx):
        raise NotImplementedError

    def get_diff1q(self, epsilon_n1, E, nu, sctx):
        raise NotImplementedError

    def get_diff2ss(self, epsilon_n1, E, nu, sctx):
        raise NotImplementedError


class YieldFaceCPP(YieldFaceCPA):
    '''
    Closest Point Projection
    '''

    def get_diff2sq(self, epsilon_n1, E, nu, sctx):
        raise NotImplementedError

    def get_diff2qq(self, epsilon_n1, E, nu, sctx):
        raise NotImplementedError


class J2(YieldFaceCPP):
    '''
    Current version by Jakub (16.09.2008)
    '''

    sigma_y = Float(100.)
    cronecker_delta = array([1., 1., 0.])
    P = diag([1., 1., 2.])

    def get_f_trial(self, xi_trial, q_1):
        xi_v = (xi_trial[0] + xi_trial[1]) / 3.
        s_xi = xi_trial - dot(xi_v, self.cronecker_delta)
        J_2 = 0.5 * dot(dot(s_xi, self.P), s_xi)
        f_trial = J_2 - 1. / 3 * (self.sigma_y + q_1)**2
        # print "YF f_trial",f_trial
        return f_trial

    def get_diff1s(self, epsilon_n1, E, nu, sctx):
        diff1s = zeros(3)
        epsilon_p = sctx.mats_state_array[:3]
        q_1 = sctx.mats_state_array[3]
        q_2 = sctx.mats_state_array[4:]
        t1 = 1.0 - nu
        t2 = E * t1
        t4 = 1 / (1.0 + nu)
        t7 = 1 / (1.0 - 2.0 * nu)
        t8 = t4 * t7
        t9 = epsilon_n1[0] - epsilon_p[0]
        t11 = t2 * t8 * t9
        t13 = E * t4
        t14 = t7 * nu
        t15 = epsilon_n1[1] - epsilon_p[1]
        t17 = t13 * t14 * t15
        t21 = t13 * t14 * t9
        t24 = t2 * t8 * t15
        diff1s[0] = 5.0 / 9.0 * t11 + 5.0 / 9.0 * t17 - 5.0 / 9.0 * \
            q_2[0] - 4.0 / 9.0 * t21 - 4.0 / 9.0 * t24 + 4.0 / 9.0 * q_2[1]
        diff1s[1] = -4.0 / 9.0 * t11 - 4.0 / 9.0 * t17 + 4.0 / 9.0 * \
            q_2[0] + 5.0 / 9.0 * t21 + 5.0 / 9.0 * t24 - 5.0 / 9.0 * q_2[1]
        diff1s[2] = t2 * t4 / t1 * \
            (epsilon_n1[2] - epsilon_p[2]) - 2.0 * q_2[2]
        return diff1s

    def get_diff1q(self, epsilon_n1, E, nu, sctx):
        diff1q = zeros(4)
        epsilon_p = sctx.mats_state_array[:3]
        q_1 = sctx.mats_state_array[3]
        q_2 = sctx.mats_state_array[4:]
        t2 = 1.0 - nu
        t3 = E * t2
        t5 = 1 / (1.0 + nu)
        t8 = 1 / (1.0 - 2.0 * nu)
        t9 = t5 * t8
        t10 = epsilon_n1[0] - epsilon_p[0]
        t12 = t3 * t9 * t10
        t14 = E * t5
        t15 = t8 * nu
        t16 = epsilon_n1[1] - epsilon_p[1]
        t18 = t14 * t15 * t16
        t22 = t14 * t15 * t10
        t25 = t3 * t9 * t16
        diff1q[0] = -2.0 / 3.0 * self.sigma_y - 2.0 / 3.0 * q_1
        diff1q[1] = -5.0 / 9.0 * t12 - 5.0 / 9.0 * t18 + 5.0 / 9.0 * \
            q_2[0] + 4.0 / 9.0 * t22 + 4.0 / 9.0 * t25 - 4.0 / 9.0 * q_2[1]
        diff1q[2] = 4.0 / 9.0 * t12 + 4.0 / 9.0 * t18 - 4.0 / 9.0 * \
            q_2[0] - 5.0 / 9.0 * t22 - 5.0 / 9.0 * t25 + 5.0 / 9.0 * q_2[1]
        diff1q[3] = -t3 * t5 / t2 * \
            (epsilon_n1[2] - epsilon_p[2]) + 2.0 * q_2[2]
        return diff1q

    def get_diff2ss(self, epsilon_n1, E, nu, sctx):
        diff2ss = zeros([3, 3])
        diff2ss[0, 0] = 5.0 / 9.0
        diff2ss[0, 1] = -4.0 / 9.0
        diff2ss[1, 0] = -4.0 / 9.0
        diff2ss[1, 1] = 5.0 / 9.0
        diff2ss[2, 2] = 2.0
        return diff2ss

    def get_diff2sq(self, epsilon_n1, E, nu, sctx):
        diff2sq = zeros([3, 4])
        diff2sq[0, 1] = -5.0 / 9.0
        diff2sq[0, 2] = 4.0 / 9.0
        diff2sq[1, 1] = 4.0 / 9.0
        diff2sq[1, 2] = -5.0 / 9.0
        diff2sq[2, 3] = -2.0
        return diff2sq

    def get_diff2qq(self, epsilon_n1, E, nu, sctx):
        diff2qq = zeros([4, 4])
        diff2qq[0, 0] = -2.0 / 3.0
        diff2qq[1, 1] = 5.0 / 9.0
        diff2qq[1, 2] = -4.0 / 9.0
        diff2qq[2, 1] = -4.0 / 9.0
        diff2qq[2, 2] = 5.0 / 9.0
        diff2qq[3, 3] = 2.0
        return diff2qq

    view_traits = View(Item('sigma_y'))

#-----------------------------------------------
# old implementation by Zhun
#-----------------------------------------------


class DruckerPrager:
    sigma_0 = 1.
    A = 0.1  # private parameter

    def get_f_trial(self, xi_trial, q_1):
        A = self.A
        sigma_0 = self.sigma_0
        P = np.mat([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [
            0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 2]])
        cronecker_delta = np.mat([1., 1., 1., 0., 0., 0.])
        xi_v = (xi_trial[0] + xi_trial[1] + xi_trial[2]) / 3
        s_xi = xi_trial - (xi_v * cronecker_delta).T
        I_1 = 3 * xi_v
        J_2 = 1. / 2 * ((s_xi).T * P * s_xi)
        f_trial = A * I_1 + np.sqrt(J_2) - np.sqrt(1 / 3) * sigma_0 - q_1
        return f_trial

    def get_diff1s(self, f_diff1s, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_1, q_2):
        epsilon_n = epsilon_n1 - d_epsilon
        A = self.A
        t2 = 1 / (1.0 + nu)
        t3 = E * t2
        t4 = E * nu
        t8 = t2 / (1.0 - 2.0 * nu)
        t10 = t3 + t4 * t8
        t11 = epsilon_n[0, 0] - epsilon_p_n[0, 0] + d_epsilon[0, 0]
        t12 = t10 * t11
        t14 = epsilon_n[0, 1] - epsilon_p_n[0, 1] + d_epsilon[0, 1]
        t16 = t4 * t8 * t14
        t17 = t16 / 3.0
        t18 = epsilon_n[0, 2] - epsilon_p_n[0, 2] + d_epsilon[0, 2]
        t20 = t4 * t8 * t18
        t21 = t20 / 3.0
        t24 = t4 * t8 * t11
        t26 = t10 * t14
        t27 = t26 / 3.0
        t28 = q_2[0, 1] / 3.0
        t29 = t10 * t18
        t30 = t29 / 3.0
        t31 = q_2[0, 2] / 3.0
        t32 = 2.0 / 3.0 * t12 + t17 + t21 - 2.0 / 3.0 * \
            q_2[0, 0] - 2.0 / 3.0 * t24 - t27 + t28 - t30 + t31
        t33 = t32 * t32
        t34 = t24 / 3.0
        t37 = t12 / 3.0
        t39 = q_2[0, 0] / 3.0
        t40 = t34 + 2.0 / 3.0 * t26 + t21 - 2.0 / 3.0 * \
            q_2[0, 1] - t37 - 2.0 / 3.0 * t16 + t39 - t30 + t31
        t41 = t40 * t40
        t45 = t34 + t17 + 2.0 / 3.0 * t29 - 2.0 / 3.0 * \
            q_2[0, 2] - t37 - 2.0 / 3.0 * t20 + t39 - t27 + t28
        t46 = t45 * t45
        t48 = t3 * (epsilon_n[0, 3] - epsilon_p_n[0, 3] + d_epsilon[0, 3])
        t50 = t48 - 2.0 * q_2[0, 3]
        t55 = t3 * (epsilon_n[0, 4] - epsilon_p_n[0, 4] + d_epsilon[0, 4])
        t57 = t55 - 2.0 * q_2[0, 4]
        t62 = t3 * (epsilon_n[0, 5] - epsilon_p_n[0, 5] + d_epsilon[0, 5])
        t64 = t62 - 2.0 * q_2[0, 5]
        t70 = np.sqrt(2.0) / np.sqrt(t33 + t41 + t46 + t50 * (t48 / 2.0 -
                                                              q_2[0, 3]) + t57 * (t55 / 2.0 - q_2[0, 4]) + t64 * (t62 / 2.0 - q_2[0, 5]))
        f_diff1s[0, 0] = A + t70 * t32 / 2.0
        f_diff1s[0, 1] = A + t70 * t40 / 2.0
        f_diff1s[0, 2] = A + t70 * t45 / 2.0
        f_diff1s[0, 3] = t70 * t50 / 2.0
        f_diff1s[0, 4] = t70 * t57 / 2.0
        f_diff1s[0, 5] = t70 * t64 / 2.0
        return

    def get_diff1q(self, f_diff1q, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_1, q_2):
        epsilon_n = epsilon_n1 - d_epsilon
        A = self.A
        t2 = 1 / (1.0 + nu)
        t3 = E * t2
        t4 = E * nu
        t8 = t2 / (1.0 - 2.0 * nu)
        t10 = t3 + t4 * t8
        t11 = epsilon_n[0, 0] - epsilon_p_n[0, 0] + d_epsilon[0, 0]
        t12 = t10 * t11
        t14 = epsilon_n[0, 1] - epsilon_p_n[0, 1] + d_epsilon[0, 1]
        t16 = t4 * t8 * t14
        t17 = t16 / 3.0
        t18 = epsilon_n[0, 2] - epsilon_p_n[0, 2] + d_epsilon[0, 2]
        t20 = t4 * t8 * t18
        t21 = t20 / 3.0
        t24 = t4 * t8 * t11
        t26 = t10 * t14
        t27 = t26 / 3.0
        t28 = q_2[0, 1] / 3.0
        t29 = t10 * t18
        t30 = t29 / 3.0
        t31 = q_2[0, 2] / 3.0
        t32 = 2.0 / 3.0 * t12 + t17 + t21 - 2.0 / 3.0 * \
            q_2[0, 0] - 2.0 / 3.0 * t24 - t27 + t28 - t30 + t31
        t33 = t32 * t32
        t34 = t24 / 3.0
        t37 = t12 / 3.0
        t39 = q_2[0, 0] / 3.0
        t40 = t34 + 2.0 / 3.0 * t26 + t21 - 2.0 / 3.0 * \
            q_2[0, 1] - t37 - 2.0 / 3.0 * t16 + t39 - t30 + t31
        t41 = t40 * t40
        t45 = t34 + t17 + 2.0 / 3.0 * t29 - 2.0 / 3.0 * \
            q_2[0, 2] - t37 - 2.0 / 3.0 * t20 + t39 - t27 + t28
        t46 = t45 * t45
        t48 = t3 * (epsilon_n[0, 3] - epsilon_p_n[0, 3] + d_epsilon[0, 3])
        t50 = t48 - 2.0 * q_2[0, 3]
        t55 = t3 * (epsilon_n[0, 4] - epsilon_p_n[0, 4] + d_epsilon[0, 4])
        t57 = t55 - 2.0 * q_2[0, 4]
        t62 = t3 * (epsilon_n[0, 5] - epsilon_p_n[0, 5] + d_epsilon[0, 5])
        t64 = t62 - 2.0 * q_2[0, 5]
        t70 = np.sqrt(2.0) / np.sqrt(t33 + t41 + t46 + t50 * (t48 / 2.0 -
                                                              q_2[0, 3]) + t57 * (t55 / 2.0 - q_2[0, 4]) + t64 * (t62 / 2.0 - q_2[0, 5]))
        f_diff1q[0, 0] = -1.0
        f_diff1q[0, 1] = -A - t70 * t32 / 2.0
        f_diff1q[0, 2] = -A - t70 * t40 / 2.0
        f_diff1q[0, 3] = -A - t70 * t45 / 2.0
        f_diff1q[0, 4] = -t70 * t50 / 2.0
        f_diff1q[0, 5] = -t70 * t57 / 2.0
        f_diff1q[0, 6] = -t70 * t64 / 2.0
        return

    def get_diff2ss(self, f_diff2ss, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_2):
        epsilon_n = epsilon_n1 - d_epsilon
        t2 = 1 / (1.0 + nu)
        t3 = E * t2
        t4 = E * nu
        t8 = t2 / (1.0 - 2.0 * nu)
        t10 = t3 + t4 * t8
        t11 = epsilon_n[0, 0] - epsilon_p_n[0, 0] + d_epsilon[0, 0]
        t12 = t10 * t11
        t14 = epsilon_n[0, 1] - epsilon_p_n[0, 1] + d_epsilon[0, 1]
        t16 = t4 * t8 * t14
        t17 = t16 / 3.0
        t18 = epsilon_n[0, 2] - epsilon_p_n[0, 2] + d_epsilon[0, 2]
        t20 = t4 * t8 * t18
        t21 = t20 / 3.0
        t24 = t4 * t8 * t11
        t26 = t10 * t14
        t27 = t26 / 3.0
        t28 = q_2[0, 1] / 3.0
        t29 = t10 * t18
        t30 = t29 / 3.0
        t31 = q_2[0, 2] / 3.0
        t32 = 2.0 / 3.0 * t12 + t17 + t21 - 2.0 / 3.0 * \
            q_2[0, 0] - 2.0 / 3.0 * t24 - t27 + t28 - t30 + t31
        t33 = t32 * t32
        t34 = t24 / 3.0
        t37 = t12 / 3.0
        t39 = q_2[0, 0] / 3.0
        t40 = t34 + 2.0 / 3.0 * t26 + t21 - 2.0 / 3.0 * \
            q_2[0, 1] - t37 - 2.0 / 3.0 * t16 + t39 - t30 + t31
        t41 = t40 * t40
        t45 = t34 + t17 + 2.0 / 3.0 * t29 - 2.0 / 3.0 * \
            q_2[0, 2] - t37 - 2.0 / 3.0 * t20 + t39 - t27 + t28
        t46 = t45 * t45
        t48 = t3 * (epsilon_n[0, 3] - epsilon_p_n[0, 3] + d_epsilon[0, 3])
        t50 = t48 - 2.0 * q_2[0, 3]
        t55 = t3 * (epsilon_n[0, 4] - epsilon_p_n[0, 4] + d_epsilon[0, 4])
        t57 = t55 - 2.0 * q_2[0, 4]
        t62 = t3 * (epsilon_n[0, 5] - epsilon_p_n[0, 5] + d_epsilon[0, 5])
        t64 = t62 - 2.0 * q_2[0, 5]
        t68 = t33 + t41 + t46 + t50 * \
            (t48 / 2.0 - q_2[0, 3]) + t57 * (t55 / 2.0 -
                                             q_2[0, 4]) + t64 * (t62 / 2.0 - q_2[0, 5])
        t69 = np.sqrt(2.0) * np.sqrt(t68) / 2.0
        t71 = 2.0 / t69 / t68
        t74 = 1 / t69
        t75 = t74 / 3.0
        t77 = t71 * t32
        t80 = t74 / 6.0
        t81 = -t77 * t40 / 4.0 - t80
        t84 = -t77 * t45 / 4.0 - t80
        t86 = t77 * t50 / 4.0
        t88 = t77 * t57 / 4.0
        t90 = t77 * t64 / 4.0
        t94 = t71 * t40
        t97 = -t94 * t45 / 4.0 - t80
        t99 = t94 * t50 / 4.0
        t101 = t94 * t57 / 4.0
        t103 = t94 * t64 / 4.0
        t107 = t71 * t45
        t109 = t107 * t50 / 4.0
        t111 = t107 * t57 / 4.0
        t113 = t107 * t64 / 4.0
        t114 = t50 * t50
        t118 = t71 * t50
        t120 = t118 * t57 / 4.0
        t122 = t118 * t64 / 4.0
        t123 = t57 * t57
        t129 = t71 * t57 * t64 / 4.0
        t130 = t64 * t64
        f_diff2ss[0, 0] = -t71 * t33 / 4.0 + t75
        f_diff2ss[0, 1] = t81
        f_diff2ss[0, 2] = t84
        f_diff2ss[0, 3] = -t86
        f_diff2ss[0, 4] = -t88
        f_diff2ss[0, 5] = -t90
        f_diff2ss[1, 0] = t81
        f_diff2ss[1, 1] = -t71 * t41 / 4.0 + t75
        f_diff2ss[1, 2] = t97
        f_diff2ss[1, 3] = -t99
        f_diff2ss[1, 4] = -t101
        f_diff2ss[1, 5] = -t103
        f_diff2ss[2, 0] = t84
        f_diff2ss[2, 1] = t97
        f_diff2ss[2, 2] = -t71 * t46 / 4.0 + t75
        f_diff2ss[2, 3] = -t109
        f_diff2ss[2, 4] = -t111
        f_diff2ss[2, 5] = -t113
        f_diff2ss[3, 0] = -t86
        f_diff2ss[3, 1] = -t99
        f_diff2ss[3, 2] = -t109
        f_diff2ss[3, 3] = -t71 * t114 / 4.0 + t74
        f_diff2ss[3, 4] = -t120
        f_diff2ss[3, 5] = -t122
        f_diff2ss[4, 0] = -t88
        f_diff2ss[4, 1] = -t101
        f_diff2ss[4, 2] = -t111
        f_diff2ss[4, 3] = -t120
        f_diff2ss[4, 4] = -t71 * t123 / 4.0 + t74
        f_diff2ss[4, 5] = -t129
        f_diff2ss[5, 0] = -t90
        f_diff2ss[5, 1] = -t103
        f_diff2ss[5, 2] = -t113
        f_diff2ss[5, 3] = -t122
        f_diff2ss[5, 4] = -t129
        f_diff2ss[5, 5] = -t71 * t130 / 4.0 + t74
        return

    def get_diff2sq(self, f_diff2sq, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_2):
        epsilon_n = epsilon_n1 - d_epsilon
        t2 = 1 / (1.0 + nu)
        t3 = E * t2
        t4 = E * nu
        t8 = t2 / (1.0 - 2.0 * nu)
        t10 = t3 + t4 * t8
        t11 = epsilon_n[0, 0] - epsilon_p_n[0, 0] + d_epsilon[0, 0]
        t12 = t10 * t11
        t14 = epsilon_n[0, 1] - epsilon_p_n[0, 1] + d_epsilon[0, 1]
        t16 = t4 * t8 * t14
        t17 = t16 / 3.0
        t18 = epsilon_n[0, 2] - epsilon_p_n[0, 2] + d_epsilon[0, 2]
        t20 = t4 * t8 * t18
        t21 = t20 / 3.0
        t24 = t4 * t8 * t11
        t26 = t10 * t14
        t27 = t26 / 3.0
        t28 = q_2[0, 1] / 3.0
        t29 = t10 * t18
        t30 = t29 / 3.0
        t31 = q_2[0, 2] / 3.0
        t32 = 2.0 / 3.0 * t12 + t17 + t21 - 2.0 / 3.0 * \
            q_2[0, 0] - 2.0 / 3.0 * t24 - t27 + t28 - t30 + t31
        t33 = t32 * t32
        t34 = t24 / 3.0
        t37 = t12 / 3.0
        t39 = q_2[0, 0] / 3.0
        t40 = t34 + 2.0 / 3.0 * t26 + t21 - 2.0 / 3.0 * \
            q_2[0, 1] - t37 - 2.0 / 3.0 * t16 + t39 - t30 + t31
        t41 = t40 * t40
        t45 = t34 + t17 + 2.0 / 3.0 * t29 - 2.0 / 3.0 * \
            q_2[0, 2] - t37 - 2.0 / 3.0 * t20 + t39 - t27 + t28
        t46 = t45 * t45
        t48 = t3 * (epsilon_n[0, 3] - epsilon_p_n[0, 3] + d_epsilon[0, 3])
        t50 = t48 - 2.0 * q_2[0, 3]
        t55 = t3 * (epsilon_n[0, 4] - epsilon_p_n[0, 4] + d_epsilon[0, 4])
        t57 = t55 - 2.0 * q_2[0, 4]
        t62 = t3 * (epsilon_n[0, 5] - epsilon_p_n[0, 5] + d_epsilon[0, 5])
        t64 = t62 - 2.0 * q_2[0, 5]
        t68 = t33 + t41 + t46 + t50 * \
            (t48 / 2.0 - q_2[0, 3]) + t57 * (t55 / 2.0 -
                                             q_2[0, 4]) + t64 * (t62 / 2.0 - q_2[0, 5])
        t69 = np.sqrt(2.0) * np.sqrt(t68) / 2.0
        t71 = 2.0 / t69 / t68
        t72 = t71 * t32
        t75 = 1 / t69
        t76 = t75 / 3.0
        t80 = t75 / 6.0
        t91 = t71 * t40
        t107 = t71 * t45
        t123 = t71 * t50
        t137 = t71 * t57
        t151 = t71 * t64
        f_diff2sq[0, 0] = 0.0
        f_diff2sq[0, 1] = t72 * t32 / 4.0 - t76
        f_diff2sq[0, 2] = t72 * t40 / 4.0 + t80
        f_diff2sq[0, 3] = t72 * t45 / 4.0 + t80
        f_diff2sq[0, 4] = t72 * t50 / 4.0
        f_diff2sq[0, 5] = t72 * t57 / 4.0
        f_diff2sq[0, 6] = t72 * t64 / 4.0
        f_diff2sq[1, 0] = 0.0
        f_diff2sq[1, 1] = t91 * t32 / 4.0 + t80
        f_diff2sq[1, 2] = t91 * t40 / 4.0 - t76
        f_diff2sq[1, 3] = t91 * t45 / 4.0 + t80
        f_diff2sq[1, 4] = t91 * t50 / 4.0
        f_diff2sq[1, 5] = t91 * t57 / 4.0
        f_diff2sq[1, 6] = t91 * t64 / 4.0
        f_diff2sq[2, 0] = 0.0
        f_diff2sq[2, 1] = t107 * t32 / 4.0 + t80
        f_diff2sq[2, 2] = t107 * t40 / 4.0 + t80
        f_diff2sq[2, 3] = t107 * t45 / 4.0 - t76
        f_diff2sq[2, 4] = t107 * t50 / 4.0
        f_diff2sq[2, 5] = t107 * t57 / 4.0
        f_diff2sq[2, 6] = t107 * t64 / 4.0
        f_diff2sq[3, 0] = 0.0
        f_diff2sq[3, 1] = t123 * t32 / 4.0
        f_diff2sq[3, 2] = t123 * t40 / 4.0
        f_diff2sq[3, 3] = t123 * t45 / 4.0
        f_diff2sq[3, 4] = t123 * t50 / 4.0 - t75
        f_diff2sq[3, 5] = t123 * t57 / 4.0
        f_diff2sq[3, 6] = t123 * t64 / 4.0
        f_diff2sq[4, 0] = 0.0
        f_diff2sq[4, 1] = t137 * t32 / 4.0
        f_diff2sq[4, 2] = t137 * t40 / 4.0
        f_diff2sq[4, 3] = t137 * t45 / 4.0
        f_diff2sq[4, 4] = t137 * t50 / 4.0
        f_diff2sq[4, 5] = t137 * t57 / 4.0 - t75
        f_diff2sq[4, 6] = t137 * t64 / 4.0
        f_diff2sq[5, 0] = 0.0
        f_diff2sq[5, 1] = t151 * t32 / 4.0
        f_diff2sq[5, 2] = t151 * t40 / 4.0
        f_diff2sq[5, 3] = t151 * t45 / 4.0
        f_diff2sq[5, 4] = t151 * t50 / 4.0
        f_diff2sq[5, 5] = t151 * t57 / 4.0
        f_diff2sq[5, 6] = t151 * t64 / 4.0 - t75
        return

    def get_diff2qq(self, f_diff2qq, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_2):
        epsilon_n = epsilon_n1 - d_epsilon
        t2 = 1 / (1.0 + nu)
        t3 = E * t2
        t4 = E * nu
        t8 = t2 / (1.0 - 2.0 * nu)
        t10 = t3 + t4 * t8
        t11 = epsilon_n[0, 0] - epsilon_p_n[0, 0] + d_epsilon[0, 0]
        t12 = t10 * t11
        t14 = epsilon_n[0, 1] - epsilon_p_n[0, 1] + d_epsilon[0, 1]
        t16 = t4 * t8 * t14
        t17 = t16 / 3.0
        t18 = epsilon_n[0, 2] - epsilon_p_n[0, 2] + d_epsilon[0, 2]
        t20 = t4 * t8 * t18
        t21 = t20 / 3.0
        t24 = t4 * t8 * t11
        t26 = t10 * t14
        t27 = t26 / 3.0
        t28 = q_2[0, 1] / 3.0
        t29 = t10 * t18
        t30 = t29 / 3.0
        t31 = q_2[0, 2] / 3.0
        t32 = 2.0 / 3.0 * t12 + t17 + t21 - 2.0 / 3.0 * \
            q_2[0, 0] - 2.0 / 3.0 * t24 - t27 + t28 - t30 + t31
        t33 = t32 * t32
        t34 = t24 / 3.0
        t37 = t12 / 3.0
        t39 = q_2[0, 0] / 3.0
        t40 = t34 + 2.0 / 3.0 * t26 + t21 - 2.0 / 3.0 * \
            q_2[0, 1] - t37 - 2.0 / 3.0 * t16 + t39 - t30 + t31
        t41 = t40 * t40
        t45 = t34 + t17 + 2.0 / 3.0 * t29 - 2.0 / 3.0 * \
            q_2[0, 2] - t37 - 2.0 / 3.0 * t20 + t39 - t27 + t28
        t46 = t45 * t45
        t48 = t3 * (epsilon_n[0, 3] - epsilon_p_n[0, 3] + d_epsilon[0, 3])
        t50 = t48 - 2.0 * q_2[0, 3]
        t55 = t3 * (epsilon_n[0, 4] - epsilon_p_n[0, 4] + d_epsilon[0, 4])
        t57 = t55 - 2.0 * q_2[0, 4]
        t62 = t3 * (epsilon_n[0, 5] - epsilon_p_n[0, 5] + d_epsilon[0, 5])
        t64 = t62 - 2.0 * q_2[0, 5]
        t68 = t33 + t41 + t46 + t50 * \
            (t48 / 2.0 - q_2[0, 3]) + t57 * (t55 / 2.0 -
                                             q_2[0, 4]) + t64 * (t62 / 2.0 - q_2[0, 5])
        t69 = np.sqrt(2.0) * np.sqrt(t68) / 2.0
        t71 = 2.0 / t69 / t68
        t74 = 1 / t69
        t75 = t74 / 3.0
        t77 = -t71 * t32
        t80 = t74 / 6.0
        t81 = t77 * t40 / 4.0 - t80
        t84 = t77 * t45 / 4.0 - t80
        t86 = -t77 * t50 / 4.0
        t88 = -t77 * t57 / 4.0
        t90 = -t77 * t64 / 4.0
        t94 = -t71 * t40
        t97 = t94 * t45 / 4.0 - t80
        t99 = -t94 * t50 / 4.0
        t101 = -t94 * t57 / 4.0
        t103 = -t94 * t64 / 4.0
        t107 = -t71 * t45
        t109 = -t107 * t50 / 4.0
        t111 = -t107 * t57 / 4.0
        t113 = -t107 * t64 / 4.0
        t114 = t50 * t50
        t118 = -t71 * t50
        t120 = -t118 * t57 / 4.0
        t122 = -t118 * t64 / 4.0
        t123 = t57 * t57
        t129 = t71 * t57 * t64 / 4.0
        t130 = t64 * t64
        f_diff2qq[0, 0] = 0.0
        f_diff2qq[0, 1] = 0.0
        f_diff2qq[0, 2] = 0.0
        f_diff2qq[0, 3] = 0.0
        f_diff2qq[0, 4] = 0.0
        f_diff2qq[0, 5] = 0.0
        f_diff2qq[0, 6] = 0.0
        f_diff2qq[1, 0] = 0.0
        f_diff2qq[1, 1] = -t71 * t33 / 4.0 + t75
        f_diff2qq[1, 2] = t81
        f_diff2qq[1, 3] = t84
        f_diff2qq[1, 4] = -t86
        f_diff2qq[1, 5] = -t88
        f_diff2qq[1, 6] = -t90
        f_diff2qq[2, 0] = 0.0
        f_diff2qq[2, 1] = t81
        f_diff2qq[2, 2] = -t71 * t41 / 4.0 + t75
        f_diff2qq[2, 3] = t97
        f_diff2qq[2, 4] = -t99
        f_diff2qq[2, 5] = -t101
        f_diff2qq[2, 6] = -t103
        f_diff2qq[3, 0] = 0.0
        f_diff2qq[3, 1] = t84
        f_diff2qq[3, 2] = t97
        f_diff2qq[3, 3] = -t71 * t46 / 4.0 + t75
        f_diff2qq[3, 4] = -t109
        f_diff2qq[3, 5] = -t111
        f_diff2qq[3, 6] = -t113
        f_diff2qq[4, 0] = 0.0
        f_diff2qq[4, 1] = -t86
        f_diff2qq[4, 2] = -t99
        f_diff2qq[4, 3] = -t109
        f_diff2qq[4, 4] = -t71 * t114 / 4.0 + t74
        f_diff2qq[4, 5] = -t120
        f_diff2qq[4, 6] = -t122
        f_diff2qq[5, 0] = 0.0
        f_diff2qq[5, 1] = -t88
        f_diff2qq[5, 2] = -t101
        f_diff2qq[5, 3] = -t111
        f_diff2qq[5, 4] = -t120
        f_diff2qq[5, 5] = -t71 * t123 / 4.0 + t74
        f_diff2qq[5, 6] = -t129
        f_diff2qq[6, 0] = 0.0
        f_diff2qq[6, 1] = -t90
        f_diff2qq[6, 2] = -t103
        f_diff2qq[6, 3] = -t113
        f_diff2qq[6, 4] = -t122
        f_diff2qq[6, 5] = -t129
        f_diff2qq[6, 6] = -t71 * t130 / 4.0 + t74
        return


class Gurson:
    sigma_0 = 1.  # private paremeter
    p = 0.001  # private parameter

    def get_f_trial(self, xi_trial, q_1):
        sigma_0 = self.sigma_0
        p = self.p
        P = np.mat([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [
            0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 2]])
        cronecker_delta = np.mat([1., 1., 1., 0., 0., 0.])
        xi_v = (xi_trial[0] + xi_trial[1] + xi_trial[2]) / 3
        s_xi = xi_trial - (xi_v * cronecker_delta).T
        I_1 = 3 * xi_v
        J_2 = 1. / 2 * ((s_xi).T * P * s_xi)
        f_trial = 3 * J_2 / sigma_0**2 + 2 * p * \
            np.cosh(I_1 / (2 * sigma_0)) - (1 + p**2 + q_1)
        return f_trial

    def get_diff1s(self, f_diff1s, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_1, q_2):
        epsilon_n = epsilon_n1 - d_epsilon
        sigma_0 = self.sigma_0
        p = self.p
        t2 = 1 / (1.0 + nu)
        t3 = E * t2
        t4 = E * nu
        t8 = t2 / (1.0 - 2.0 * nu)
        t10 = t3 + t4 * t8
        t11 = epsilon_n[0, 0] - epsilon_p_n[0, 0] + d_epsilon[0, 0]
        t12 = t10 * t11
        t14 = epsilon_n[0, 1] - epsilon_p_n[0, 1] + d_epsilon[0, 1]
        t16 = t4 * t8 * t14
        t17 = t16 / 3.0
        t18 = epsilon_n[0, 2] - epsilon_p_n[0, 2] + d_epsilon[0, 2]
        t20 = t4 * t8 * t18
        t21 = t20 / 3.0
        t24 = t4 * t8 * t11
        t26 = t10 * t14
        t27 = t26 / 3.0
        t28 = q_2[0, 1] / 3.0
        t29 = t10 * t18
        t30 = t29 / 3.0
        t31 = q_2[0, 2] / 3.0
        t33 = sigma_0 * sigma_0
        t34 = 1 / t33
        t41 = 1 / sigma_0
        t44 = np.sinh((t12 + 2.0 * t16 + 2.0 * t20 -
                       q_2[0, 0] + 2.0 * t24 + t26 - q_2[0, 1] + t29 - q_2[0, 2]) * t41 / 2.0)
        t46 = p * t44 * t41
        t48 = t24 / 3.0
        t51 = t12 / 3.0
        t53 = q_2[0, 0] / 3.0
        f_diff1s[0, 0] = 3.0 * (2.0 / 3.0 * t12 + t17 + t21 - 2.0 / 3.0 *
                                q_2[0, 0] - 2.0 / 3.0 * t24 - t27 + t28 - t30 + t31) * t34 + t46
        f_diff1s[0, 1] = 3.0 * (t48 + 2.0 / 3.0 * t26 + t21 - 2.0 / 3.0 *
                                q_2[0, 1] - t51 - 2.0 / 3.0 * t16 + t53 - t30 + t31) * t34 + t46
        f_diff1s[0, 2] = 3.0 * (t48 + t17 + 2.0 / 3.0 * t29 - 2.0 / 3.0 *
                                q_2[0, 2] - t51 - 2.0 / 3.0 * t20 + t53 - t27 + t28) * t34 + t46
        f_diff1s[0, 3] = 3.0 * (t3 * (epsilon_n[0, 3] - epsilon_p_n[0,
                                                                    3] + d_epsilon[0, 3]) - 2.0 * q_2[0, 3]) * t34
        f_diff1s[0, 4] = 3.0 * (t3 * (epsilon_n[0, 4] - epsilon_p_n[0,
                                                                    4] + d_epsilon[0, 4]) - 2.0 * q_2[0, 4]) * t34
        f_diff1s[0, 5] = 3.0 * (t3 * (epsilon_n[0, 5] - epsilon_p_n[0,
                                                                    5] + d_epsilon[0, 5]) - 2.0 * q_2[0, 5]) * t34
        return

    def get_diff1q(self, f_diff1q, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_1, q_2):
        epsilon_n = epsilon_n1 - d_epsilon
        sigma_0 = self.sigma_0
        p = self.p
        t2 = 1 / (1.0 + nu)
        t3 = E * t2
        t4 = E * nu
        t8 = t2 / (1.0 - 2.0 * nu)
        t10 = t3 + t4 * t8
        t11 = epsilon_n[0, 0] - epsilon_p_n[0, 0] + d_epsilon[0, 0]
        t12 = t10 * t11
        t14 = epsilon_n[0, 1] - epsilon_p_n[0, 1] + d_epsilon[0, 1]
        t16 = t4 * t8 * t14
        t17 = t16 / 3.0
        t18 = epsilon_n[0, 2] - epsilon_p_n[0, 2] + d_epsilon[0, 2]
        t20 = t4 * t8 * t18
        t21 = t20 / 3.0
        t24 = t4 * t8 * t11
        t26 = t10 * t14
        t27 = t26 / 3.0
        t28 = q_2[0, 1] / 3.0
        t29 = t10 * t18
        t30 = t29 / 3.0
        t31 = q_2[0, 2] / 3.0
        t33 = sigma_0 * sigma_0
        t34 = 1 / t33
        t41 = 1 / sigma_0
        t44 = np.sinh((t12 + 2.0 * t16 + 2.0 * t20 -
                       q_2[0, 0] + 2.0 * t24 + t26 - q_2[0, 1] + t29 - q_2[0, 2]) * t41 / 2.0)
        t46 = p * t44 * t41
        t48 = t12 / 3.0
        t50 = q_2[0, 0] / 3.0
        t51 = t24 / 3.0
        f_diff1q[0, 0] = -1.0
        f_diff1q[0, 1] = 3.0 * (-2.0 / 3.0 * t12 - t17 - t21 + 2.0 / 3.0 *
                                q_2[0, 0] + 2.0 / 3.0 * t24 + t27 - t28 + t30 - t31) * t34 - t46
        f_diff1q[0, 2] = 3.0 * (t48 + 2.0 / 3.0 * t16 - t21 - t50 - t51 -
                                2.0 / 3.0 * t26 + 2.0 / 3.0 * q_2[0, 1] + t30 - t31) * t34 - t46
        f_diff1q[0, 3] = 3.0 * (t48 - t17 + 2.0 / 3.0 * t20 - t50 - t51 +
                                t27 - t28 - 2.0 / 3.0 * t29 + 2.0 / 3.0 * q_2[0, 2]) * t34 - t46
        f_diff1q[0, 4] = 3.0 * (-t3 * (epsilon_n[0, 3] - epsilon_p_n[0,
                                                                     3] + d_epsilon[0, 3]) + 2.0 * q_2[0, 3]) * t34
        f_diff1q[0, 5] = 3.0 * (-t3 * (epsilon_n[0, 4] - epsilon_p_n[0,
                                                                     4] + d_epsilon[0, 4]) + 2.0 * q_2[0, 4]) * t34
        f_diff1q[0, 6] = 3.0 * (-t3 * (epsilon_n[0, 5] - epsilon_p_n[0,
                                                                     5] + d_epsilon[0, 5]) + 2.0 * q_2[0, 5]) * t34
        return

    def get_diff2ss(self, f_diff2ss, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_2):
        epsilon_n = epsilon_n1 - d_epsilon
        sigma_0 = self.sigma_0
        p = self.p
        t1 = sigma_0 * sigma_0
        t2 = 1 / t1
        t5 = 1 / (1.0 + nu)
        t7 = E * nu
        t11 = t5 / (1.0 - 2.0 * nu)
        t13 = E * t5 + t7 * t11
        t14 = epsilon_n[0, 0] - epsilon_p_n[0, 0] + d_epsilon[0, 0]
        t16 = epsilon_n[0, 1] - epsilon_p_n[0, 1] + d_epsilon[0, 1]
        t20 = epsilon_n[0, 2] - epsilon_p_n[0, 2] + d_epsilon[0, 2]
        t33 = np.cosh((t13 * t14 + 2.0 * t7 * t11 * t16 + 2.0 * t7 * t11 * t20 -
                       q_2[0, 0] + 2.0 * t7 * t11 * t14 + t13 * t16 - q_2[0, 1] + t13 * t20 - q_2[0, 2]) / sigma_0 / 2.0)
        t36 = p * t33 * t2 / 2.0
        t37 = 2.0 * t2 + t36
        t38 = -t2 + t36
        t39 = 6.0 * t2
        f_diff2ss[0, 0] = t37
        f_diff2ss[0, 1] = t38
        f_diff2ss[0, 2] = t38
        f_diff2ss[0, 3] = 0.0
        f_diff2ss[0, 4] = 0.0
        f_diff2ss[0, 5] = 0.0
        f_diff2ss[1, 0] = t38
        f_diff2ss[1, 1] = t37
        f_diff2ss[1, 2] = t38
        f_diff2ss[1, 3] = 0.0
        f_diff2ss[1, 4] = 0.0
        f_diff2ss[1, 5] = 0.0
        f_diff2ss[2, 0] = t38
        f_diff2ss[2, 1] = t38
        f_diff2ss[2, 2] = t37
        f_diff2ss[2, 3] = 0.0
        f_diff2ss[2, 4] = 0.0
        f_diff2ss[2, 5] = 0.0
        f_diff2ss[3, 0] = 0.0
        f_diff2ss[3, 1] = 0.0
        f_diff2ss[3, 2] = 0.0
        f_diff2ss[3, 3] = t39
        f_diff2ss[3, 4] = 0.0
        f_diff2ss[3, 5] = 0.0
        f_diff2ss[4, 0] = 0.0
        f_diff2ss[4, 1] = 0.0
        f_diff2ss[4, 2] = 0.0
        f_diff2ss[4, 3] = 0.0
        f_diff2ss[4, 4] = t39
        f_diff2ss[4, 5] = 0.0
        f_diff2ss[5, 0] = 0.0
        f_diff2ss[5, 1] = 0.0
        f_diff2ss[5, 2] = 0.0
        f_diff2ss[5, 3] = 0.0
        f_diff2ss[5, 4] = 0.0
        f_diff2ss[5, 5] = t39
        return

    def get_diff2sq(self, f_diff2sq, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_2):
        epsilon_n = epsilon_n1 - d_epsilon
        sigma_0 = self.sigma_0
        p = self.p
        t1 = sigma_0 * sigma_0
        t2 = 1 / t1
        t5 = 1 / (1.0 + nu)
        t7 = E * nu
        t11 = t5 / (1.0 - 2.0 * nu)
        t13 = E * t5 + t7 * t11
        t14 = epsilon_n[0, 0] - epsilon_p_n[0, 0] + d_epsilon[0, 0]
        t16 = epsilon_n[0, 1] - epsilon_p_n[0, 1] + d_epsilon[0, 1]
        t20 = epsilon_n[0, 2] - epsilon_p_n[0, 2] + d_epsilon[0, 2]
        t33 = np.cosh((t13 * t14 + 2.0 * t7 * t11 * t16 + 2.0 * t7 * t11 * t20 -
                       q_2[0, 0] + 2.0 * t7 * t11 * t14 + t13 * t16 - q_2[0, 1] + t13 * t20 - q_2[0, 2]) / sigma_0 / 2.0)
        t36 = p * t33 * t2 / 2.0
        t37 = -2.0 * t2 - t36
        t38 = t2 - t36
        t39 = 6.0 * t2
        f_diff2sq[0, 0] = 0.0
        f_diff2sq[0, 1] = t37
        f_diff2sq[0, 2] = t38
        f_diff2sq[0, 3] = t38
        f_diff2sq[0, 4] = 0.0
        f_diff2sq[0, 5] = 0.0
        f_diff2sq[0, 6] = 0.0
        f_diff2sq[1, 0] = 0.0
        f_diff2sq[1, 1] = t38
        f_diff2sq[1, 2] = t37
        f_diff2sq[1, 3] = t38
        f_diff2sq[1, 4] = 0.0
        f_diff2sq[1, 5] = 0.0
        f_diff2sq[1, 6] = 0.0
        f_diff2sq[2, 0] = 0.0
        f_diff2sq[2, 1] = t38
        f_diff2sq[2, 2] = t38
        f_diff2sq[2, 3] = t37
        f_diff2sq[2, 4] = 0.0
        f_diff2sq[2, 5] = 0.0
        f_diff2sq[2, 6] = 0.0
        f_diff2sq[3, 0] = 0.0
        f_diff2sq[3, 1] = 0.0
        f_diff2sq[3, 2] = 0.0
        f_diff2sq[3, 3] = 0.0
        f_diff2sq[3, 4] = -t39
        f_diff2sq[3, 5] = 0.0
        f_diff2sq[3, 6] = 0.0
        f_diff2sq[4, 0] = 0.0
        f_diff2sq[4, 1] = 0.0
        f_diff2sq[4, 2] = 0.0
        f_diff2sq[4, 3] = 0.0
        f_diff2sq[4, 4] = 0.0
        f_diff2sq[4, 5] = -t39
        f_diff2sq[4, 6] = 0.0
        f_diff2sq[5, 0] = 0.0
        f_diff2sq[5, 1] = 0.0
        f_diff2sq[5, 2] = 0.0
        f_diff2sq[5, 3] = 0.0
        f_diff2sq[5, 4] = 0.0
        f_diff2sq[5, 5] = 0.0
        f_diff2sq[5, 6] = -t39
        return

    def get_diff2qq(self, f_diff2qq, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_2):
        epsilon_n = epsilon_n1 - d_epsilon
        sigma_0 = self.sigma_0
        p = self.p
        t1 = sigma_0 * sigma_0
        t2 = 1 / t1
        t5 = 1 / (1.0 + nu)
        t7 = E * nu
        t11 = t5 / (1.0 - 2.0 * nu)
        t13 = E * t5 + t7 * t11
        t14 = epsilon_n[0, 0] - epsilon_p_n[0, 0] + d_epsilon[0, 0]
        t16 = epsilon_n[0, 1] - epsilon_p_n[0, 1] + d_epsilon[0, 1]
        t20 = epsilon_n[0, 2] - epsilon_p_n[0, 2] + d_epsilon[0, 2]
        t33 = np.cosh((t13 * t14 + 2.0 * t7 * t11 * t16 + 2.0 * t7 * t11 * t20 -
                       q_2[0, 0] + 2.0 * t7 * t11 * t14 + t13 * t16 - q_2[0, 1] + t13 * t20 - q_2[0, 2]) / sigma_0 / 2.0)
        t36 = p * t33 * t2 / 2.0
        t37 = 2.0 * t2 + t36
        t38 = -t2 + t36
        t39 = 6.0 * t2
        f_diff2qq[0, 0] = 0.0
        f_diff2qq[0, 1] = 0.0
        f_diff2qq[0, 2] = 0.0
        f_diff2qq[0, 3] = 0.0
        f_diff2qq[0, 4] = 0.0
        f_diff2qq[0, 5] = 0.0
        f_diff2qq[0, 6] = 0.0
        f_diff2qq[1, 0] = 0.0
        f_diff2qq[1, 1] = t37
        f_diff2qq[1, 2] = t38
        f_diff2qq[1, 3] = t38
        f_diff2qq[1, 4] = 0.0
        f_diff2qq[1, 5] = 0.0
        f_diff2qq[1, 6] = 0.0
        f_diff2qq[2, 0] = 0.0
        f_diff2qq[2, 1] = t38
        f_diff2qq[2, 2] = t37
        f_diff2qq[2, 3] = t38
        f_diff2qq[2, 4] = 0.0
        f_diff2qq[2, 5] = 0.0
        f_diff2qq[2, 6] = 0.0
        f_diff2qq[3, 0] = 0.0
        f_diff2qq[3, 1] = t38
        f_diff2qq[3, 2] = t38
        f_diff2qq[3, 3] = t37
        f_diff2qq[3, 4] = 0.0
        f_diff2qq[3, 5] = 0.0
        f_diff2qq[3, 6] = 0.0
        f_diff2qq[4, 0] = 0.0
        f_diff2qq[4, 1] = 0.0
        f_diff2qq[4, 2] = 0.0
        f_diff2qq[4, 3] = 0.0
        f_diff2qq[4, 4] = t39
        f_diff2qq[4, 5] = 0.0
        f_diff2qq[4, 6] = 0.0
        f_diff2qq[5, 0] = 0.0
        f_diff2qq[5, 1] = 0.0
        f_diff2qq[5, 2] = 0.0
        f_diff2qq[5, 3] = 0.0
        f_diff2qq[5, 4] = 0.0
        f_diff2qq[5, 5] = t39
        f_diff2qq[5, 6] = 0.0
        f_diff2qq[6, 0] = 0.0
        f_diff2qq[6, 1] = 0.0
        f_diff2qq[6, 2] = 0.0
        f_diff2qq[6, 3] = 0.0
        f_diff2qq[6, 4] = 0.0
        f_diff2qq[6, 5] = 0.0
        f_diff2qq[6, 6] = t39
        return


class CamClay:
    sigma_0 = 1.
    p_c = 4.
    M = 1.

    def get_f_trial(self, xi_trial, q_1):
        sigma_0 = self.sigma_0
        p_c = self.p_c
        M = self.M
        P = np.mat([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [
            0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 2]])
        cronecker_delta = np.mat([1., 1., 1., 0., 0., 0.])
        xi_v = (xi_trial[0] + xi_trial[1] + xi_trial[2]) / 3
        s_xi = xi_trial - (xi_v * cronecker_delta).T
        I_1 = 3 * xi_v
        J_2 = 1. / 2 * ((s_xi).T * P * s_xi)
        f_trial = 27 * J_2 + M**2 * I_1 * (I_1 + 3 * p_c) - q_1
        return f_trial

    def get_diff1s(self, f_diff1s, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_1, q_2):
        epsilon_n = epsilon_n1 - d_epsilon
        p_c = self.p_c
        M = self.M
        t2 = 1 / (1.0 + nu)
        t3 = E * t2
        t4 = E * nu
        t8 = t2 / (1.0 - 2.0 * nu)
        t10 = t3 + t4 * t8
        t11 = epsilon_n[0, 0] - epsilon_p_n[0, 0] + d_epsilon[0, 0]
        t12 = t10 * t11
        t14 = epsilon_n[0, 1] - epsilon_p_n[0, 1] + d_epsilon[0, 1]
        t16 = t4 * t8 * t14
        t17 = 9.0 * t16
        t18 = epsilon_n[0, 2] - epsilon_p_n[0, 2] + d_epsilon[0, 2]
        t20 = t4 * t8 * t18
        t21 = 9.0 * t20
        t24 = t4 * t8 * t11
        t26 = t10 * t14
        t27 = 9.0 * t26
        t28 = 9.0 * q_2[0, 1]
        t29 = t10 * t18
        t30 = 9.0 * t29
        t31 = 9.0 * q_2[0, 2]
        t32 = M * M
        t33 = 2.0 * t16
        t34 = 2.0 * t20
        t35 = 2.0 * t24
        t38 = t32 * (t12 + t33 + t34 - q_2[0, 0] + t35 +
                     t26 - q_2[0, 1] + t29 - q_2[0, 2] + 3.0 * p_c)
        t40 = t32 * (t12 + t33 + t34 -
                     q_2[0, 0] + t35 + t26 - q_2[0, 1] + t29 - q_2[0, 2])
        t41 = 18.0 * t12 + t17 + t21 - 18.0 * \
            q_2[0, 0] - 18.0 * t24 - t27 + t28 - t30 + t31 + t38 + t40
        t42 = 9.0 * t12
        t44 = 9.0 * q_2[0, 0]
        t45 = 9.0 * t24
        t48 = -t42 - 18.0 * t16 + t21 + t44 + t45 + 18.0 * \
            t26 - 18.0 * q_2[0, 1] - t30 + t31 + t38 + t40
        t52 = -t42 + t17 - 18.0 * t20 + t44 + t45 - t27 + \
            t28 + 18.0 * t29 - 18.0 * q_2[0, 2] + t38 + t40
        f_diff1s[0, 0] = t41
        f_diff1s[0, 1] = t48
        f_diff1s[0, 2] = t52
        f_diff1s[0, 3] = 27.0 * t3 * \
            (epsilon_n[0, 3] - epsilon_p_n[0, 3] +
             d_epsilon[0, 3]) - 54.0 * q_2[0, 3]
        f_diff1s[0, 4] = 27.0 * t3 * \
            (epsilon_n[0, 4] - epsilon_p_n[0, 4] +
             d_epsilon[0, 4]) - 54.0 * q_2[0, 4]
        f_diff1s[0, 5] = 27.0 * t3 * \
            (epsilon_n[0, 5] - epsilon_p_n[0, 5] +
             d_epsilon[0, 5]) - 54.0 * q_2[0, 5]
        return

    def get_diff1q(self, f_diff1q, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_1, q_2):
        epsilon_n = epsilon_n1 - d_epsilon
        p_c = self.p_c
        M = self.M
        t2 = 1 / (1.0 + nu)
        t3 = E * t2
        t4 = E * nu
        t8 = t2 / (1.0 - 2.0 * nu)
        t10 = t3 + t4 * t8
        t11 = epsilon_n[0, 0] - epsilon_p_n[0, 0] + d_epsilon[0, 0]
        t12 = t10 * t11
        t14 = epsilon_n[0, 1] - epsilon_p_n[0, 1] + d_epsilon[0, 1]
        t16 = t4 * t8 * t14
        t17 = 9.0 * t16
        t18 = epsilon_n[0, 2] - epsilon_p_n[0, 2] + d_epsilon[0, 2]
        t20 = t4 * t8 * t18
        t21 = 9.0 * t20
        t24 = t4 * t8 * t11
        t26 = t10 * t14
        t27 = 9.0 * t26
        t28 = 9.0 * q_2[0, 1]
        t29 = t10 * t18
        t30 = 9.0 * t29
        t31 = 9.0 * q_2[0, 2]
        t32 = M * M
        t33 = 2.0 * t16
        t34 = 2.0 * t20
        t35 = 2.0 * t24
        t38 = t32 * (t12 + t33 + t34 - q_2[0, 0] + t35 +
                     t26 - q_2[0, 1] + t29 - q_2[0, 2] + 3.0 * p_c)
        t40 = t32 * (t12 + t33 + t34 -
                     q_2[0, 0] + t35 + t26 - q_2[0, 1] + t29 - q_2[0, 2])
        t41 = -18.0 * t12 - t17 - t21 + 18.0 * \
            q_2[0, 0] + 18.0 * t24 + t27 - t28 + t30 - t31 - t38 - t40
        t42 = 9.0 * t12
        t44 = 9.0 * q_2[0, 0]
        t45 = 9.0 * t24
        t48 = t42 + 18.0 * t16 - t21 - t44 - t45 - 18.0 * \
            t26 + 18.0 * q_2[0, 1] + t30 - t31 - t38 - t40
        t52 = t42 - t17 + 18.0 * t20 - t44 - t45 + t27 - \
            t28 - 18.0 * t29 + 18.0 * q_2[0, 2] - t38 - t40
        f_diff1q[0, 0] = -1.0
        f_diff1q[0, 1] = t41
        f_diff1q[0, 2] = t48
        f_diff1q[0, 3] = t52
        f_diff1q[0, 4] = -27.0 * t3 * \
            (epsilon_n[0, 3] - epsilon_p_n[0, 3] +
             d_epsilon[0, 3]) + 54.0 * q_2[0, 3]
        f_diff1q[0, 5] = -27.0 * t3 * \
            (epsilon_n[0, 4] - epsilon_p_n[0, 4] +
             d_epsilon[0, 4]) + 54.0 * q_2[0, 4]
        f_diff1q[0, 6] = -27.0 * t3 * \
            (epsilon_n[0, 5] - epsilon_p_n[0, 5] +
             d_epsilon[0, 5]) + 54.0 * q_2[0, 5]
        return

    def get_diff2ss(self, f_diff2ss, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_2):
        M = self.M
        t1 = M * M
        t2 = 2.0 * t1
        t3 = 18.0 + t2
        t4 = -9.0 + t2
        f_diff2ss[0, 0] = t3
        f_diff2ss[0, 1] = t4
        f_diff2ss[0, 2] = t4
        f_diff2ss[0, 3] = 0.0
        f_diff2ss[0, 4] = 0.0
        f_diff2ss[0, 5] = 0.0
        f_diff2ss[1, 0] = t4
        f_diff2ss[1, 1] = t3
        f_diff2ss[1, 2] = t4
        f_diff2ss[1, 3] = 0.0
        f_diff2ss[1, 4] = 0.0
        f_diff2ss[1, 5] = 0.0
        f_diff2ss[2, 0] = t4
        f_diff2ss[2, 1] = t4
        f_diff2ss[2, 2] = t3
        f_diff2ss[2, 3] = 0.0
        f_diff2ss[2, 4] = 0.0
        f_diff2ss[2, 5] = 0.0
        f_diff2ss[3, 0] = 0.0
        f_diff2ss[3, 1] = 0.0
        f_diff2ss[3, 2] = 0.0
        f_diff2ss[3, 3] = 54.0
        f_diff2ss[3, 4] = 0.0
        f_diff2ss[3, 5] = 0.0
        f_diff2ss[4, 0] = 0.0
        f_diff2ss[4, 1] = 0.0
        f_diff2ss[4, 2] = 0.0
        f_diff2ss[4, 3] = 0.0
        f_diff2ss[4, 4] = 54.0
        f_diff2ss[4, 5] = 0.0
        f_diff2ss[5, 0] = 0.0
        f_diff2ss[5, 1] = 0.0
        f_diff2ss[5, 2] = 0.0
        f_diff2ss[5, 3] = 0.0
        f_diff2ss[5, 4] = 0.0
        f_diff2ss[5, 5] = 54.0
        return

    def get_diff2sq(self, f_diff2sq, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_2):
        M = self.M
        t1 = M * M
        t2 = 2.0 * t1
        t3 = -18.0 - t2
        t4 = 9.0 - t2
        f_diff2sq[0, 0] = 0.0
        f_diff2sq[0, 1] = t3
        f_diff2sq[0, 2] = t4
        f_diff2sq[0, 3] = t4
        f_diff2sq[0, 4] = 0.0
        f_diff2sq[0, 5] = 0.0
        f_diff2sq[0, 6] = 0.0
        f_diff2sq[1, 0] = 0.0
        f_diff2sq[1, 1] = t4
        f_diff2sq[1, 2] = t3
        f_diff2sq[1, 3] = t4
        f_diff2sq[1, 4] = 0.0
        f_diff2sq[1, 5] = 0.0
        f_diff2sq[1, 6] = 0.0
        f_diff2sq[2, 0] = 0.0
        f_diff2sq[2, 1] = t4
        f_diff2sq[2, 2] = t4
        f_diff2sq[2, 3] = t3
        f_diff2sq[2, 4] = 0.0
        f_diff2sq[2, 5] = 0.0
        f_diff2sq[2, 6] = 0.0
        f_diff2sq[3, 0] = 0.0
        f_diff2sq[3, 1] = 0.0
        f_diff2sq[3, 2] = 0.0
        f_diff2sq[3, 3] = 0.0
        f_diff2sq[3, 4] = -54.0
        f_diff2sq[3, 5] = 0.0
        f_diff2sq[3, 6] = 0.0
        f_diff2sq[4, 0] = 0.0
        f_diff2sq[4, 1] = 0.0
        f_diff2sq[4, 2] = 0.0
        f_diff2sq[4, 3] = 0.0
        f_diff2sq[4, 4] = 0.0
        f_diff2sq[4, 5] = -54.0
        f_diff2sq[4, 6] = 0.0
        f_diff2sq[5, 0] = 0.0
        f_diff2sq[5, 1] = 0.0
        f_diff2sq[5, 2] = 0.0
        f_diff2sq[5, 3] = 0.0
        f_diff2sq[5, 4] = 0.0
        f_diff2sq[5, 5] = 0.0
        f_diff2sq[5, 6] = -54.0
        return

    def get_diff2qq(self, f_diff2qq, epsilon_n1, d_epsilon, epsilon_p_n, E, nu, q_2):
        M = self.M
        t1 = M * M
        t2 = 2.0 * t1
        t3 = 18.0 + t2
        t4 = -9.0 + t2
        f_diff2qq[0, 0] = 0.0
        f_diff2qq[0, 1] = 0.0
        f_diff2qq[0, 2] = 0.0
        f_diff2qq[0, 3] = 0.0
        f_diff2qq[0, 4] = 0.0
        f_diff2qq[0, 5] = 0.0
        f_diff2qq[0, 6] = 0.0
        f_diff2qq[1, 0] = 0.0
        f_diff2qq[1, 1] = t3
        f_diff2qq[1, 2] = t4
        f_diff2qq[1, 3] = t4
        f_diff2qq[1, 4] = 0.0
        f_diff2qq[1, 5] = 0.0
        f_diff2qq[1, 6] = 0.0
        f_diff2qq[2, 0] = 0.0
        f_diff2qq[2, 1] = t4
        f_diff2qq[2, 2] = t3
        f_diff2qq[2, 3] = t4
        f_diff2qq[2, 4] = 0.0
        f_diff2qq[2, 5] = 0.0
        f_diff2qq[2, 6] = 0.0
        f_diff2qq[3, 0] = 0.0
        f_diff2qq[3, 1] = t4
        f_diff2qq[3, 2] = t4
        f_diff2qq[3, 3] = t3
        f_diff2qq[3, 4] = 0.0
        f_diff2qq[3, 5] = 0.0
        f_diff2qq[3, 6] = 0.0
        f_diff2qq[4, 0] = 0.0
        f_diff2qq[4, 1] = 0.0
        f_diff2qq[4, 2] = 0.0
        f_diff2qq[4, 3] = 0.0
        f_diff2qq[4, 4] = 54.0
        f_diff2qq[4, 5] = 0.0
        f_diff2qq[4, 6] = 0.0
        f_diff2qq[5, 0] = 0.0
        f_diff2qq[5, 1] = 0.0
        f_diff2qq[5, 2] = 0.0
        f_diff2qq[5, 3] = 0.0
        f_diff2qq[5, 4] = 0.0
        f_diff2qq[5, 5] = 54.0
        f_diff2qq[5, 6] = 0.0
        f_diff2qq[6, 0] = 0.0
        f_diff2qq[6, 1] = 0.0
        f_diff2qq[6, 2] = 0.0
        f_diff2qq[6, 3] = 0.0
        f_diff2qq[6, 4] = 0.0
        f_diff2qq[6, 5] = 0.0
        f_diff2qq[6, 6] = 54.0
        return


# class Rankine:

# class Tresca:

# class MohrCoulomb:

if __name__ == '__main__':
    xi_trial = array([2., 1., 0.])
    q_1 = 0.
    yf = J2()
    yf.get_f_trial(xi_trial, q_1)
    yf.configure_traits()
