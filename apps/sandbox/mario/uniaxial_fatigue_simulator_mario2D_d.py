#!/usr/bin/env python
# coding: utf-8

# # Simulation of fatigue for uniaxial stress state

# Assume a uniaxial stress state with $\sigma_{11} = \bar{\sigma}(\theta)$ representing the loading function. All other components of the stress tensor are assumed zero
# \begin{align}
# \sigma_{ab} = 0; \forall a,b \in (0,1,2), a = b \neq 1
# \end{align}
#

# In[1]:


from apps.sandbox.mario.Micro2Dplot_d import Micro2Dplot_d
from apps.sandbox.mario.vmats2D_mpl_csd_eeq import MATSXDMplCDSEEQ
from apps.sandbox.mario.vmats2D_mpl_d_eeq import MATSXDMicroplaneDamageEEQ
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import MATS3DDesmorat
from ibvpy.mats.mats3D.mats3D_sdamage.vmats3D_sdamage import MATS3DScalarDamage
import matplotlib.pyplot as plt
import numpy as np


def get_eps_ab(eps_O):

    eps_ab = np.zeros((2, 2))
    eps_ab[0, 0] = eps_O[0]
    eps_ab[0, 1] = eps_O[2]
    eps_ab[1, 0] = eps_O[2]
    eps_ab[1, 1] = eps_O[1]

    return eps_ab


def get_sig_O(sig_ab):

    sig_O = np.zeros((3, 1))
    sig_O[0] = sig_ab[0, 0]
    sig_O[1] = sig_ab[1, 1]
    sig_O[2] = sig_ab[1, 0]

    return sig_O


def get_K_OP(D_abcd):

    K_OP = np.zeros((3, 3))

    K_OP[0, 0] = D_abcd[0, 0, 0, 0]
    K_OP[0, 1] = D_abcd[0, 0, 1, 1]
    K_OP[0, 2] = 2 * D_abcd[0, 0, 0, 1]

    K_OP[1, 0] = D_abcd[1, 1, 0, 0]
    K_OP[1, 1] = D_abcd[1, 1, 1, 1]
    K_OP[1, 2] = 2 * D_abcd[1, 1, 0, 1]

    K_OP[2, 0] = D_abcd[0, 1, 0, 0]
    K_OP[2, 1] = D_abcd[0, 1, 1, 1]
    K_OP[2, 2] = 2 * D_abcd[0, 1, 0, 1]

    return K_OP


m = MATSXDMicroplaneDamageEEQ()
plot = Micro2Dplot_d()


def get_UF_t(F, n_t):

    n_mp = 360
    omega = np.zeros((n_mp, ))
    kappa = np.zeros((n_mp, ))
    eps_T = np.zeros((n_mp, 2))
    eps_T = np.zeros((n_mp, ))
    sctx = np.zeros((n_mp, 5))
    sctx = sctx[np.newaxis, :, :]

    # total number of DOFs
    n_O = 3
    # Global vectors
    F_ext = np.zeros((n_O,), np.float_)
    F_O = np.zeros((n_O,), np.float_)
    U_k_O = np.zeros((n_O,), dtype=np.float_)
    # Setup the system matrix with displacement constraints
    # Time stepping parameters
    t_n1, t_max, t_step = 0, 1, 1 / n_t
    # Iteration parameters
    k_max, R_acc = 1000, 1e-3
    # Record solutions
    U_t_list, F_t_list = [np.copy(U_k_O)], [np.copy(F_O)]

    # Load increment loop
    while t_n1 <= t_max:
        #print('t:', t_n1)
        F_ext[0] = F(t_n1)
        k = 0
        # Equilibrium iteration loop
        while k < k_max:
            # Transform the primary vector to field
            eps_ab = get_eps_ab(U_k_O)
            # Stress and material stiffness

            D_abcd, sig_ab = m.get_corr_pred(
                eps_ab, 1, kappa
            )

            # Internal force
            F_O = get_sig_O(sig_ab).reshape(3,)
            # Residuum
            R_O = F_ext - F_O
            # System matrix
            K_OP = get_K_OP(D_abcd)
            # Convergence criterion
            R_norm = np.linalg.norm(R_O)
            delta_U_O = np.linalg.solve(K_OP, R_O)
            U_k_O += delta_U_O
            if R_norm < R_acc:
                # Convergence reached
                break
            # Next iteration
            k += 1

        else:
            print('no convergence')
            break

        # Update states variables after convergence
        kappa = m.update_state_variables(eps_ab,  kappa)
        omega = m._get_omega(kappa)
        eps_T = m._get_e_T_Emna(eps_ab)
        eps_N = m._get_e_N_Emn(eps_ab)

#         omegaN = omegaN.reshape(n_mp, )

        sctx_aux = np.concatenate(
            (kappa.reshape(n_mp, 1), omega.reshape(n_mp, 1), eps_T, eps_N.reshape(n_mp, 1)), axis=1)

        sctx_aux = sctx_aux[np.newaxis, :, :]
        sctx = np.concatenate((sctx, sctx_aux))
        U_t_list.append(np.copy(U_k_O))
        F_t_list.append(F_O)

        t_n1 += t_step

    U_t, F_t = np.array(U_t_list), np.array(F_t_list)
    return U_t, F_t, sctx


n_cycles = 1
T = 1 / n_cycles


def loading_history(t): return (np.sin(np.pi / T * (t - T / 2)) + 1) / 2


U, F, S = get_UF_t(
    F=lambda t: -80 * loading_history(t),
    n_t=50 * n_cycles
)


# **Examples of postprocessing**:
# Plot the axial strain against the lateral strain


# eps_N = np.zeros((len(S), len(S[1])))
# eps_T = np.zeros((len(S), len(S[1])))
# norm_alphaT = np.zeros((len(S), len(S[1])))
norm_eps_T = np.zeros((len(S), len(S[1])))
#
for i in range(len(F)):
    norm_eps_T[i] = np.sqrt(
        np.einsum('...i,...i->... ', S[i, :, 2:4], S[i, :, 2:4]))
#
#
# t = np.arange(len(F))
# fig, axs = plt.subplots(2, 5)
# for i in range(len(S[1])):
#     axs[0, 0].plot(F[:, 0], S[:, i, 0])
# axs[0, 0].set_title('omegaN')
#
# for i in range(len(S)):
#     axs[0, 1].plot(F[:, 0], S[:, i, 1])
# axs[0, 1].set_title('rN')
#
# for i in range(len(S)):
#     axs[0, 2].plot(F[:, 0], S[:, i, 2])
# axs[0, 2].set_title('alphaN')
#
# for i in range(len(S)):
#     axs[0, 3].plot(F[:, 0], S[:, i, 3])
# axs[0, 3].set_title('zN')
#
# for i in range(len(S)):
#     axs[0, 4].plot(F[:, 0], S[:, i, 4])
# axs[0, 4].set_title('eps_pN')
#
# for i in range(len(S)):
#     axs[1, 0].plot(F[:, 0], S[:, i, 6])
# axs[1, 0].set_title('omegaT')
#
# for i in range(len(S)):
#     axs[1, 1].plot(F[:, 0], S[:, i, 7])
# axs[1, 1].set_title('zT')
#
# for i in range(len(S)):
#     axs[1, 2].plot(F[:, 0], norm_alphaT[:, i])
# axs[1, 2].set_title('alphaT')
#
# for i in range(len(S[1])):
#     axs[1, 2].plot(F[:, 0], norm_eps_pT[:, i])
# axs[1, 3].set_title('eps_pT')
#
# # axs[1, 4].plot(F[:, 0], S[:, 6, 4], 'g')
# # #axs[1, 2].plot(F[:, 0], S[:, 26, 2], 'r')
# # axs[1, 4].set_title('eps_pN')
#
#
# plt.show()
#
#
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 4))
ax1.plot(U[:, 0], U[:, 1])
ax2.plot(U[:, 0], F[:, 0])

plt.show()

n_mp = 360
plot.get_2Dviz(n_mp, S[:, :, 0], S[:, :, 1], norm_eps_T, S[:, :, 4])
