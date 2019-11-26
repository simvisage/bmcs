#!/usr/bin/env python
# coding: utf-8

# # Simulation of fatigue for uniaxial stress state

# Assume a uniaxial stress state with $\sigma_{11} = \bar{\sigma}(\theta)$ representing the loading function. All other components of the stress tensor are assumed zero
# \begin{align}
# \sigma_{ab} = 0; \forall a,b \in (0,1,2), a = b \neq 1
# \end{align}
#

# In[1]:

import copy

from ibvpy.mats.mats3D.mats3D_microplane import MATS3DMplCSDEEQ
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import MATS3DDesmorat
from ibvpy.mats.mats3D.mats3D_sdamage.vmats3D_sdamage import MATS3DScalarDamage

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

#from .vmats3D_mpl_csd_eeq import MATS3DMplCSDEEQ


sp.init_printing()


# # Assembly operators
# To construct the mapping between the tensorial representation of the
# stress and states and the equilibrium equations used to solve to find
# the displacement for a given load increment let us introduce the tensor
# mapping operators:

# **Kronecker delta**: defined as
# \begin{align}
# \delta_{ab} = 1 \;  \mathrm{for} \; a = b \;  \mathrm{and} \; \; \delta_{ab} = 0 \; \mathrm{for} \; a \neq 0
# \end{align}

# In[2]:


DELTA = np.identity(3)


# **Levi-Civita operator**: defined as

# In[3]:


# Levi Civita symbol
EPS = np.zeros((3, 3, 3), dtype='f')
EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1


DD = np.hstack([DELTA, np.zeros_like(DELTA)])
EEPS = np.hstack([np.zeros_like(EPS), EPS])

GAMMA = np.einsum(
    'ik,jk->kij', DD, DD
) + np.einsum(
    'ikj->kij', np.fabs(EEPS)
)


def get_eps_ab(eps_O): return np.einsum(
    'Oab,...O->...ab', GAMMA, eps_O
)[np.newaxis, ...]


GAMMA_inv = np.einsum(
    'aO,bO->Oab', DD, DD
) + 0.5 * np.einsum(
    'aOb->Oab', np.fabs(EEPS)
)


def get_sig_O(sig_ab): return np.einsum(
    'Oab,...ab->...O', GAMMA_inv, sig_ab
)[0, ...]


GG = np.einsum(
    'Oab,Pcd->OPabcd', GAMMA_inv, GAMMA_inv
)


def get_K_OP(D_abcd):
    return np.einsum(
        'OPabcd,abcd->OP', GG, D_abcd
    )

# The above operators provide the three mappings
# map the primary variable to from vector to field
# map the residuum field to evctor (assembly operator)
# map the gradient of the residuum field to system matrix


m = MATS3DMplCSDEEQ()
m.configure_traits()


def get_UF_t(eps_max, time_function, n_t):
    state_vars = {
        name: np.zeros(shape, dtype=np.float_)[np.newaxis, ...]
        for name, shape in m.state_var_shapes.items()
    }
    # total number of DOFs
    n_O = 6
    # Global vectors
    F_ext = np.zeros((n_O,), np.float_)
    F_O = np.zeros((n_O,), np.float_)
    U_k_O = np.zeros((n_O,), dtype=np.float_)
    # Construct index maps distinguishing the controlled displacements
    # and free displacements. Here we simply say the the control displacement
    # is the first one. Then index maps are constructed using the np.where
    # function returning the indexes of True positions in a logical array.
    CONTROL = 0
    FREE = slice(1, None)  # This means all except the first index, i.e. [1:]
    # Setup the system matrix with displacement constraints
    # Time stepping parameters
    t_n1, t_max, t_step = 0, 1, 1 / n_t
    t_n = 0
    # Iteration parameters
    k_max, R_acc = 100, 1e-3
    # Record solutions
    U_t_list, F_t_list = [np.copy(U_k_O)], [np.copy(F_O)]
    state_var_list = [copy.deepcopy(state_vars)]
    # Load increment loop
    while t_n1 <= t_max:
        print('t:', t_n1)
        # Get the displacement increment for this step
        delta_U = eps_max * (time_function(t_n1) - time_function(t_n))
        k = 0
        # Equilibrium iteration loop
        while k < k_max:
            # Transform the primary vector to field
            eps_ab = get_eps_ab(U_k_O)
            # Stress and material stiffness
            trial_state_vars = copy.deepcopy(state_vars)
            sig_ab, D_abcd = m.get_corr_pred(
                eps_ab, 1, **trial_state_vars
            )
            # Internal force
            F_O = get_sig_O(sig_ab)
            # System matrix
            K_OP = get_K_OP(D_abcd[0])
            # Get the balancing forces - NOTE - for more displacements
            # this should be an assembly operator.
            # KU remains a 2-d array so we have to make it a vector
            KU = K_OP[:, CONTROL] * delta_U
            # Residuum
            R_O = F_ext - F_O - KU
            # Convergence criterion
            R_norm = np.linalg.norm(R_O[FREE])
            if R_norm < R_acc:
                # Convergence reached
                state_vars = copy.deepcopy(trial_state_vars)
                break
            # Next iteration -
            delta_U_O = np.linalg.solve(K_OP[FREE, FREE], R_O[FREE])
            # Update total displacement
            U_k_O[FREE] += delta_U_O
            # Update control displacement
            U_k_O[CONTROL] += delta_U
            # Note - control displacement nonzero only in the first iteration.
            delta_U = 0
            k += 1
        else:
            print('no convergence')
            break

        # Target time of the next load increment
        U_t_list.append(np.copy(U_k_O))
        F_t_list.append(F_O)
        state_var_list.append(copy.deepcopy(state_vars))
        t_n = t_n1
        t_n1 += t_step

    U_t, F_t = np.array(U_t_list), np.array(F_t_list)
    return U_t, F_t, state_var_list


n_cycles = 8
T = 1 / n_cycles


def loading_history(t): return (np.sin(np.pi / T * (t - T / 2)) + 1) / 2


U, F, S = get_UF_t(
    eps_max=-0.02,
    time_function=lambda t: loading_history(t),
    n_t=50
)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
ax1.plot(U[:, (1, 2)], U[:, 0])
ax1.set_ylabel(r'$\varepsilon_{\mathrm{axial}}$')
ax1.set_xlabel(r'$\varepsilon_{\mathrm{lateral}}$')
ax2.plot(U[:, 0], F[:, 0])
ax2.set_ylabel(r'$\sigma_{\mathrm{axial}}$')
ax2.set_xlabel(r'$\varepsilon_{\mathrm{axial}}$')
plt.show()
