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

from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import MATS3DDesmorat
from ibvpy.mats.mats3D.mats3D_sdamage.vmats3D_sdamage import MATS3DScalarDamage
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
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


# In[11]:

from .vmats3D_mpl_csd_eeq import MATS3DMplCSDEEQ
m = MATS3DMplCSDEEQ()


def get_UF_t(F, n_t):
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
    # Setup the system matrix with displacement constraints
    # Time stepping parameters
    t_n1, t_max, t_step = 0, 1, 1 / n_t
    # Iteration parameters
    k_max, R_acc = 1000, 1e-3
    # Record solutions
    U_t_list, F_t_list = [np.copy(U_k_O)], [np.copy(F_O)]
    state_var_list = [copy.deepcopy(state_vars)]
    # Load increment loop
    while t_n1 <= t_max:
        print('t:', t_n1)
        F_ext[0] = F(t_n1)
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
            # Residuum
            R_O = F_ext - F_O
            # System matrix
            K_OP = get_K_OP(D_abcd[0])
            # Convergence criterion
            R_norm = np.linalg.norm(R_O)
            if R_norm < R_acc:
                # Convergence reached
                state_vars = copy.deepcopy(trial_state_vars)
                break
            # Next iteration
            delta_U_O = np.linalg.solve(K_OP, R_O)
            U_k_O += delta_U_O
            k += 1
        else:
            print('no convergence')
            break

        # Target time of the next load increment
        U_t_list.append(np.copy(U_k_O))
        F_t_list.append(F_O)
        state_var_list.append(copy.deepcopy(state_vars))
        t_n1 += t_step

    U_t, F_t = np.array(U_t_list), np.array(F_t_list)
    return U_t, F_t, state_var_list


# In[14]:

n_cycles = 10
T = 1 / n_cycles


def loading_history(t): return (np.sin(np.pi / T * (t - T / 2)) + 1) / 2


U, F, S = get_UF_t(
    F=lambda t: -70 * loading_history(t),
    n_t=100
)


# **Examples of postprocessing**:
# Plot the axial strain against the lateral strain

# In[15]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 4))
ax1.plot(U[:, 0], U[:, (1, 2)])
ax2.plot(U[:, 0], F[:, 0])
plt.show()

# In[ ]: