'''
Created on 15.03.2018

@author: abaktheer
'''
import matplotlib.pyplot as plt
import numpy as np


#=========================================================================
# numerical results (creep fatigue) (L, H_L )
#=========================================================================
ax1 = plt.subplot(221)


u_1, f_1 = np.loadtxt(
    r'E:\Publishing\Conferences\FIB_2018_(2)\results\exp\monotonic_16_active.txt')


u_2, f_2 = np.loadtxt(
    r'E:\Publishing\Conferences\FIB_2018_(2)\results\exp\monotonic_16_passiv.txt')

u_3, f_3 = np.loadtxt(
    r'E:\Publishing\Conferences\FIB_2018_(2)\results\exp\monotonic_16_passiv_2.txt')

ax1.plot(f_1, u_1, "k", label='loaded')
ax1.plot(f_2, u_2, "--k", label='unloaded')
ax1.plot(f_3, u_3, "--b", label='unloaded')


u_2 = np.loadtxt(
    r'E:\Models_Implementation\python_results\U_loaded_slip2.txt')
f_2 = np.loadtxt(
    r'E:\Models_Implementation\python_results\F2.txt')


ax1.plot(u_2, f_2, linestyle='-', linewidth=2, color='g')


# ax1.fill_between(
#     n_1[1:-1], -s_1[1:-1], -s_2[1:-1], facecolor='gray', alpha=0.2)


ax1.set_xlabel('slip[mm]')
ax1.set_ylabel('force[kN]')
#ax1.set_xlim(-0.2, 0.5)
#ax1.set_ylim(1.75, 3.0)
ax1.legend(loc=2)


#=========================================================================
# exp results (LS2)
#=========================================================================
ax2 = plt.subplot(223)


u_1, f_1 = np.loadtxt(
    r'E:\Publishing\Conferences\FIB_2018_(2)\results\exp\LS2_F_w_new.txt')

ax2.plot(f_1, u_1, "k", linewidth=2,)

ax2.set_xlim(0, 0.5)

# ax1.fill_between(
#     n_1[1:-1], -s_1[1:-1], -s_2[1:-1], facecolor='gray', alpha=0.2)


#=========================================================================
# numerical results (LS2 )
#=========================================================================

ax3 = plt.subplot(224)

u_2 = np.loadtxt(
    r'E:\Models_Implementation\python_results\U_loaded_slip2.txt')
f_2 = np.loadtxt(
    r'E:\Models_Implementation\python_results\F2.txt')

#ax3.plot(u_2, f_2, linestyle='-', marker='o', color='b')
ax2.plot(u_2, f_2, linestyle='-', linewidth=2, color='g')
ax3.plot(u_2, f_2, linestyle='-', linewidth=2, color='g')
ax3.set_xlim(0, 0.5)

#
# #=========================================================================
# # fatigue creep curves (LS2 )
# #=========================================================================
#
# ax4 = plt.subplot(222)
#
# # exp
# n_max = np.loadtxt(
#     r'E:\Publishing\Conferences\FIB_2018_(2)\results\exp\LS2_N_max.txt')
# s_max = np.loadtxt(
#     r'E:\Publishing\Conferences\FIB_2018_(2)\results\exp\LS2_S_max.txt')
# n_min = np.loadtxt(
#     r'E:\Publishing\Conferences\FIB_2018_(2)\results\exp\LS2_N_min.txt')
# s_min = np.loadtxt(
#     r'E:\Publishing\Conferences\FIB_2018_(2)\results\exp\LS2_S_min.txt')
#
# # num
# N_max = np.loadtxt(
#     r'E:\Models_Implementation\python_results\n2.txt')
# S_max = np.loadtxt(
#     r'E:\Models_Implementation\python_results\s_loaded2.txt')
# N_min = np.loadtxt(
#     r'E:\Models_Implementation\python_results\n2.txt')
# S_min = np.loadtxt(
#     r'E:\Models_Implementation\python_results\s_min_loaded.txt')
#
# # F_max = np.loadtxt(
# #     r'E:\Models_Implementation\python_results\F_max_unloaded.txt')
# # F_min = np.loadtxt(
# #     r'E:\Models_Implementation\python_results\F_min_unloaded.txt')
#
# ax4.plot(n_max[1:] + 0.5, s_max[1:], linestyle='-', linewidth=2, color='k')
# ax4.plot(n_min[1:] + 0.5, s_min[1:], linestyle='-', linewidth=2, color='k')
# ax4.plot(N_max[1:] + 0.5, S_max[1:], linestyle='-', linewidth=2, color='r')
# ax4.plot(N_min[1:] + 0.5, S_min[1:], linestyle='-', linewidth=2, color='b')
# #ax4.plot(N_min + 0.5, F_max, linestyle='-', linewidth=2, color='r')
# #ax4.plot(N_min + 0.5, F_min, linestyle='-', linewidth=2, color='r')
# ax4.set_ylim(-0.0, 0.4)

plt.show()
