'''
Created on 30 Dec 2019

@author: fseemab
'''
import matplotlib.pyplot as plt
import numpy as np

ax1 = plt.subplot(221)

u_1 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\verify\bond_cum_damage\pullout_axisymmetric_model\sigNm000lp-20000.txt')
u = np.transpose(u_1)
ln = np.linspace(0, 80, 5)
t = [591, 1086, 1507, 1882, 2223, 2538, 2832, 3109, 3371, 3620]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax1.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax1.set_xlabel('Embedded Length[mm]')
ax1.set_ylabel('Sigma_N')
ax1.legend()
plt.title('m = 0 lp = -20,000')

ax2 = plt.subplot(222)

u_2 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\verify\bond_cum_damage\pullout_axisymmetric_model\sigNm01lp-20000.txt')
u = np.transpose(u_2)
ln = np.linspace(0, 80, 5)
t = [625, 1163, 1623, 2032, 2404, 2747, 3067, 3368, 3653, 3924]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax2.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax2.set_xlabel('Embedded Length[mm]')
ax2.set_ylabel('Sigma_N')
ax2.legend()
plt.title('m = 0.1 lp = -20,000')

ax3 = plt.subplot(223)

u_3 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\verify\bond_cum_damage\pullout_axisymmetric_model\sigNm05lp-20000.txt')
u = np.transpose(u_3)
ln = np.linspace(0, 80, 5)
t = [728, 1391, 1968, 2484, 2954, 3388, 3792, 4172, 4531, 4873]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax3.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax3.set_xlabel('Embedded Length[mm]')
ax3.set_ylabel('Sigma_N')
ax3.legend()
plt.title(' m = 0.5 lp = -20,000')

ax4 = plt.subplot(224)

u_4 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\verify\bond_cum_damage\pullout_axisymmetric_model\sigNm1lp-20000.txt')
u = np.transpose(u_4)
ln = np.linspace(0, 80, 5)
t = [826, 1608, 2296, 2948, 3544, 4086, 4589, 5060, 5585, 5927]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax4.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax4.set_xlabel('Embedded Length[mm]')
ax4.set_ylabel('Sigma_N')
ax4.legend()
plt.title(' m = 1 lp = -20,000')

plt.show()
