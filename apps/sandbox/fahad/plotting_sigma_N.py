'''
Created on 30 Dec 2019

@author: fseemab
'''
import matplotlib.pyplot as plt
import numpy as np

ax1 = plt.subplot(221)

u_1 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\verify\bond_cum_damage\pullout_axisymmetric_model\sigNm0.txt')
u = np.transpose(u_1)
ln = np.linspace(0, 80, 5)
t = [590, 1083, 1504, 1879, 2220, 2535, 2829, 3106, 3368, 3618]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax1.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax1.set_xlabel('Embedded Length[mm]')
ax1.set_ylabel('Sigma_N')
ax1.legend()
plt.title('m = 0')

ax2 = plt.subplot(222)

u_2 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\verify\bond_cum_damage\pullout_axisymmetric_model\sigNm01.txt')
u = np.transpose(u_2)
ln = np.linspace(0, 80, 5)
t = [622, 1150, 1600, 2000, 2364, 2699, 3095, 3306, 3584, 3849]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax2.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax2.set_xlabel('Embedded Length[mm]')
ax2.set_ylabel('Sigma_N')
ax2.legend()
plt.title(' m = 0.1')

ax3 = plt.subplot(223)

u_3 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\verify\bond_cum_damage\pullout_axisymmetric_model\sigNm05.txt')
u = np.transpose(u_3)
ln = np.linspace(0, 80, 5)
t = [736, 1383, 1937, 2428, 2873, 3282, 3663, 4020, 4357, 4677]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax3.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax3.set_xlabel('Embedded Length[mm]')
ax3.set_ylabel('Sigma_N')
ax3.legend()
plt.title('m = 0.5')

ax4 = plt.subplot(224)

u_4 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\verify\bond_cum_damage\pullout_axisymmetric_model\sigNm1.txt')
u = np.transpose(u_4)
ln = np.linspace(0, 80, 5)
t = [873, 1658, 2332, 2928, 3467, 3961, 4420, 4849, 5253, 5636]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax4.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax4.set_xlabel('Embedded Length[mm]')
ax4.set_ylabel('Sigma_N')
ax4.legend()
plt.title(' m = 1')

plt.show()
