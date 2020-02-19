'''
Created on 2 Jan 2020

@author: fseemab
'''
import matplotlib.pyplot as plt
import numpy as np

ax1 = plt.subplot(221)

u_1 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\verify\bond_cum_damage\pullout_axisymmetric_model\sigNm0lp0tan.txt')
u = np.transpose(u_1)
ln = np.linspace(0, 80, 5)
t = [7, 14, 20, 25, 30, 35, 39, 43, 47, 51]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax1.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax1.set_xlabel('Embedded Length[mm]')
ax1.set_ylabel('Sigma_N')
ax1.legend()
plt.title('m = 0 lp = 0 (TS)')

ax2 = plt.subplot(222)

u_2 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\verify\bond_cum_damage\pullout_axisymmetric_model\sigNm01lp0tan.txt')
u = np.transpose(u_2)
ln = np.linspace(0, 80, 5)
t = [8, 16, 23, 29, 35, 41, 46, 51, 56, 61]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax2.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax2.set_xlabel('Embedded Length[mm]')
ax2.set_ylabel('Sigma_N')
ax2.legend()
plt.title('m = 0.1 lp = 0 (TS)')

ax3 = plt.subplot(223)

u_3 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\verify\bond_cum_damage\pullout_axisymmetric_model\sigNm05lp0tan.txt')
u = np.transpose(u_3)
ln = np.linspace(0, 80, 5)
t = [10, 19, 27, 35, 43, 51, 58, 65, 72, 79]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax3.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax3.set_xlabel('Embedded Length[mm]')
ax3.set_ylabel('Sigma_N')
ax3.legend()
plt.title(' m = 0.5 lp = 0 (TS)')

ax4 = plt.subplot(224)

u_4 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\verify\bond_cum_damage\pullout_axisymmetric_model\sigNm1lp0tan.txt')
u = np.transpose(u_4)
ln = np.linspace(0, 80, 5)
t = [13, 25, 36, 47, 57, 67, 76, 85 , 93, 101]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax4.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax4.set_xlabel('Embedded Length[mm]')
ax4.set_ylabel('Sigma_N')
ax4.legend()
plt.title(' m = 1 lp = 0 (TS)')

plt.show()
