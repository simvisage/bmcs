'''
Created on 2 Jan 2020

@author: fseemab
'''
import matplotlib.pyplot as plt
import numpy as np

ax1 = plt.subplot(221)

u_1 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\sandbox\fahad\sigNm0lp0(newdellam).txt')
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
plt.title('m = 0 lp = 0 (SS new del lam)')

ax2 = plt.subplot(222)

u_2 = np.loadtxt(
   r'C:\Users\fseemab\bmcs-master\apps\sandbox\fahad\sigNm01lp0(newdellam).txt')
u = np.transpose(u_2)
ln = np.linspace(0, 80, 5)
t = [639, 1175, 1630, 2034, 2401, 2739, 3054, 3350, 3631, 3898]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax2.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax2.set_xlabel('Embedded Length[mm]')
ax2.set_ylabel('Sigma_N')
ax2.legend()
plt.title('m = 0.1 lp = 0 (SS new del lam)')

ax3 = plt.subplot(223)

u_3 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\sandbox\fahad\sigNm05lp0(newdellam).txt')
u = np.transpose(u_3)
ln = np.linspace(0, 80, 5)
t = [861, 1587, 2183, 2705, 3174, 3603, 4001, 4373, 4724, 5057]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax3.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax3.set_xlabel('Embedded Length[mm]')
ax3.set_ylabel('Sigma_N')
ax3.legend()
plt.title(' m = 0.5 lp = 0 (SS new del lam)')

ax4 = plt.subplot(224)

u_4 = np.loadtxt(
    r'C:\Users\fseemab\bmcs-master\apps\sandbox\fahad\sigNm07lp0(newdellam).txt')
u = np.transpose(u_4)
ln = np.linspace(0, 80, 5)
t = [984, 1822, 2494, 3076, 3597, 4073, 4513, 4923 , 5309, 5674]
colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow', 'orange', 'pink', 'brown', 'grey' ]
timeinstant = np.arange(1, 11) * 0.1
for timi, tstep, color in zip(timeinstant, t, colors):
    ax4.plot(ln, u[:, tstep], color=color, label='time instant = %g' % timi)
ax4.set_xlabel('Embedded Length[mm]')
ax4.set_ylabel('Sigma_N')
ax4.legend()
plt.title(' m = 0.7 lp = 0 (SS new del lam))')

plt.show()
