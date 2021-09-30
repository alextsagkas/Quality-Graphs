import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from scipy import integrate

# Write with LaTeX
rc('text', usetex=True)
rc('font', family='serif')

# Read .csv file
B_s, B_e = [], []
H_s, H_e = [], []
m_s, m_e = [], []

file = pd.read_csv('D2-Measurments.csv', header=None)

B_e, B_s = file[1][10:-11], file[6][12:]
H_e, H_s = file[2][10:-11], file[7][12:]
m_e, m_s = file[3][10:-11], file[8][12:]

m_s[22], m_s[40] = 0, 0
m_e[16], m_e[30] = 0, 0

# Convert to numbers
B_e, B_s = [int(Be) for Be in B_e], [int(Bs) for Bs in B_s]
H_e, H_s = [int(He) for He in H_e], [int(Hs) for Hs in H_s]
m_e, m_s = [float(me) for me in m_e], [float(ms) for ms in m_s]

# Integrals
BB_e, HH_e = [Be*0.001 + 0.950 for Be in B_e], [He + 5050 for He in H_e]

## Plot to see where we integrate 
# plt.plot(HH_e[:14], BB_e[:14], color='black')
# plt.plot(HH_e[13:-1], BB_e[13:-1], color='red')
# plt.show()

Ie1, Ie2 = integrate.simps(y=BB_e[:14], x=HH_e[:14], even='avg'), integrate.simps(y=BB_e[13:-1], x=HH_e[13:-1], even='avg')

Ie = Ie1 + Ie2

BB_s, HH_s = [Bs*0.001 + 0.550 for Bs in B_s], [Hs + 8000 for Hs in H_s]

## Plot to see where we integrate 
# plt.plot(HH_s[:21], BB_s[:21], color='black')
# plt.plot(HH_s[20:], BB_s[20:], color='red')
# plt.show()

Is1, Is2 = integrate.simps(BB_s[:21], HH_s[:21], even='avg'), integrate.simps(BB_s[20:], HH_s[20:], even = 'avg')

Is = Is1+Is2

# Plot
fig1, (ax1, ax2) = plt.subplots(1,2)

# Compact
ax1.plot(H_s, B_s, color='black', linewidth=1, marker='.')
ax1.set_xlabel(r'$H$' r' (A/m)')
ax1.set_ylabel(r'$B$' r' (mT)')
ax1.tick_params(labelsize = 6)
ax1.set_xticks(ticks = np.arange(-8000,8001,2000))
ax1.set_yticks(ticks = np.arange(-600,601,100))
ax1.set_title("Compact Core")
ax1.grid()

# Laminated
ax2.plot(H_e, B_e, color='red', linewidth=1, marker='.')
ax2.set_xlabel(r'$H$' r' (A/m)')
ax2.set_ylabel(r'$B$' r' (mT)')
ax2.tick_params(labelsize = 6)
ax2.set_xticks(ticks = np.arange(-6000,6001,2000))
ax2.set_yticks(ticks = np.arange(-1000,1001,100))
ax2.set_title("Laminated Core")
ax2.grid()

fig1.tight_layout()

# Fix infinity
## Print to see where infinities occur 
# print(m_s)

m_s.pop(10)
H_s.pop(10)
m_s.pop(27)
H_s.pop(27)

# print(m_s, H_s)

m_e.pop(6)
H_e.pop(6)
m_e.pop(19)
H_e.pop(19)

# Plot
fig2, (ax3, ax4) = plt.subplots(1,2)

# Compact
ax3.plot(H_s, m_s, color='black', linewidth=1, marker='.')
ax3.set_xlabel(r'$H$' r' (A/m)')
ax3.set_ylabel(r'$\mu$' r' ($10^{-3}$ N/A$^2$)')
ax3.tick_params(labelsize = 6)
ax3.set_xticks(ticks = np.arange(-8000,8001,2000))
ax3.set_yticks(ticks = np.arange(-0.05,0.31,0.05))
ax3.set_title("Compact Core")
ax3.grid()

# Laminated
ax4.plot(H_e, m_e, color='red', linewidth=1, marker='.')
ax4.set_xlabel(r'$H$' r' (A/m)')
ax4.set_ylabel(r'$\mu$' r' ($10^{-3}$ N/A$^2$)')
ax4.tick_params(labelsize = 6)
ax4.set_xticks(ticks = np.arange(-6000,6001,2000))
ax4.set_yticks(ticks = np.arange(0.17,0.25,0.01))
ax4.set_title("Laminated Core")
ax4.grid()

fig2.tight_layout()

plt.show()

print(f'\nWork on Laminated core = {Ie} J, \nWork on Cmpact Core = {Is} J')
