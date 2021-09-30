import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

#Write with LaTeX
rc('text', usetex=True)
rc('font', family='serif')

def func(x, a, b):
    return (a * x) + b

# Data
B1 = np.array([9.38, 12.46, 15.57])
dB1 = np.array([0.04, 0.04, 0.04])
r1 = np.array([0.217, 0.28, 0.38])
dr1 = np.array([0.024, 0.04, 0.07])

B2 = np.array([9.38, 12.46, 15.57])
dB2 = np.array([0.04, 0.04, 0.04])
r2 = np.array([0.2, 0.2500, 0.33])
dr2 = np.array([0.02, 0.03, 0.06])

# Fitting
x = np.linspace(0.15, 0.4, 5)

popt1, pcov1 = curve_fit(func, r1, B1, sigma=1./(dB1*dB1))
perr1 = np.sqrt(np.diag(pcov1))
popt2, pcov2 = curve_fit(func, r2, B2, sigma=1./(dB2*dB2))
perr2 = np.sqrt(np.diag(pcov2))

# Plot
fig, ax = plt.subplots(1, 1)

# B1 = B1(1/r1)
ax.errorbar(r1, B1, xerr = dr1, yerr = dB1, capsize=3, color='black', elinewidth=1, markeredgewidth=1, linestyle='None', marker='o', label='Calculated \n Values of $B_1$')
ax.plot(x, func(x, *popt1), color='orange', label='$B1 = B1(1/r_1)$', linewidth=1.5)

# B2 = B2(1/r2)
ax.errorbar(r2, B2, xerr = dr2, yerr = dB2, capsize=3, color='black', elinewidth=1, markeredgewidth=1, linestyle='None', marker='s', label='Calculated \n Values of $B_2$')
ax.plot(x, func(x, *popt2), color='royalblue', label='$B2 = B2(1/r_2)$', linewidth=1.5)

# Figure Specifications
ax.set_ylabel('$B$ $(\mathrm{10^{-4}\,\mathrm{T}})$')
ax.set_xlabel('$1/r$ $(\mathrm{1/\mathrm{cm}})$')

ax.legend(loc = 'upper left', prop={'size': 11})

# Show the major grid lines with dark grey lines
ax.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines
ax.minorticks_on()
ax.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

# fix quality
fig.tight_layout()

plt.show()

# Print lines' slopes and constant coefficients
print(f"\n\n a1 = {'%0.5f'%popt1[0]} ± {'%0.5f'%perr1[0]}", f",b1 = {'%0.5f'%popt1[1]} ± {'%0.5f'%perr1[1]}")
print(f"\n\n a2 = {'%0.5f'%popt2[0]} ± {'%0.5f'%perr2[0]}", f",b2 = {'%0.5f'%popt2[1]} ± {'%0.5f'%perr2[1]}")
