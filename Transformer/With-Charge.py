import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit

#Write with LaTeX
rc('text', usetex=True)
rc('font', family='serif')

# Data and Errors
I_1 = np.array([0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.23, 0.23])
dI_1 = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

I_2 = np.array([0.873, 0.960, 1.108 ,1.293, 1.468, 1.720, 1.941, 1.967])
dI_2 = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])

V_2 = np.array([4.85, 4.83, 4.76, 4.74, 4.70, 4.67, 4.66, 4.65])
dV_2 = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

cos_1 = np.array([0.855, 0.849, 0.844, 0.808, 0.874, 0.941, 0.925, 0.958])
dcos_1 = np.array([0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05])

a = np.array([0.47, 0.488, 0.527, 0.607, 0.6, 0.618 ,0.646, 0.631])
da = np.array([0.026, 0.026, 0.026, 0.030, 0.026, 0.024, 0.023, 0.022])

# Function to Fit Data on
def func1(x, a, b):
    return (a * x) + b

def func2(x, a, b, c):
    return (a * x**2) + (b * x) + c

def func3(x, a, b, c, d):
    return (a * x**3) + (b * x**2) + (c * x) + d

def func4(x, a, b, c, d, e):
    return (a * x**4) + (b * x**3) + (c * x**2) + (d * x) + e

# Fitting
x = np.linspace(0.873, 1.967, 100)
X = np.linspace(0, 0.96, 40)

# Fit Data and Take Account Errors too
popt_I1I2, pcov_I1I2 = curve_fit(func1, I_2, I_1, sigma=1./(dI_1*dI_1))

popt_I2V2, pcov_I2V2 = curve_fit(func4, I_2, V_2, sigma=1./(dV_2*dV_2))

popt_phi1I2, pcov_Phi1I2 = curve_fit(func3, I_2, cos_1, sigma=1./(dcos_1*dcos_1))

popt_aI2, pcov_aI2 = curve_fit(func3, I_2, a, sigma=1./(da*da))

# Plot Diagrams
fig, (ax_I1I2, ax_V2I2, ax_phi1I2, ax_aI2) = plt.subplots(4, 1)
fig.set_size_inches(w=6.2, h=9.2)

# I2 = I2(I1)

ax_I1I2.errorbar(I_2, I_1, xerr = dI_2, yerr = dI_1, color='black', capsize=3, elinewidth=1, markeredgewidth=1, linestyle='None', marker='.', label='Experimental \n Values of $I_1$')
ax_I1I2.plot(x, func1(x, *popt_I1I2), color='red', label='$I_1 = I_1(I_2)$', linewidth=1.5)

ax_I1I2.set_ylabel('$I_1$ $(\mathrm{A})$')
ax_I1I2.set_xlabel('$I_2$ $(\mathrm{A})$')
ax_I1I2.legend(loc = 'lower right', prop={'size': 11})

# Show the major grid lines with dark grey lines
ax_I1I2.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines
ax_I1I2.minorticks_on()
ax_I1I2.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)


# V2 = V2(I2)

ax_V2I2.errorbar(I_2, V_2, xerr= dI_2, yerr = dV_2, color='black', capsize=3, elinewidth=1, markeredgewidth=1, linestyle='None', marker='.', label='Experimental \n Values of $V_2$')
ax_V2I2.plot(x, func4(x, *popt_I2V2), color='green', label='$V_2 = V_2(I_2)$', linewidth=1.5)

ax_V2I2.set_ylabel('$V_2$ $(\mathrm{V})$')
ax_V2I2.set_xlabel('$I_2$ $(\mathrm{A})$')
ax_V2I2.legend(loc = 'upper right', prop={'size': 11})

# Show the major grid lines with dark grey lines
ax_V2I2.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines
ax_V2I2.minorticks_on()
ax_V2I2.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)


# cos(phi1) = cos(phi1)(I2)

ax_phi1I2.errorbar(I_2, cos_1, xerr=dI_2, yerr=dcos_1, color='black', capsize=3, elinewidth=1, markeredgewidth=1, linestyle='None', marker='.', label='Calculated \n Values of $\cos\phi_1$')
ax_phi1I2.plot(x, func3(x, *popt_phi1I2), color='blue', label='$\cos\phi_1 = \cos\phi_1(I_2)$', linewidth=1.5)

ax_phi1I2.set_ylabel('$\cos\phi_1$')
ax_phi1I2.set_xlabel('$I_2$ $(\mathrm{A})$')
ax_phi1I2.legend(loc = 'lower right', prop={'size': 11})

# Show the major grid lines with dark grey lines
ax_phi1I2.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines
ax_phi1I2.minorticks_on()
ax_phi1I2.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)


# a = a(I2)

ax_aI2.errorbar(I_2, a, xerr = dI_2, yerr=da, color='black', capsize=3, elinewidth=1, markeredgewidth=1, linestyle='None', marker='.', label='Calculated \n Values of $a$')
ax_aI2.plot(x, func3(x, *popt_aI2), color='orange', label='$a = a(I_2)$', linewidth=1.5)

ax_aI2.set_ylabel('$a$')
ax_aI2.set_xlabel('$I_2$ $(\mathrm{A})$')
ax_aI2.legend(loc = 'lower right', prop={'size': 11})

# Show the major grid lines with dark grey lines
ax_aI2.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines
ax_aI2.minorticks_on()
ax_aI2.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

# fix quality
fig.tight_layout()

plt.show()