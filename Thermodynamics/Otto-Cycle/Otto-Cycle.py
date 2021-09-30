import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from scipy import integrate
from scipy.optimize import curve_fit

# Write with LaTeX
rc('text', usetex=True)
rc('font', family='serif')

#Read .csv file (the .csv files are produced with the Logger Pro Software in Lab)
file1 = pd.read_csv('Otto-1.csv', header=None)
file2 = pd.read_csv('Otto-2.csv', header=None)

v1 = file1[11][1:]
v1 = np.array([float(i)*0.1 for i in v1])
v2 = file2[11][1:]
v2 = np.array([float(i)*0.1 for i in v2])

p1 = file1[8][1:]
p1 = np.array([float(i)*10**3 for i in p1])
p2 = file2[8][1:]
p2 = np.array([float(i)*10**3 for i in p2])

# Integrals
W43_1 = integrate.simps(p1[180:186], v1[180:186], even="avg")
W21_1 = integrate.simps(p1[567:573], v1[567:573], even="avg")
W_1 = integrate.trapz(p1, v1)

W43_2 = integrate.simps(p2[145:153], v2[145:153], even='avg')
W21_2 = integrate.simps(p2[495:502], v2[495:502], even='avg')
W_2 = integrate.trapz(p2, v2)

# log Axis
logv43_1 = np.log(v1[180:186])
logp43_1 = np.log(p1[180:186])
logv21_1 = np.log(v1[567:573])
logp21_1 = np.log(p1[567:573])

logv43_2 = np.log(v2[145:153])
logp43_2 = np.log(p2[145:153])
logv21_2 = np.log(v2[495:502])
logp21_2 = np.log(p2[495:502])

# Fitting
def func(x, a, b):
    return (a * x) + b

popt43_1, pcov43_1 = curve_fit(func, logv43_1, logp43_1)
err43_1 = np.sqrt(np.diag(pcov43_1))
popt21_1, pcov21_1 = curve_fit(func, logv21_1, logp21_1)
err21_1 = np.sqrt(np.diag(pcov21_1))

popt43_2, pcov43_2 = curve_fit(func, logv43_2, logp43_2)
err43_2 = np.sqrt(np.diag(pcov43_2))
popt21_2, pcov21_2 = curve_fit(func, logv21_2, logp21_2)
err21_2 = np.sqrt(np.diag(pcov21_2))

# Fill Between
def f(x, a, b):
    return (a / (b+x))

popt1, pcov1 = curve_fit(f, v1[567:573], p1[567:573])
popt11, pcov11 = curve_fit(f, v1[180:186], p1[180:186])

popt2, pcov2 = curve_fit(f, v2[495:502], p2[495:502])
popt22, pcov22 = curve_fit(f, v2[145:153], p2[145:153])

x = np.linspace(0.00097, 0.00036, 100)

# P - V
fig1, ax1 = plt.subplots(2,1)

ax1[0].fill_between(x, f(x, *popt1), f(x, *popt11), color='royalblue', label="$W_1$", alpha=0.7)
ax1[0].scatter(v1, p1, color='black', marker=".")
ax1[0].plot(v1, p1, color = 'royalblue', lw=2)
ax1[0].set_ylabel("$P_1$ $(\\mathrm{kPa})$")
ax1[0].set_xlabel("$V_1$ $(10^{-4}\,\\mathrm{m^3})$")
ax1[0].set_title("First Cycle")
ax1[0].legend()
ax1[0].grid(b=True, which='major', color='#666666', linestyle='--')
ax1[0].minorticks_on()
ax1[0].grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

ax1[1].fill_between(x, f(x, *popt1), f(x, *popt11), color='darkorange', label="$W_2$", alpha=0.7)
ax1[1].scatter(v2, p2, color='black', marker=".", lw=2)
ax1[1].plot(v2, p2, color = 'darkorange')
ax1[1].set_ylabel("$P_2$ $(\\mathrm{kPa})$")
ax1[1].set_title("Second Cycle")
ax1[1].set_xlabel("$V_2$ $(10^{-4}\,\\mathrm{m^3})$")
ax1[1].legend()
ax1[1].grid(b=True, which='major', color='#666666', linestyle='--')
ax1[1].minorticks_on()
ax1[1].grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

fig1.tight_layout()

# logP - logV
fig2, ax2 = plt.subplots(2,2)

ax2[0][0].scatter(logv43_1, logp43_1, color='black', marker=".")
ax2[0][0].plot(logv43_1, func(logv43_1, *popt43_1), color = 'dodgerblue', label=f'$y = m_1x + c_1$')
ax2[0][0].set_ylabel("$\\log P_1$")
ax2[0][0].set_xlabel("$\\log V_1$")
ax2[0][0].legend()
ax2[0][0].set_title("First Adiabatic Compression")
ax2[0][0].grid(b=True, which='major', color='#666666', linestyle='--')
ax2[0][0].minorticks_on()
ax2[0][0].grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

ax2[0][1].scatter(logv21_1, logp21_1, color='black', marker=".")
ax2[0][1].plot(logv21_1, func(logv21_1, *popt21_1), color = 'gold', label=f'$y = M_1x + C_1$')
ax2[0][1].set_ylabel("$\\log P_1$")
ax2[0][1].set_xlabel("$\\log V_1$")
ax2[0][1].legend()
ax2[0][1].set_title("First Adiabatic Relaxation")
ax2[0][1].grid(b=True, which='major', color='#666666', linestyle='--')
ax2[0][1].minorticks_on()
ax2[0][1].grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

ax2[1][0].scatter(logv43_2, logp43_2, color='black', marker=".")
ax2[1][0].plot(logv43_2, func(logv43_2, *popt43_2), color = 'red', label=f'$y = m_2x + c_2$')
ax2[1][0].set_ylabel("$\\log P_2$")
ax2[1][0].set_xlabel("$\\log V_2$")
ax2[1][0].legend()
ax2[1][0].set_title("Second Adiabatic Compression")
ax2[1][0].grid(b=True, which='major', color='#666666', linestyle='--')
ax2[1][0].minorticks_on()
ax2[1][0].grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

ax2[1][1].scatter(logv21_2, logp21_2, color='black', marker=".")
ax2[1][1].plot(logv21_2, func(logv21_2, *popt21_2), color = 'blueviolet', label=f'$y = M_2x + C_2$')
ax2[1][1].set_ylabel("$\\log P_2$")
ax2[1][1].set_xlabel("$\\log V_2$")
ax2[1][1].legend()
ax2[1][1].set_title("Second Adiabatic Relaxation")
ax2[1][1].grid(b=True, which='major', color='#666666', linestyle='--')
ax2[1][1].minorticks_on()
ax2[1][1].grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

fig2.tight_layout()

plt.show()

# Print the Work Done 
print(f"W43_1 = {W43_1} J, W21_1={W21_1} J, W_1 = {W43_1+W21_1} J =? {W_1} J \n")
print(f"W43_2 = {W43_2} J, W21_2={W21_2} J, W_2 = {W43_2+W21_2} J =? {W_2} J \n\n")

# Print the Lines' Slope
print(f"a43_1 ± δa43_1 = {popt43_1[0]} ± {err43_1[0]}, a21_1 ± δa21_1 = {popt21_1[0]} ± {err21_1[0]}\n")
print(f"a43_2 ± δa43_2 = {popt43_2[0]} ± {err43_2[0]}, a21_1 ± δa21_1 = {popt21_2[0]} ± {err21_2[0]}\n")
