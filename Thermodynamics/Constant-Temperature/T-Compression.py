import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from scipy import integrate
from scipy.optimize import curve_fit

# Write with LaTeX
rc('text', usetex=True)
rc('font', family='serif')

#Read .csv file
file1 = pd.read_csv('T_Compression-1.csv', header=None)
file2 = pd.read_csv('T_Compression-2.csv', header=None)

# Temperatures
t1 = file1[16][1:]
t1 = np.array([float(i) for i in t1])
t2 = file2[16][1:]
t2 = np.array([float(i) for i in t2])

# Volumes
v1 = file1[11][1:]
v1 = np.array([float(i)*0.1 for i in v1])
v2 = file2[11][1:]
v2 = np.array([float(i)*0.1 for i in v2])

# Pressure
p1 = file1[8][1:]
p1 = np.array([float(i)*10**3 for i in p1])
p2 = file2[8][1:]
p2 = np.array([float(i)*10**3 for i in p2])

# Logarithmic Axis
logp1 = np.log(p1)
logp2 = np.log(p2)

logv1 = np.log(v1)
logv2 = np.log(v2)


# Fitting
# The function we use to fit our data on
def func(x, a, b):
    return (a * x) + b

# popt1 = (a1,b1)
popt1, pcov1 = curve_fit(func, logv1, logp1)
# err1 = (δa1,δb1)
err1 = np.sqrt(np.diag(pcov1))

# popt2 = (a,b)
popt2, pcov2 = curve_fit(func, logv2, logp2)
# err2 = (δa2,δb2)
err2 = np.sqrt(np.diag(pcov2))

# Integrals (=work done)
I1 = integrate.trapz(p1, v1)
I2 = integrate.trapz(p2, v2)

# Plots
fig, ax = plt.subplots(2,2)

p1 = p1*0.001
v1 = v1*10**4
p2 = p2*0.001
v2 = v2*10**4

ax[0,0].scatter(logv1, logp1, color='black', marker=".")
ax[0,0].plot(logv1, func(logv1, *popt1), color = 'darkorange', label="$y=\\alpha_1 x + \\beta_1$")
ax[0,0].set_ylabel("$\\log P_1$")
ax[0,0].set_xlabel("$\\log V_1$")
ax[0,0].legend()
ax[0][0].set_title("First Compression")
ax[0,0].grid(b=True, which='major', color='#666666', linestyle='--')
ax[0,0].minorticks_on()
ax[0,0].grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

ax[1,0].plot(v1, p1, color='seagreen')
ax[1,0].set_ylim((0,130))
ax[1,0].set_ylabel("$P_1$ $(\\mathrm{kPa})$")
ax[1,0].set_xlabel("$V_1$ $(10^{-4}\,\\mathrm{m}^3)$")
ax[1,0].set_title("First Compression")
ax[1,0].fill_between(v1, p1, color='seagreen', label="$W_1$", alpha=0.6)
ax[1,0].legend()
ax[1,0].grid(b=True, which='major', color='#666666', linestyle='--')
ax[1,0].minorticks_on()
ax[1,0].grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

ax[0,1].scatter(logv2, logp2, color='black', marker=".")
ax[0,1].plot(logv2, func(logv2, *popt2), color = 'darkorange', label="$y=\\alpha_2 x + \\beta_2$")
ax[0,1].set_ylabel("$\\log P_2$")
ax[0,1].set_xlabel("$\\log V_2$")
ax[0,1].legend()
ax[0,1].set_title("Second Compression")
ax[0,1].grid(b=True, which='major', color='#666666', linestyle='--')
ax[0,1].minorticks_on()
ax[0,1].grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

ax[1,1].plot(v2, p2, color='seagreen')
ax[1,1].set_ylim((0,130))
ax[1,1].set_ylabel("$P_2$ $(\\mathrm{kPa})$")
ax[1,1].set_xlabel("$V_2$ $(10^{-4}\,\\mathrm{m}^3)$")
ax[1,1].set_title("Second Compression")
ax[1,1].fill_between(v2, p2, color='seagreen', label="$W_2$", alpha=0.6)
ax[1,1].legend(loc="upper right")
ax[1,1].grid(b=True, which='major', color='#666666', linestyle='--')
ax[1,1].minorticks_on()
ax[1,1].grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

fig.tight_layout()

plt.show()

# Print lines' slopes and constant terms
print(f"a_1 ± δa_1 = {popt1[0]} ± {err1[0]}, b_1 ± δb_1 = {popt1[1]} ± {err1[1]} \n")
print(f"a_2 ± δa_2 = {popt2[0]} ± {err2[0]}, b_2 ± δb_2 = {popt2[1]} ± {err2[1]} \n")
# Print mean temperature (=const)
print(f"T_1 = {np.mean(t1)} K, T_2 = {np.mean(t2)} K \n")
# Print work done
print(f"I_1 = {I1} J, I_2 = {I2} J")
