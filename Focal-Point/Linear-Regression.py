import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from scipy.optimize import curve_fit

# Write with LaTeX
rc('text', usetex=True)
rc('font', family='serif')

# Read .csv file
file = pd.read_csv('Exp1.csv', header=None)

a = file[0][19:33]
da = file[1][19:33]

b = file[2][19:33]
db = file[3][19:33]

aa = file[4][19:33]
daa = file[5][19:33]

bb = file[6][19:33]
dbb = file[7][19:33]

# Consentrate the Data
A = np.concatenate((a,aa))
dA = np.concatenate((da,daa))

B = np.concatenate((b,bb))
dB = np.concatenate((db,dbb))

# Convert them to numbers
A = np.array([float(i) for i in A])
dA = np.array([float(i) for i in dA])

B = np.array([float(i) for i in B])
dB = np.array([float(i) for i in dB])

# Fit Linear Function to Data
def func(x, a, b):
    return a*x + b

popt, pcov = curve_fit(func, xdata=A, ydata=B)
err = np.sqrt(np.diag(pcov))

# Visualize the Data by Ploting Them with their Errors
fig, ax = plt.subplots(1,1)

ax.errorbar(A, B, xerr=dA, yerr=dB, color='black', capsize=3, elinewidth=1, markeredgewidth=1, linestyle='None', marker='.')
ax.plot(A, func(A, *popt), label=f'$y={format(popt[0], ".2f")}x+{format(popt[1],".3f")}$', color = 'red')
ax.set_ylabel("$1/\\beta$ $(1/\mathrm{cm})$")
ax.set_xlabel("$1/\\alpha$ $(1/\mathrm{cm})$")
ax.legend()

fig.tight_layout()

plt.show()

# Print line slope and constant coefficient
print(f"k ± dk = {popt[0]} ± {err[0]},\n\n m ± dm = {popt[1]} ± {err[1]}")