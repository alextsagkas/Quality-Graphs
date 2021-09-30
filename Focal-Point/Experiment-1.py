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

a = file[1][1:16]
a = np.array([float(i) for i in a])
da = np.array([0.05 for i in range(15)])

b = file[2][1:16]
b = np.array([float(i) for i in b])
db = np.array([0.05 for i in range(15)])

aa = file[3][1:16]
aa = np.array([float(i) for i in aa])
daa = np.array([0.05 for i in range(15)])

bb = file[4][1:16]
bb = np.array([float(i) for i in bb])
dbb = np.array([0.05 for i in range(15)])

# Consentrate Measurements
y = np.concatenate((a,aa)) + np.concatenate((b,bb))
x = np.concatenate((a,aa))

# Compute Errors with error propagation rule
dy = np.sqrt(2) * np.concatenate((da,daa))
dx = np.concatenate((da,daa))

# Fit to Data
# Functions to fit
def func(x, a, b):
    return x**2/(a * x + b)

def f(x, a, b):
    return a * x + b

def F(y, a, b):
    return a * y + b

# Use not only the data but also its errors
popt, pcov = curve_fit(func, xdata=x, ydata=y, sigma=1./(dy*dy))
err = np.sqrt(np.diag(pcov))

popt1, pcov1 = curve_fit(F, xdata=y[:3], ydata=x[:3], sigma=1./(dx[:3]*dx[:3]))
err1 = np.sqrt(np.diag(pcov1))

popt2, pcov2 = curve_fit(f, xdata=x[15:-2], ydata=y[15:-2], sigma=1./(dy[15:-2]*dy[15:-2]))
err2 = np.sqrt(np.diag(pcov2))

# Plot to Visualize
fig, ax = plt.subplots(1,1)

xx = np.linspace(17.78, 111.6, 100)
yy1 = np.linspace(60, 130, 20)
xx2 = np.linspace(25, 111.5, 20)

ax.errorbar(x, y, xerr=dx, yerr=dy, color='black', capsize=3, elinewidth=1, markeredgewidth=1, linestyle='None', marker='.')
ax.plot(xx, func(xx, *popt), label='$y=\\frac{x^2}{(%1.1f x  %1.2f)}$' % (popt[0], popt[1]), color = 'blue')
ax.plot(F(yy1, *popt1), yy1, label='$x=%.2f y + %.2f$' % (popt1[0], popt1[1]), color = 'green')
ax.plot(xx2, f(xx2, *popt2), label='$y=%.2f x + %.2f$' % (popt2[0], popt2[1]), color = 'magenta')
ax.set_xlabel("$\\alpha$ $(\mathrm{cm})$")
ax.set_ylabel("$\\alpha+\\beta$ $(\mathrm{cm})$")
ax.legend()

fig.tight_layout()

plt.show()

# Print line slopes and constant coefficients
print(f" a ± da = {popt[0]} ± {err[0]},\n\n b ± db = {popt[1]} ± {err[1]}\n\n\n")
print(f" a ± da = {popt1[0]} ± {err1[0]},\n\n b ± db = {popt1[1]} ± {err1[1]}\n\n\n")
print(f" a ± da = {popt2[0]} ± {err2[0]},\n\n b ± db = {popt2[1]} ± {err2[1]}")
