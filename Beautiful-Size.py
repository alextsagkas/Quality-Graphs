import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit

# Taken from the link in README.md 
def set_size(width, fraction=1, subplots=(1, 1)):

    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 -1 ) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

if __name__ == "__main__":

    # Write with LaTeX
    rc('text', usetex=True)
    rc('font', family='serif')

    # Data
    N = np.array([20, 40, 60, 80, 100])

    Dx = np.array([6.8, 14.3, 20.3, 27.3, 32.3])
    dDx = np.array([0.9, 0.9, 0.9, 0.9, 0.9])

    # Linear Fit
    def func(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(func, xdata=N, ydata=Dx, sigma=1./(dDx**2))
    err = np.sqrt(np.diag(pcov))


    #Plot
    fig, ax = plt.subplots(1, figsize=(set_size(450)))

    ax.errorbar(x=N, y=Dx, yerr=dDx, color='red', capsize=3, elinewidth=1, markeredgewidth=1, linestyle='None', marker='.', label='Experimental Data')
    ax.plot(N, func(N, *popt), color='black', label=f'$\\Delta x = {format(popt[0],".2f")}N + {format(popt[1], ".1f")}$')
    ax.set_ylabel("$\\Delta x$ $(\\mathrm{\\mu m})$")
    ax.set_xlabel("$N$ (Number of Fringes)")
    ax.legend(loc='upper left', prop={'size': 12})
    ax.grid(b=True, which='major', color='#666666', linestyle='--')
    ax.minorticks_on()
    ax.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

    fig.tight_layout()

    plt.show()

    # Print line slope and constant coefficient
    print(f"\na±da ={format(popt[0])}±{err[0]},\n\nb±db={format(popt[1])}±{err[1]}")