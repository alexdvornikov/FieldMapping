import numpy as np
import os
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

# Pretty fonts for figures (if have LaTex enabled)
mpl.rc('text', usetex = True)
mpl.rc('font', family='SignPainter')

# Stepping along z in 2cm increments and picking an xy bin (2 by 2 cm)
means = np.load('means.npy')
variances = np.load('variances.npy')

bins = {'x': np.linspace(-300, 300, 31),
        'y': np.linspace(-800, 380, 61),
        'z': np.linspace(-300, 300, 31),
        }


xbins = 0.5*(bins['x'][1:] + bins['x'][:-1])
ybins = 0.5*(bins['y'][1:] + bins['y'][:-1])
zbins = 0.5*(bins['z'][1:] + bins['z'][:-1])

# Set x or y bin if interested in certain vertical or hozintal lines along the anode
reference_xbins = [-250, -50, 50, 250]
fs = 25
for k,xbin in enumerate(xbins):
    for ybin in ybins:
        if (195 < ybin < 215):
            if xbin in reference_xbins:

                yval = []
                yerr = []

                xind, yind = np.where(xbins == xbin), np.where(ybins == ybin)
                for i, binCenter in enumerate(zbins):
                    yval.extend( means[xind,yind,i] )
                    yerr.extend( variances[xind,yind,i] )

                xval = np.array( zbins )
                yval = np.array( yval )
                error = np.array( yerr )

                yval = yval.flatten()
                error = error.flatten()


                plt.errorbar(xval/10, yval/10, yerr=error/10, fmt='--.', capsize=4, elinewidth=1,
                label= 'x = ' + str(round(xbin/10)) + ' cm'  )
                # plt.fill_between(xval/10, (yval - error/2)/10, (yval + error/2)/10, alpha=0.1)
                plt.title('y = ' + str(round(ybin/10)) + ' cm', fontsize=fs)
                # plt.plot(xval/10, yval/10, label= 'x = ' + str(round(xbin/10)) + ' cm' )
                plt.legend(fontsize=fs, frameon=False, loc='upper right')
                plt.xlabel('Drift Direction [cm]', fontsize=fs)
                plt.ylabel(r'$\Delta x'r'$ [cm]', fontsize = fs)
                plt.xticks(fontsize=fs)
                plt.yticks(fontsize=fs)

plt.ylim(-2, 2)
plt.show()