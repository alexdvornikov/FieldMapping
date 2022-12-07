import numpy as np
import os
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Plot a series of 2D displacement maps along a given scan and projection direction')
parser.add_argument('inputDirs', nargs='+',
                    help="input displacement selection")
parser.add_argument('-p', '--projection',
                    help="Which direction of displacement to plot ('x', 'y', 'z')")
parser.add_argument('-s', '--scan',
                    help="Which axis to scan ('x', 'y', 'z')")
parser.add_argument('-w', '--useWeights',
                    action = 'store_true',
                    help="use axial/radial weighting")
parser.add_argument('-o', '--output',
                    default = 'cathode_crosser_volume_plots',
                    help='output directory')

args = parser.parse_args()


projection = args.projection
scanDir = args.scan

if scanDir == 'x':
    plotAx = ('z', 'y')
elif scanDir == 'y':
    plotAx = ('x', 'z')
elif scanDir == 'z':
    plotAx = ('x', 'y')

unit_vectors = {'x': np.array([1, 0, 0]),
                'y': np.array([0, 1, 0]),
                'z': np.array([0, 0, 1]),
                }

bins = {'x': np.linspace(-300, 300, 31),
        'y': np.linspace(-800, 380, 61),
        'z': np.linspace(-300, 300, 31),
        }

# histkwargs = {'bins': (bins['x'], bins['y'], bins['z'])}
histkwargs = {'bins': [bins['x'], bins['y'], bins['z']]}

if args.inputDirs == None:
    args.inputDirs = ['./cathode_crossers_volumetric']

print("input", args.inputDirs)
infileList = [os.path.join(inputDir, infile)
              for inputDir in args.inputDirs
              for infile in os.listdir(inputDir)
              ]

denom_hist = np.zeros(shape = [len(bins['x'])-1,
                               len(bins['y'])-1,
                               len(bins['z'])-1])
disp_hist = np.zeros(shape = [len(bins['x'])-1,
                              len(bins['y'])-1,
                              len(bins['z'])-1
                              ])
var_hist = np.zeros(shape = [len(bins['x'])-1,
                             len(bins['y'])-1,
                             len(bins['z'])-1
                             ])
occ_hist = np.zeros(shape = [len(bins['x'])-1,
                             len(bins['y'])-1,
                             len(bins['z'])-1
                             ])



REC = []
DIS = []
for path in tqdm(infileList):

    true = np.load(path)[0]
    reco = np.load(path)[1]
    pca_dir = np.load(path)[2]
    if np.any(reco):
        ds = np.diff(np.array([true, reco]) , axis = 0)[0]
        # if args.useWeights:
        #     sigma = np.sqrt(np.power(np.dot(pca_dir,
        #                                     unit_vectors[projection])
        #                              * sigma_axial, 2)
        #                     + np.power(np.linalg.norm(np.cross(pca_dir,
        #                                                        unit_vectors[projection]),
        #                                               axis = -1)
        #                                * sigma_radial, 2))
        # else:
        #     sigma = np.ones(ds.shape[0]) # unweighted (in this case dividing by one for each bin)


        displacement_proj = np.dot(ds, unit_vectors[projection])

        REC.extend( np.array(reco) )
        DIS.extend( np.array(displacement_proj) )

        # denom_hist += np.histogramdd(reco,
        #                              weights = 1./sigma,
        #                              **histkwargs)[0]
        # disp_hist += np.histogramdd(reco,
        #                             weights = displacement_proj/sigma,
        #                             **histkwargs)[0]
        # occ_hist += np.histogramdd(reco,
        #                            **histkwargs)[0]

        # var_hist += np.histogramdd(reco,
        #                            weights = np.power(displacement_proj, 2)/sigma,
        #                            **histkwargs)[0]

# disp_mean = disp_hist/denom_hist
# disp_var = var_hist/denom_hist - np.power(disp_mean, 2)
# disp_std = np.sqrt(disp_var)

# disp_std = np.sqrt((var_hist - np.power(disp_hist,2))/denom_hist)
# disp_std = np.power(occ_hist, -1./2)
# disp_std_stat = np.abs(disp_mean)*np.power(occ_hist, -0.5)

# disp_std_combined = np.sqrt(np.power(disp_std, 2) + 
#                             np.power(disp_std_stat, 2))

# frac_error_combined = disp_std_combined/disp_mean


#---------------------------------------------------------------------------------#
# Alex
from scipy import stats
mean = stats.binned_statistic_dd(np.array(REC), np.array(DIS), statistic='mean', **histkwargs)
std = stats.binned_statistic_dd(np.array(REC), np.array(DIS),
                                 binned_statistic_result=mean,
                                 statistic='std')

means = mean.statistic
variances = std.statistic

np.save('means.npy', means)
np.save('variances.npy', variances)
#---------------------------------------------------------------------------------#

from matplotlib import cm
import matplotlib.pyplot as plt

# imshowKwargs = {"origin": 'lower',
#                 "extent": (np.min(bins[plotAx[0]]),
#                            np.max(bins[plotAx[0]]),
#                            np.min(bins[plotAx[1]]),
#                            np.max(bins[plotAx[1]]))}

# nonNaN = sliceHist[~np.isnan(sliceHist)]
# vext = np.max([np.max(nonNaN), -np.min(nonNaN)])

# imshowKwargs.update({"cmap": 'viridis',
#                         "vmin": vext,
#                         "vmax": -vext})

    
# scanBinCenters = 0.5*(bins[scanDir][1:] + bins[scanDir][:-1])
# for i, binCenter in enumerate(scanBinCenters):
#     fig = plt.figure()
#     if scanDir == 'x':
#         thisSlice = sliceHist[i,:,:]
#     elif scanDir == 'y':
#         thisSlice = sliceHist[:,i,:].T
#     elif scanDir == 'z':
#         thisSlice = sliceHist[:,:,i].T


    # # use this cmap for diverging (zero-centered) quantities
    # plt.imshow(thisSlice, **imshowKwargs)

    # plt.xlabel(plotAx[0]+r' [mm]')
    # plt.ylabel(plotAx[1]+r' [mm]')
    # plt.title(scanDir+r' = '+str(binCenter)+ r' mm')
    # plt.gca().set_aspect('equal')
    # cb = plt.colorbar()

    # # set the title as you need
    # # cb.set_label(r'$\Delta '+projection+r'$ [mm]')
    # # cb.set_label(r'Occupancy')
    # cb.set_label(r'$\sigma (\Delta '+projection+r'$) [mm]')
    # plt.subplots_adjust(left=-0.9, bottom=0.125, right=1, top=0.9)

    # # you might decide to name the output file something meaningful
    # plt.savefig(os.path.join(args.output, quantity+'_slice_'+scanDir+'_'+str(binCenter)+'.png'),
    #             dpi=300)
    # # plt.show()
    # plt.clf()

#---------------------------------------------------------------------------------#
# Alex
#---------------------------------------------------------------------------------#
# Stepping along z in 2cm increments and picking an xy bin (2 by 2 cm)
xind, yind = np.where(bins['x'] == -220), np.where(bins['y'] == -210)

yval = []
yerr = []
scanBinCenters = 0.5*(bins[scanDir][1:] + bins[scanDir][:-1])
for i, binCenter in enumerate(scanBinCenters):
    fig = plt.figure()
    if scanDir == 'z':
        yval.extend( means[xind,yind,i] )
        yerr.extend( variances[xind,yind,i] )

plt.errorbar(scanBinCenters, yval, yerr=yerr, fmt='o-', label='Data')
plt.show()