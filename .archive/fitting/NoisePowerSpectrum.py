# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %%
# %%javascript
# IPython.OutputArea.prototype._should_scroll = function(lines) {
#     return false;
# }

# %%
# import packages
import os

import numpy as np
from numpy import log as ln
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator

from NPSmethods import *


# %% pre-settings

imgSysData = {
    "CCDPixelSize": 13,  # pixel size of the CCD, in micron 
    "magnification":
    27,  # 799.943 / 29.9099, # magnification of the imaging system 
    "wavelen": 0.852,  # wavelength of the imaging beam, in micron 
    "NA": 0.37,  # numerical aperture of the objective 
    "ODtoAtom": 13
}

choices = {
    "if_Save": True,
    "do_Fit": True,
    "plot_2dGas": True,
    "plot_NoisePowSpec": True,
    "plot_NoisePowSpec_LineCut": True,
    "plot_Pupil": True,
    "plot_PSF": True,
    "plot_PSF_LineCut": True,
    "plot_PSF_abs2": True,
    "plot_PSF_abs2_LineCut": True,
    "plot_Sk": True,
    "normalize": False
}

from os.path import dirname, join
from os import getcwd

cwd = getcwd()

baseDir = dirname(cwd)

DATA_DIR = join(baseDir, 'DATA')

DATASET_ID = '221529'
# the id of dataset where img are stored.

imgDir = join(DATA_DIR, DATASET_ID)
# the directory where the images are stored

resDir = join(cwd, 'results')
# the directory where the results should be stored

trapRegion = (slice(100, 200), slice(100, 200))
# the region where the atoms located, [(xmin, xmax), (ymin, ymax)] (pixel)

noiseRegion = (slice(0, 300), slice(0, 300))
# the region chosen for noise analysis, [(xmin, xmax), (ymin, ymax)] (pixel)

numOfImgsInEachRun = 50  # number of images for each run of analysis

rampingParameter = 0

# %%

showExampleImg(imgDir,
               numOfImgsInEachRun,
               rampingParameter,
               trapRegion=trapRegion,
               noiseRegion=noiseRegion,
               vRange=[0, 0.5])
# %%
atomODs, atomODAvg, noiseODs, noiseODAvg, imgIndexMin, imgIndexMax = readInImages(
    imgDir, numOfImgsInEachRun, rampingParameter, trapRegion, noiseRegion)

# %%
K_x, K_y, M2k_Exp, M2k_Fit, popt, atomODAvg = doCalibration(
    imgDir, resDir, trapRegion, noiseRegion, numOfImgsInEachRun,
    rampingParameter, imgSysData, choices)

# %%
plt.imshow(np.log10(M2k_Exp), cmap='jet')
plt.axis('off')
plt.colorbar()
plt.show()

k_x, k_y, K_x, K_y = getFreq(imgSysData["CCDPixelSize"],
                             imgSysData["magnification"], M2k_Exp.shape)
k, S_azmAvg = azmAvg(K_x, K_y, M2k_Exp)
plt.plot(k, S_azmAvg)
plt.xlim(0.2, 3)
plt.show()

from scipy import ndimage
sx, sy = M2k_Exp.shape
X, Y = np.ogrid[0:sx, 0:sy]
r = np.hypot(X - sx / 2, Y - sy / 2)

rbin = (100 * r / r.max()).astype(np.int)
rbin[0:sx // 2, :] = 0
rbin[:, 0:sy // 2] = 0
radial_mean = ndimage.mean(M2k_Exp,
                           labels=rbin,
                           index=np.arange(1,
                                           rbin.max() + 1))
plt.plot(radial_mean)
plt.show()

# %%
choices = {
    "if_Save": False,
    "normalize": False,
    "plot_Sk_azmAvg": False,
    "plot_NPS": False,
    "plot_S2d": False
}

imgDir = r"D:\raw_image\081557"
# the directory where the images are stored

resDir = r"D:\results"
# the directory where the results should be stored

trapRegion = (slice(130, 170), slice(130, 170))
# the region where the atoms located, [(xmin, xmax), (ymin, ymax)] (pixel)

noiseRegion = (slice(0, 300), slice(0, 300))
# the region chosen for noise analysis, [(xmin, xmax), (ymin, ymax)] (pixel)

numOfImgsInEachRun = 60  # number of images for each run of analysis

rampingParameter = 0

#showExampleImg(imgDir, numOfImgsInEachRun, rampingParameter, trapRegion=trapRegion, noiseRegion=noiseRegion, vRange=[0, 1])
#K_x, K_y, S, k, S_azmAvg = \
#    doAnalysis(popt, imgDir, resDir, trapRegion, noiseRegion, numOfImgsInEachRun, rampingParameter, imgSysData, choices)

# %%
