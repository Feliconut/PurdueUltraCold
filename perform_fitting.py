# %%
# %%
import os

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from UltraCold import MTF, NPS, OD, Sk, plot_od_avg
from UltraCold.util import get_freq

# %%
# %% pre-settings

imgSysData = {
    # pixel size of the CCD, in micron
    "CCDPixelSize": 13,
    # 799.943 / 29.9099, # magnification of the imaging system
    "magnification": 27,
    # wavelength of the imaging beam, in micron
    "wavelen": 0.852,
    # numerical aperture of the objective
    "NA": 0.37,
    #
    "ODtoAtom": 13
}

# from os import getcwd
# from os.path import dirname, join

# # cwd = os.getcwd()

# baseDir = dirname(__file__)
# cwd = baseDir
# # baseDir = dirname(cwd)

# DATA_DIR = join(baseDir, 'DATA')

# DATASET_ID = '221529'
# # the id of dataset where img are stored.

# img_dir = join(DATA_DIR, DATASET_ID)
# # the directory where the images are stored

# res_dir = join(cwd, 'results')
# # the directory where the results should be stored

# trapRegion = (slice(100, 200), slice(100, 200))
# # the region where the atoms located, [(xmin, xmax), (ymin, ymax)] (pixel)

# noiseRegion = (slice(0, 300), slice(0, 90))
# # the region chosen for noise analysis, [(xmin, xmax), (ymin, ymax)] (pixel)

# numOfImgsInEachRun = 50  # number of images for each run of analysis

# rampingParameter = 0


# %%
def fit(M2k_Exp, imgSysData=imgSysData, saveDir=None):
    res = MTF.fit(M2k_Exp, imgSysData, dict_format=True)
    M2k_Fit, params = res['M2k'], res['params']
    # %%
    # params['K_x'] = params['Kx']
    # params['K_y'] = params['Ky']
    # params['k_x'] = params['kx']
    # params['k_y'] = params['ky']
    params['saveDir'] = saveDir
    return fit_visualizations(params, M2k_Exp, M2k_Fit)


    # NPS.visualize_exp_fit(M2k_Exp, **params, M2k_Fit=M2k_Fit)
    # NPS.visualize_line_cut(M2k_Exp, **params, M2k_Fit=M2k_Fit)
    # %%
def fit_visualizations(params, M2k_Exp, M2k_Fit):
    from UltraCold.plotting import (fplot_NPS_ExpAndFit, fplot_PSF,
                                    fplot_PSF_abs2, fplot_PSF_abs2_LineCut,
                                    fplot_PSF_LineCut, fplot_pupil,
                                    fplot_NPS_ExpAndFit_LineCut)

    fplot_pupil(**params)
    plt.cla()
    plt.clf()
    fplot_PSF(**params)
    plt.cla()
    plt.clf()
    fplot_PSF_LineCut(**params)
    plt.cla()
    plt.clf()
    fplot_PSF_abs2(**params)
    plt.cla()
    plt.clf()
    fplot_PSF_abs2_LineCut(**params)
    plt.cla()
    plt.clf()
    # resolution, _1, _2 =
    # print("The Rayleigh-criterion resolution is approximately {:.1f} micron".
    #       format(resolution))
    fplot_NPS_ExpAndFit(
        M2k_Exp,
        **params,
        M2k_Fit=M2k_Fit,
    )
    plt.cla()
    plt.clf()
    fplot_NPS_ExpAndFit_LineCut(
        M2k_Exp,
        **params,
        M2k_Fit=M2k_Fit,
    )
    plt.cla()
    plt.clf()


# %%
