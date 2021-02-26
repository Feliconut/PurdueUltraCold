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

from os import getcwd
from os.path import dirname, join

# cwd = os.getcwd()

baseDir = dirname(__file__)
cwd = baseDir
# baseDir = dirname(cwd)

DATA_DIR = join(baseDir, 'DATA')

DATASET_ID = '221529'
# the id of dataset where img are stored.

img_dir = join(DATA_DIR, DATASET_ID)
# the directory where the images are stored

res_dir = join(cwd, 'results')
# the directory where the results should be stored

trapRegion = (slice(100, 200), slice(100, 200))
# the region where the atoms located, [(xmin, xmax), (ymin, ymax)] (pixel)

noiseRegion = (slice(0, 300), slice(0, 90))
# the region chosen for noise analysis, [(xmin, xmax), (ymin, ymax)] (pixel)

numOfImgsInEachRun = 50  # number of images for each run of analysis

rampingParameter = 0

# %% TEST

plot_od_avg(img_dir,trapRegion,noiseRegion)
# %%
# ODs_atom, ODs_atom_avg, ODs_noise, ODs_noise_avg = OD_data
from UltraCold import OD

*OD_data, img_index_range = OD.from_image_dir(img_dir,
                                              ramping_param=rampingParameter,
                                              noiseRegion=noiseRegion,
                                              trapRegion=trapRegion)
ODs_atom, ODs_noise = OD_data

# %%
M2k_Exp, M2k_Exp_atom, M2k_Exp_noise = MTF.from_od(OD_data)

*_, K_X, K_Y = get_freq(imgSysData["CCDPixelSize"],
                        imgSysData["magnification"], M2k_Exp.shape)

NPS.visualize_exp(M2k_Exp, K_X, K_Y)

# %% # Calibrate
M2k_Exp, _ = NPS.from_od(ODs_noise, imgSysData=imgSysData)
M2k_Fit, rms_min, popt, pcov, k_x, k_y, K_x, K_y, d = MTF.fit(
    M2k_Exp, imgSysData)

# %%

NPS.visualize_exp_fit(M2k_Exp, K_x, K_y, M2k_Fit)
NPS.visualize_line_cut(M2k_Exp, k_x, k_y, M2k_Fit)
# %%
A_fit, tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit, delta_s_fit = popt
params = {
    'A_fit': A_fit,
    'tau_fit': tau_fit,
    'S0_fit': S0_fit,
    'alpha_fit': alpha_fit,
    'phi_fit': phi_fit,
    'beta_fit': beta_fit,
    'delta_s_fit': delta_s_fit,
    'd': d
}
from UltraCold.plotting import (fplot_NPS_ExpAndFit, fplot_PSF, fplot_PSF_abs2,
                                fplot_PSF_abs2_LineCut, fplot_PSF_LineCut,
                                fplot_pupil)

fplot_pupil(**params)
fplot_PSF(**params)
fplot_PSF_LineCut(**params)
fplot_PSF_abs2(**params)
resolution, _1, _2 = fplot_PSF_abs2_LineCut(**params)
print(
    "The Rayleigh-criterion resolution is approximately {:.1f} micron".format(
        resolution))
fplot_NPS_ExpAndFit(
    K_x,
    K_y,
    M2k_Exp,
    M2k_Fit,
)

# %%
