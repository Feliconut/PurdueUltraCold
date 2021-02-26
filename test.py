# %%
import os

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean

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

# %% experiment for prediction

DATASET_ID = "221529"
img_dir = join(DATA_DIR, DATASET_ID)
ODs_atom_1, ODs_noise_1, _ = OD.from_image_dir(img_dir,
                                               ramping_param=0,
                                               noiseRegion=noiseRegion,
                                               trapRegion=trapRegion)
DATASET_ID = "131805"
img_dir = join(DATA_DIR, DATASET_ID)
ODs_atom_2, ODs_noise_2, _ = OD.from_image_dir(img_dir,
                                               ramping_param=0,
                                               noiseRegion=noiseRegion,
                                               trapRegion=trapRegion)

# %%
ODs_noise_1_mean = np.mean(ODs_noise_1, axis=0)
ODs_noise_2_mean = np.mean(ODs_noise_2, axis=0)


# %%
def img_cost(img1, img2, norm=2):
    assert img1.shape == img2.shape
    return np.linalg.norm(img1 - img2, ord=norm)


# %%
plt.hist([img_cost(img, ODs_noise_1_mean) for img in ODs_noise_1])

plt.hist([img_cost(img, ODs_noise_2_mean) for img in ODs_noise_2])

# from itertools import product
# plt.hist([img_cost(img1,img2) for img1, img2 in product(ODs_noise_1,ODs_noise_2)])
# %%

# %%
ODs_noise_1_flattened = [img.flatten() for img in ODs_noise_1]
ODs_noise_2_flattened = [img.flatten() for img in ODs_noise_2]

from scipy.cluster.vq import kmeans
res = kmeans(ODs_noise_1_flattened + ODs_noise_2_flattened, 2)
(cent1, cent2), distortion = res
plt.hist([img_cost(img.flatten(), cent1) for img in ODs_noise_1])
plt.hist([img_cost(img.flatten(), cent2) for img in ODs_noise_2])
plt.show()
# plt.hist([img_cost(img.flatten(),cent1) for img in ODs_noise_2])
# plt.hist([img_cost(img.flatten(),cent2) for img in ODs_noise_1])

print(img_cost(cent1, cent2))
print(img_cost(cent1, ODs_noise_1_mean.flatten()))
print(img_cost(cent2, ODs_noise_2_mean.flatten()))

# %%


# %%
def ODmean_from_OD_1(OD):
    from scipy import signal
    _img = OD
    scharr = np.ones((3, 3))
    scharr = scharr / scharr.sum()
    _img_conv = signal.convolve2d(_img, scharr, mode='same')
    ODmean_pred = _img_conv
    return ODmean_pred


def ODmean_from_OD_2(OD):
    return np.ones((OD.shape)) * OD.mean()


def img_cost_fft(od1, od2):
    from numpy.fft import fft2, fftshift
    freq1 = fftshift(fft2(fftshift(od1)))
    freq2 = fftshift(fft2(fftshift(od2)))
    return img_cost(freq1, freq2, norm=2)


# %%
ODmean_pred = ODmean_from_OD_1(ODs_noise_1[0])
# OD.visualize(ODmean_pred)
# %%
img_cost(ODmean_pred, ODs_noise_1_mean)
# 0.8
img_cost_fft(ODs_noise_1_mean, ODmean_pred)
# 146
# %%
# OD_noise_1_delta = ODs_noise_1-ODs_noise_1_mean
# %%
plt.hist(
    [img_cost(ODmean_from_OD_1(od), ODs_noise_1_mean) for od in ODs_noise_1])
# 0.7 - 1.1
# %%
plt.hist(
    [img_cost(ODmean_from_OD_2(od), ODs_noise_1_mean) for od in ODs_noise_1])
#0.8 - 1
# %%
plt.hist([
    img_cost_fft(ODmean_from_OD_1(od), ODs_noise_1_mean) for od in ODs_noise_1
])
# 130 - 180
# %%
plt.hist([
    img_cost_fft(ODmean_from_OD_2(od), ODs_noise_1_mean) for od in ODs_noise_1
])
# 150 - 160
# %%

IMG_IDs = ['131805',141001,141153]


def cost_func(
    ods,
    f_od_nps,
    f_ods_nps,
    img_cost_func,
    plot=True
):

    nps_ref = f_ods_nps(ods)
    costs = [img_cost_func(f_od_nps(od), nps_ref) for od in ods]
    if plot: plt.hist(costs)
    # print(costs)
    print(np.mean(costs))
    return costs


def cost_multi_sets(
    img_data_ids,
    f_od_nps,
    f_ods_nps,
    img_cost_func,
):
    for img_id in img_data_ids:
        img_dir = join(DATA_DIR, str(img_id))
        _, ODs_noise, _ = OD.from_image_dir(img_dir,
                                            ramping_param=0,
                                            noiseRegion=noiseRegion,
                                            trapRegion=trapRegion)
        costs = cost_func(
            ODs_noise,
            f_od_nps,
            f_ods_nps,
            img_cost_func,
            plot=False
        )
        plt.hist(costs, label=img_id,alpha=0.3)
    plt.legend()
    plt.show()
    


# %%

# %%
# optimal, reality
cost_func(ods=ODs_noise_1,
          f_od_nps=lambda od: NPS.from_od(
              [od], ODs_noise_1_mean, imgSysData=imgSysData)[0],
          f_ods_nps=lambda ods: NPS.from_od(ods, imgSysData=imgSysData)[0],
          img_cost_func=img_cost)


# 6.7e5
# %%
def pred(od):
    od_mean_pred = ODmean_from_OD_2(od)
    nps, _ = NPS.from_od(od, od_mean_pred, imgSysData=imgSysData)
    return nps


cost_func(ods=ODs_noise_1,
          f_od_nps=pred,
          f_ods_nps=lambda ods: NPS.from_od(ods, imgSysData=imgSysData)[0],
          img_cost_func=img_cost)


# 3.3e7
# %%
def pred(od):
    od_mean_pred = ODmean_from_OD_1(od)
    nps, _ = NPS.from_od(od, od_mean_pred, imgSysData=imgSysData)
    return nps


cost_multi_sets(
    img_data_ids=IMG_IDs,
    f_od_nps=pred,
    f_ods_nps=lambda ods: NPS.from_od(ods, imgSysData=imgSysData)[0],
    img_cost_func=img_cost)
# 3.3e7

 # %%
