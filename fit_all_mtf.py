"""
GOAL: Generate fitting result for all datasets.

@author Xiaoyu Liu

SPECIFICATION:

For each dataset id, we have its results in a separate folder.
The results include:
(DATA IMAGES)
- OD image of each measurement
- Mean OD image
- OD_FFT image of each measurement
- MEAN OD_FFT (MTF_EXP) image
(FITTING IMAGES)
- pupil function image
- psf image
- psf_linecut image
- psf_abs2 image
- psf_abs2_linecut image
- NPS fit image
(FITTING RESULT)
- FITTED PARAMETERS
"""

# %%
from os import mkdir
from os.path import exists, join

import numpy as np

SAVE_DIR = "./results/20210516_MTF_FITS"
if not exists(SAVE_DIR):
    mkdir(SAVE_DIR)

# %%
# Util functions
from numpy import mean, percentile, pi, exp
from numpy.fft import fftshift, fft2
# from perform_fitting import fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.sparse import data


def odfft(od):
    return abs(fftshift(fft2(fftshift(od))))


def generic_plot(img,
                 vmin,
                 vmax,
                 cm=cm.jet,
                 name="Untitled",
                 title="Untitled"):
    plt.cla()
    plt.clf()
    plt.imshow(img, cm, vmin=vmin, vmax=vmax)
    plt.title(title)
    # plt.legend()
    plt.colorbar()
    plt.savefig(SAVE_DIR + '/' + name)


def get_percentile_cut(img, q, two_tailed=False):
    if two_tailed:

        def p(x):
            alpha = (100 - q) / 2
            return percentile(x, alpha), percentile(x, 100 - alpha)

        return p(img.flatten())
    return percentile(img.flatten(), q)


def plot_od(od, od_id, dataset_id, **kwargs):
    vmin, vmax = get_percentile_cut(od, 99.6, two_tailed=True)
    title = f"OD-{dataset_id}#{od_id}"
    name = f"{dataset_id}/{title}"
    generic_plot(od, vmin, vmax, name=name, title=title, **kwargs)


def plot_mtf(od, od_id, dataset_id, **kwargs):
    vmax = get_percentile_cut(od, 99.65, two_tailed=False)
    title = f"MTF-{dataset_id}#{od_id}"
    name = f"{dataset_id}/{title}"
    generic_plot(od, 0, vmax, name=name, title=title, **kwargs)


def save_fit_figures(figs):
    names = "pupil, psf, psf_linecut, psf_abs2, psf_abs2_linecut, nps, nps_linecut".split(
        ", ")
    for fig, name in zip(figs, names):
        plt.savefig(name, fig)


# %%
# Fitting cost function
def m2k_fit_cost_concentric(size, radius, decay_radius):
    center = size // 2

    def fdecay(x, y):
        r = np.abs((x - center)**2 + (y - center)**2)
        return 1 - exp(-(r / (decay_radius / 3))**2)

    def fcirc(x, y):
        return np.abs((x - center)**2 + (y - center)**2) <= radius

    decay_mask = np.fromfunction(fdecay, (size, size))
    nonempty_mask = np.fromfunction(fcirc, (size, size))
    weight = decay_mask * nonempty_mask

    def fcost(fit: np.ndarray, obs):
        return np.sqrt(np.mean((fit - obs)**2 * weight))

    return fcost


# %%
# READ THE DATA
from UltraCold import MTF, OD

para_bounds = [[1, 0.5, -20, -20, -pi, -30, -pi], [50, 10, 20, 20, pi, 30, pi]]

if __name__ == '__main__':
    for ods, dataset_id in OD.get_dataset("031550",
                                          mode='group',
                                          auto_trap=True):
        para_bounds = [[1, 0, 0, 0.8, -0.5, -0.3, 0],
                       [50, 10, 0.1, 1, -0.4, 0, 0.1]]
        # for ods, dataset_id in OD.iter_through_dir(mode='group', auto_trap=True):
        odmean = mean(ods, axis=0)
        od_fft_mean = mean(list(map(odfft, ods)), axis=0)

        M2k_Exp = od_fft_mean
        from perform_fitting import imgSysData, fit_visualizations
        res = MTF.fit(
            M2k_Exp,
            imgSysData,
            dict_format=True,
            paras_bounds=para_bounds,
            m2k_fit_cost=m2k_fit_cost_concentric(100, 15, 1),
        )

        # generate visualize
        M2k_Fit, params = res['M2k'], res['params']

        saveDir = join(SAVE_DIR, dataset_id)
        params['saveDir'] = saveDir

        # create folder
        try:
            mkdir(saveDir)
        except:
            pass

        # plot OD mean
        try:
            plot_od(odmean, 'mean', dataset_id)
        except:
            pass

        # plot OD fft mean
        try:
            plot_mtf(od_fft_mean, 'mtf-exp', dataset_id)
        except:
            pass

        # perform fitting
        fit_visualizations(params, M2k_Exp, M2k_Fit)

        # plot OD fft fit
        try:
            plot_mtf(M2k_Fit, 'mtf-fit', dataset_id)
        except:
            pass

        # write the parameters
        with open(join(saveDir, "fit.txt"), 'w+') as f:
            lines = [
                f"A = {params['A_fit']}", \
                f"tau = {params['tau_fit']}", \
                f"S0 = {params['S0_fit']}", \
                f"alpha = {params['alpha_fit']}", \
                f"phi = {params['phi_fit']}", \
                f"beta = {params['beta_fit']}", \
                f"delta_s = {params['delta_s_fit']}" \
                f"pcov = {params['pcov']}", \
                f"para_bounds = "+repr(para_bounds), \
            ]
            f.write('\n'.join(lines))

# %%

# %%
