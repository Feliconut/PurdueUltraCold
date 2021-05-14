"""
GOAL: Generate fitting result for all datasets.

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

SAVE_DIR = "./results/20210514_MTF_FITS"
if not exists(SAVE_DIR):
    mkdir(SAVE_DIR)

# %%
# Util functions
from numpy import mean, percentile
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
    plt.imshow(img, cm, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.legend()
    plt.colorbar()
    plt.savefig(SAVE_DIR + '/' + name)
    plt.close()


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
# READ THE DATA
from UltraCold import MTF, OD

for ods, dataset_id in OD.iter_through_dir(mode='group', auto_trap=True):
    plt.close()
    # create folder
    try:
        saveDir = join(SAVE_DIR, dataset_id)
        mkdir(saveDir)
    except:
        pass

    # plot OD mean
    odmean = mean(ods, axis=0)
    try:
        plot_od(odmean, 'mean', dataset_id)
    except:
        pass

    # plot OD fft mean
    od_fft_mean = mean(list(map(odfft, ods)), axis=0)
    try:
        plot_mtf(od_fft_mean, 'mtf-exp', dataset_id)
    except:
        pass

    # for i, od in enumerate(ods):
    #     plot_od(od, i, dataset_id)
    M2k_Exp = od_fft_mean
    from perform_fitting import imgSysData, fit_visualizations
    res = MTF.fit(M2k_Exp, imgSysData, dict_format=True)
    M2k_Fit, params = res['M2k'], res['params']
    # print(params)
    params['saveDir'] = saveDir
    fit_visualizations(params, M2k_Exp, M2k_Fit)

    # plot OD fft fit
    try:
        plot_mtf(M2k_Fit, 'mtf-fit', dataset_id)
    except:
        pass

    pass

    
# %%

# %%
