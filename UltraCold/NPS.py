'Calculation of the Noise Power Spectrum (NPS)'
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import abs, pi
from numpy.fft import fft2, fftshift

from .util import get_freq


def _an_from_od(OD, imgSysData=None):
    """
    Calculate atom surface density (sd) from optical density (od).
    """

    if imgSysData["ODtoAtom"] == 'beer':
        px = imgSysData["CCDPixelSize"] / imgSysData["magnification"]
        AtomNum = OD * 2 * pi / (3 * imgSysData["wavelen"]**2) * px**2
    else:
        AtomNum = OD * imgSysData["ODtoAtom"]
    return AtomNum


def from_od(ODs, OD_avg=None, norm=True, imgSysData=None):
    """
    Calculate the noise power spectrum (NPS), from a set of images and their
    average. Note that `ODs` is considered as an ensemble, which means each
    image of `ODs` should be taken under identical conditions, and `ODAvg`
    is considered as the ensemble average.

    ODs: list, each element of which is a numpy.ndarray, the matrix of each 
        image.
    ODAvg: numpy.ndarray, the ensemble average of `ODs`.
    norm: bool. If true, use the atom number to normalize the noise power 
        spectrum. If false, use OD to calculate. In the latter case, the 
        absolute value of the noise power spectrum is meaningless.

    NPSs_avg: numpy.ndarray, the average of noisePowSpecs, which is taken as 
        ensemble average
    NPSs: list, each element of which is the noise power spectrums of
        an image in `ODs`. 
    """
    NPSs = []
    if OD_avg is None:
        OD_avg = np.mean(ODs, axis=0)
    for OD in ODs:
        noise = OD - OD_avg
        if norm: noise = _an_from_od(noise, imgSysData)
        noiseFFT = fftshift(fft2(fftshift(noise)))
        NPS = abs(noiseFFT)**2
        NPSs.append(NPS)

    NPSs_avg = sum(NPSs) / len(NPSs)
    return NPSs_avg, NPSs


def visualize(NPS, K_X, K_Y):
    NPS[NPS == NPS.max()] = 0
    fig_NPS = plt.figure()
    ax_NPS = fig_NPS.add_subplot(111)
    pc = ax_NPS.pcolor(K_X, K_Y, NPS, cmap=cm.jet, shading='auto')
    plt.colorbar(pc)
    ax_NPS.set_aspect(1)
    ax_NPS.set_xlabel('$k_x$ ($\\mu$m$^{-1}$)')
    ax_NPS.set_ylabel('$k_y$ ($\\mu$m$^{-1}$)')
    ax_NPS.set_title('Noise power spectrum')
    return fig_NPS


def visualize_exp(M2k_Exp, K_x, K_y, axes=None, cMap=cm.jet, vRange=None):
    """
    Plot noise power spectrum (only experimental result)
    """
    if axes == None:
        fig_NPS_Exp = plt.figure(figsize=(6, 4.5))
        ax_NPS_Exp = fig_NPS_Exp.add_subplot(111)
    else:
        ax_NPS_Exp = axes
    if vRange == None:
        pc_NPS_Exp = ax_NPS_Exp.pcolor(K_x,
                                       K_y,
                                       M2k_Exp,
                                       cmap=cMap,
                                       shading='auto')
    else:
        pc_NPS_Exp = ax_NPS_Exp.pcolor(K_x,
                                       K_y,
                                       M2k_Exp,
                                       cmap=cMap,
                                       vmin=vRange[0],
                                       vmax=vRange[1],
                                       shading='auto')
    plt.colorbar(pc_NPS_Exp)
    ax_NPS_Exp.set_aspect(1)
    ax_NPS_Exp.set_title("Noise Power Spectrum (Exp.)")
    ax_NPS_Exp.set_xlabel('$k_x$ ($\\mu$m$^{-1}$)')
    ax_NPS_Exp.set_ylabel('$k_y$ ($\\mu$m$^{-1}$)')
    return fig_NPS_Exp, ax_NPS_Exp


def visualize_exp_fit(M2k_Exp,
                      K_x,
                      K_y,
                      M2k_Fit=None,
                      fig=None,
                      cMap=cm.jet,
                      vRange_Exp=None,
                      vRange_Fit=None,
                      **kwargs):
    """
    Plot noise power spectrum (experimental result and fit result)
    """
    if fig == None:
        fig_NPS = plt.figure('NPS', figsize=(12, 5))
    else:
        fig_NPS = fig

    if vRange_Exp == None:
        vMin_Exp = 0
        vMax_Exp = M2k_Fit.max()
    if vRange_Fit == None:
        vMin_Fit = 0
        vMax_Fit = M2k_Fit.max()

    ax_NPS_Exp = fig_NPS.add_subplot(121)
    pc_NPS_Exp = ax_NPS_Exp.pcolor(K_x,
                                   K_y,
                                   M2k_Exp,
                                   cmap=cMap,
                                   vmin=vMin_Exp,
                                   vmax=vMax_Exp,
                                   shading='auto')
    divider = make_axes_locatable(ax_NPS_Exp)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(pc_NPS_Exp, cax=cax, extend='both')
    ax_NPS_Exp.set_aspect(1)
    ax_NPS_Exp.set_title("Noise Power Spectrum (Exp.)")
    ax_NPS_Exp.set_xlabel('$k_x$ ($\\mu$m$^{-1}$)')
    ax_NPS_Exp.set_ylabel('$k_y$ ($\\mu$m$^{-1}$)')

    ax_NPS_Fit = fig_NPS.add_subplot(122)
    pc_NPS_Fit = ax_NPS_Fit.pcolor(K_x,
                                   K_y,
                                   M2k_Fit,
                                   cmap=cMap,
                                   vmin=vMin_Fit,
                                   vmax=vMax_Fit,
                                   shading='auto')
    divider = make_axes_locatable(ax_NPS_Fit)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(pc_NPS_Fit, cax=cax, extend='both')
    ax_NPS_Fit.set_aspect(1)
    ax_NPS_Fit.set_title("Noise Power Spectrum (Fit)")
    ax_NPS_Fit.set_xlabel('$k_x$ ($\\mu$m$^{-1}$)')
    ax_NPS_Fit.set_ylabel('$k_y$ ($\\mu$m$^{-1}$)')
    return fig_NPS, ax_NPS_Exp, ax_NPS_Fit


def visualize_line_cut(M2k_Exp, k_x, k_y, M2k_Fit, **kwargs):
    """
    Plot noise power spectrum - line cut
    """
    ll = M2k_Exp.shape[0]
    k_r = np.linspace(-4, 4, ll)
    t_scan = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    linetype = [['.-b', '-b'], ['.-k', '-k'], ['.-r', '-r'], ['.-m', '-m']]
    title = ['0', 'pi/4', 'pi/2', '3*pi/4']

    fig_NPS_LineCut = plt.figure('NPS_LineCut', figsize=(12, 9))
    for k in range(len(t_scan)):
        t = t_scan[k]

        kx, ky = k_r * np.cos(t), k_r * np.sin(t)
        ix = np.array(k_x.shape[0] * (kx - k_x.min()) /
                      (k_x.max() - k_x.min()),
                      dtype=int)
        iy = np.array(k_y.shape[0] * (ky - k_y.min()) /
                      (k_y.max() - k_y.min()),
                      dtype=int)

        M2k_r_exp = M2k_Exp[iy, ix]
        M2k_r_fit = M2k_Fit[iy, ix]

        axNPS_LineCut = fig_NPS_LineCut.add_subplot(2, 2, k + 1)
        axNPS_LineCut.plot(k_r,
                           M2k_r_exp,
                           linetype[k][0],
                           linewidth=0.5,
                           label=title[k] + '-Exp.')
        axNPS_LineCut.plot(k_r,
                           M2k_r_fit,
                           linetype[k][1],
                           linewidth=2,
                           label=title[k] + '-Fit')
        axNPS_LineCut.legend()
        axNPS_LineCut.set_xlabel('$k$ ($\\mu$m$^{-1}$)')
        axNPS_LineCut.set_ylabel('Noise Power Spectrum (a.u.)')
        axNPS_LineCut.set_ylim([0, 1.5 * M2k_Fit.max()])
        #axNPS_LineCut.set_ylim([0, 1])
    return fig_NPS_LineCut
