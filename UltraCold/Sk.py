'Calculation of the static structure factor'

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from .util import azm_avg, get_freq


def from_nps_mtf(NPS, M2k, imgSysData, if_plot=False):
    """
    Given the normalized density fluctuation power spectrum, calculate the 
    static structure factor and do the azimuthal average.
    """
    M2k[M2k == 0] = np.inf
    # from (eqn. 8, Hung 2011 "extracting density-density correlations")
    S = NPS / M2k
    _1, _2, K_x, K_y = get_freq(imgSysData["CCDPixelSize"],
                                imgSysData["magnification"], S.shape)

    k, S_azmAvg = azm_avg(K_x, K_y, S)

    # extrapolation = InterpolatedUnivariateSpline(k[k.shape[0]//4:3*k.shape[0]//4], S_azmAvg[k.shape[0]//4:3*k.shape[0]//4], k=2)
    # S_k0 = extrapolation(0)

    return K_x, K_y, S, k, S_azmAvg


def visualize2d(S, K_X, K_Y):
    S[S == S.max()] = 0
    fig_S2d = plt.figure()
    ax_S2d = fig_S2d.add_subplot(111)
    pc = ax_S2d.pcolor(K_X, K_Y, S, cmap=cm.jet, shading='auto')
    plt.colorbar(pc)
    ax_S2d.set_aspect(1)
    ax_S2d.set_xlabel('$k_x$ ($\\mu$m$^{-1}$)')
    ax_S2d.set_ylabel('$k_y$ ($\\mu$m$^{-1}$)')
    ax_S2d.set_title('Static structure factor')
    return fig_S2d


def visualize_azmAvg(S_azmAvg, k):
    fig_S_azm = plt.figure('static structure factor', figsize=(6, 4))
    ax_S_azm = fig_S_azm.add_subplot(111)
    ax_S_azm.plot(k[1:], S_azmAvg[1:], '.b')
    ax_S_azm.set_xlabel('$k$ ($\\mu$m$^{-1}$)')
    ax_S_azm.set_ylabel('$S(k)$ (a.u.)')
    ax_S_azm.set_title('Static structure factor (azimuthal average)')
