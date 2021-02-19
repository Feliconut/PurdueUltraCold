import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .PupilFunc import pupil_func, visualize as fplot_pupil
from .NPS import visualize_exp_fit as fplot_NPS_ExpAndFit


def fplot_PSF(tau_fit,
              S0_fit,
              alpha_fit,
              phi_fit,
              beta_fit,
              delta_s_fit,
              d,
              if_Save=False,
              saveDir=None,
              **kwargs):
    """
    Plot the PSF as defined in Chen-Lung Hung et al 2011 New J. Phys. 13 075019
    """

    rg_PSF = 8
    px_PSF = 0.05

    px_Pupil = 1 / (2 * rg_PSF)
    rg_Pupil = 1 / (2 * px_PSF)
    sampleNum = np.int(2 * rg_Pupil / px_Pupil)

    x = np.linspace(-rg_Pupil, rg_Pupil, sampleNum)
    Xi, Eta = np.meshgrid(x, x)
    R_p_PSFplt, Theta_p_PSFplt = np.abs(Xi + 1j * Eta), np.angle(Xi + 1j * Eta)

    Pupil = pupil_func(R_p_PSFplt, Theta_p_PSFplt, tau_fit, S0_fit, alpha_fit,
                       phi_fit, beta_fit)

    U = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Pupil)))
    PSF = np.real(np.exp(1j * delta_s_fit) * U)

    fx = np.fft.fftshift(np.fft.fftfreq(x.shape[0], px_Pupil))
    X, Y = np.meshgrid(fx * d * 2 * np.pi, fx * d * 2 * np.pi)

    fig_PSF = plt.figure('PSF', figsize=(6, 4.5))
    ax_PSF = fig_PSF.add_subplot(111)
    MM = np.max([np.abs(PSF.max()), np.abs(PSF.min())])
    pc_PSF = ax_PSF.pcolor(X, Y, PSF, cmap=cm.bwr, vmin=-MM, vmax=MM)
    ax_PSF.set_aspect(1)
    ax_PSF.set_xlim([-10, 10])
    ax_PSF.set_ylim([-10, 10])
    ax_PSF.set_title('Point spread function')
    ax_PSF.set_xlabel('$x$ in object plane ($\\mu$m)')
    ax_PSF.set_ylabel('$y$ in object plane ($\\mu$m)')
    plt.colorbar(pc_PSF)
    if if_Save:
        plt.savefig(saveDir + "\\PSF.png", dpi='figure')
    return fig_PSF, ax_PSF


def fplot_PSF_LineCut(tau_fit,
                      S0_fit,
                      alpha_fit,
                      phi_fit,
                      beta_fit,
                      delta_s_fit,
                      d,
                      if_Save=False,
                      saveDir=None,
                      **kwargs):
    """
    Plot the linecut of PSF as defined in Chen-Lung Hung et al 2011 New J. Phys. 13 075019
    """

    rg_PSF = 8
    px_PSF = 0.05

    px_Pupil = 1 / (2 * rg_PSF)
    rg_Pupil = 1 / (2 * px_PSF)
    sampleNum = np.int(2 * rg_Pupil / px_Pupil)

    x = np.linspace(-rg_Pupil, rg_Pupil, sampleNum)
    Xi, Eta = np.meshgrid(x, x)
    R_p_PSFplt, Theta_p_PSFplt = np.abs(Xi + 1j * Eta), np.angle(Xi + 1j * Eta)

    Pupil = pupil_func(R_p_PSFplt, Theta_p_PSFplt, tau_fit, S0_fit, alpha_fit,
                       phi_fit, beta_fit)

    cind = x.shape[0] // 2

    U = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Pupil)))
    PSF = np.real(np.exp(1j * delta_s_fit) * U)

    fx = np.fft.fftshift(np.fft.fftfreq(x.shape[0], px_Pupil))
    X, Y = np.meshgrid(fx * d * 2 * np.pi, fx * d * 2 * np.pi)

    Pupil2 = np.array(R_p_PSFplt <= 1, dtype=float)
    U2 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Pupil2)))
    PSF2 = np.real(np.exp(1j * delta_s_fit) * U2)

    PSF = PSF / PSF[cind, cind]
    PSF2 = PSF2 / PSF2[cind, cind]

    fig_PSF_LineCut = plt.figure('PSF_LineCut', figsize=(6, 4.5))
    ax_PSF_LineCut = fig_PSF_LineCut.add_subplot(111)
    ax_PSF_LineCut.plot(X[cind, :], PSF[cind, :], '-k', label='Fit')
    ax_PSF_LineCut.set_title('Point spread function')
    ax_PSF_LineCut.plot(X[cind, :], PSF2[cind, :], '--k', label='Ideal')
    ax_PSF_LineCut.xaxis.set_major_locator(MultipleLocator(1))
    ax_PSF_LineCut.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax_PSF_LineCut.set_xlabel('$x$ in object plane ($\\mu$m)')
    ax_PSF_LineCut.set_ylabel('Point spread function (a.u.)')
    ax_PSF_LineCut.legend()
    ax_PSF_LineCut.grid(True)
    ax_PSF_LineCut.set_xlim([-10, 10])
    if if_Save:
        plt.savefig(saveDir + "\\PSF_LineCut.png", dpi='figure')
    return fig_PSF_LineCut, ax_PSF_LineCut


def fplot_PSF_abs2(tau_fit,
                   S0_fit,
                   alpha_fit,
                   phi_fit,
                   beta_fit,
                   delta_s_fit,
                   d,
                   if_Save=False,
                   saveDir=None,
                   **kwargs):
    """
    Plot the PSF as traditionally defined
    """

    rg_PSF = 8
    px_PSF = 0.05

    px_Pupil = 1 / (2 * rg_PSF)
    rg_Pupil = 1 / (2 * px_PSF)
    sampleNum = np.int(2 * rg_Pupil / px_Pupil)

    x = np.linspace(-rg_Pupil, rg_Pupil, sampleNum)
    Xi, Eta = np.meshgrid(x, x)
    R_p_PSFplt, Theta_p_PSFplt = np.abs(Xi + 1j * Eta), np.angle(Xi + 1j * Eta)

    Pupil = pupil_func(R_p_PSFplt, Theta_p_PSFplt, tau_fit, S0_fit, alpha_fit,
                       phi_fit, beta_fit)

    cind = x.shape[0] // 2

    U = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Pupil)))
    PSF = np.abs(np.exp(1j * delta_s_fit) * U)**2

    fx = np.fft.fftshift(np.fft.fftfreq(x.shape[0], px_Pupil))
    X, Y = np.meshgrid(fx * d * 2 * np.pi, fx * d * 2 * np.pi)

    fig_PSF_abs2 = plt.figure('PSF_abs2', figsize=(6, 4.5))
    ax_PSF_abs2 = fig_PSF_abs2.add_subplot(111)
    MM = np.max([np.abs(PSF.max()), np.abs(PSF.min())])
    pc = ax_PSF_abs2.pcolor(X, Y, PSF, cmap=cm.bwr, vmin=-MM, vmax=MM)
    ax_PSF_abs2.set_aspect(1)
    ax_PSF_abs2.set_xlim([-10, 10])
    ax_PSF_abs2.set_ylim([-10, 10])
    ax_PSF_abs2.set_title('Point spread function (classical def.)')
    ax_PSF_abs2.set_xlabel('$x$ in object plane ($\\mu$m)')
    ax_PSF_abs2.set_ylabel('$y$ in object plane ($\\mu$m)')
    plt.colorbar(pc)
    if if_Save:
        plt.savefig(saveDir + "\\PSF_abs2.png", dpi='figure')
    return fig_PSF_abs2, ax_PSF_abs2


# plot the PSF_abs2 - line cut
def lin_interp(x, y, i, half):
    return x[i] + (x[i + 1] - x[i]) * ((half - y[i]) / (y[i + 1] - y[i]))


def uPercent_max_x(x, y):
    h = max(y) / 100.0
    signs = np.sign(np.add(y, -h))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [
        lin_interp(x, y, zero_crossings_i[0], h),
        lin_interp(x, y, zero_crossings_i[1], h)
    ], h


def calcResolution(PSF_abs2_linecut, x):
    # find the two crossing points
    hmx, h = uPercent_max_x(x, PSF_abs2_linecut)
    res = (hmx[1] - hmx[0]) / 2
    return res, hmx[1], hmx[0], h


def fplot_PSF_abs2_LineCut(tau_fit,
                           S0_fit,
                           alpha_fit,
                           phi_fit,
                           beta_fit,
                           delta_s_fit,
                           d,
                           if_Save=False,
                           saveDir=None,
                           **kwargs):
    """
    Plot the linecut of PSF as traditionally defined. Estimate the resolution.
    """

    rg_PSF = 8
    px_PSF = 0.05

    px_Pupil = 1 / (2 * rg_PSF)
    rg_Pupil = 1 / (2 * px_PSF)
    sampleNum = np.int(2 * rg_Pupil / px_Pupil)

    x = np.linspace(-rg_Pupil, rg_Pupil, sampleNum)
    Xi, Eta = np.meshgrid(x, x)
    R_p_PSFplt, Theta_p_PSFplt = np.abs(Xi + 1j * Eta), np.angle(Xi + 1j * Eta)

    Pupil = pupil_func(R_p_PSFplt, Theta_p_PSFplt, tau_fit, S0_fit, alpha_fit,
                       phi_fit, beta_fit)

    cind = x.shape[0] // 2

    U = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Pupil)))
    PSF = np.abs(np.exp(1j * delta_s_fit) * U)**2

    fx = np.fft.fftshift(np.fft.fftfreq(x.shape[0], px_Pupil))
    X, Y = np.meshgrid(fx * d * 2 * np.pi, fx * d * 2 * np.pi)

    Pupil2 = np.array(R_p_PSFplt <= 1, dtype=float)
    U2 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Pupil2)))
    PSF2 = np.abs(np.exp(1j * delta_s_fit) * U2)**2

    PSF = PSF / PSF[cind, cind]
    PSF2 = PSF2 / PSF2[cind, cind]

    fig_PSF_abs2_LineCut = plt.figure('PSF_abs2_LineCut', figsize=(6, 4.5))
    ax_PSF_abs2_LineCut = fig_PSF_abs2_LineCut.add_subplot(111)
    ax_PSF_abs2_LineCut.plot(X[cind, :], PSF[cind, :], '-k', label='Fit')
    ax_PSF_abs2_LineCut.set_title('Point spread function (classical def.)')
    ax_PSF_abs2_LineCut.plot(X[cind, :], PSF2[cind, :], '--k', label='Ideal')
    ax_PSF_abs2_LineCut.xaxis.set_major_locator(MultipleLocator(1))
    ax_PSF_abs2_LineCut.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax_PSF_abs2_LineCut.legend()
    ax_PSF_abs2_LineCut.grid(True)
    ax_PSF_abs2_LineCut.set_xlim([-10, 10])
    ax_PSF_abs2_LineCut.set_xlabel('$x$ in object plane ($\\mu$m)')
    if if_Save:
        plt.savefig(saveDir + "\\PSF_abs2_LineCut.png", dpi='figure')

    resolution, b, a, h = calcResolution(PSF[cind, :], X[cind, :])
    ax_PSF_abs2_LineCut.plot([a, b], [h, h], '-b')
    return resolution, fig_PSF_abs2_LineCut, ax_PSF_abs2_LineCut
