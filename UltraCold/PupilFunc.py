from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np


def pupil_func(R_p, Theta_p, tau, S0, alpha, phi, beta):
    """
    Given the polar coordinates, aberration parameters, 
    calculate the exit pupil function.

    ----------
    parameters

    R_p: radial coordinate
    Theta_p: azimuthal coordinate
    tau: describe the radial transmission decaying of the pupil. 
        $T(r) = T_0 \\exp\\left( -r^2 / tau^2 \\right)$
    S0: spherical aberration, $S_0 r^4$
    alpha, phi: astigmatism, 
        $\\alpha r^2 \\cos\\left(2\\theta - 2\\phi\\right)$
    beta: defocus, $\\beta r^2$

    ------
    return

    Exit pupil function
    """
    U = np.exp(-(R_p / tau)**2) * np.array(R_p <= 1, dtype=float)
    Phase = S0 * (R_p**4) + \
            alpha * (R_p**2) * np.cos(2*Theta_p - 2*phi) + \
            beta * (R_p**2)
    return U * np.exp(1j * Phase)


def visualize(tau_fit,
              S0_fit,
              alpha_fit,
              phi_fit,
              beta_fit,
              if_Save=False,
              saveDir=None,
              **kwargs):
    """
    Plot the pupil
    """

    r_p_pupilplt = np.linspace(0, 1, 200)
    theta_p_pupilplt = np.linspace(-np.pi, np.pi, 300)

    R_p_pupilplt, Theta_p_pupilplt = np.meshgrid(r_p_pupilplt,
                                                 theta_p_pupilplt)

    pupilplt = pupil_func(R_p_pupilplt, Theta_p_pupilplt, tau_fit, S0_fit,
                          alpha_fit, phi_fit, beta_fit)

    X_pupilplt = R_p_pupilplt * np.cos(Theta_p_pupilplt)
    Y_pupilplt = R_p_pupilplt * np.sin(Theta_p_pupilplt)

    fig_pupil = plt.figure('pupil', figsize=(12, 6))
    ax_pupil = fig_pupil.add_subplot(121)
    pc_pupil = ax_pupil.pcolor(X_pupilplt,
                               Y_pupilplt,
                               np.angle(pupilplt),
                               cmap=cm.twilight_shifted,
                               vmin=-np.pi,
                               vmax=np.pi)
    ax_pupil.set_aspect(1)
    ax_pupil.set_title('Phase of exit pupil (radian)')
    ax_pupil.axis('off')
    divider = make_axes_locatable(ax_pupil)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(pc_pupil, cax=cax)

    ax_pupil_2 = fig_pupil.add_subplot(122)
    pc_pupil_2 = ax_pupil_2.pcolor(X_pupilplt,
                                   Y_pupilplt,
                                   np.angle(pupilplt),
                                   cmap=cm.RdYlGn)
    ax_pupil_2.set_aspect(1)
    ax_pupil_2.set_title('Phase of exit pupil (radian)')
    ax_pupil_2.axis('off')
    divider = make_axes_locatable(ax_pupil_2)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(pc_pupil_2, cax=cax)
    if if_Save:
        plt.savefig(saveDir + "\\Pupil.png", dpi='figure')
    return fig_pupil, ax_pupil, ax_pupil_2
