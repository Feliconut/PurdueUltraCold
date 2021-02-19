import numpy as np
from scipy.optimize.minpack import curve_fit

from . import NPS
from .util import get_freq
from .PupilFunc import pupil_func


# %%
def from_od(OD_data, norm=False, imgSysData=None):
    """
    This is a combination of `ReadInImages` and `calcNPS__`. 
    Calculate the experimental imaging response function.

    ----------
    parameters

    same as `ReadInImages`

    -------
    returns

    M2k_Exp: numpy.ndarray, the calculated imaging response function 
        aalready subtracted by pure noise result).
    M2k_Exp_atom: numpy.ndarray, the imaging response function calculated 
        from images of atoms (suffered from shot noise).
    M2k_Exp_noise: numpy.ndarray, the imaging response function calculated 
        from images of pure noise (only shot noise affects)
    imgIndexMin, imgIndexMax, atomODs, noiseODs, atomODAvg, noiseODAvg: 
        same as `readInImages` 
    """
    ODs_atom, ODs_atom_avg, ODs_noise, ODs_noise_avg = OD_data

    # M2k_Exp is same as NPSs_avg.
    M2k_Exp_atom, _ = NPS.from_od(ODs_atom, ODs_atom_avg, norm, imgSysData)
    M2k_Exp_noise, _ = NPS.from_od(ODs_noise, ODs_noise_avg, norm, imgSysData)

    M2k_Exp_atom = M2k_Exp_atom / ODs_atom_avg.sum()
    M2k_Exp_noise = M2k_Exp_noise / ODs_atom_avg.sum()

    M2k_Exp_atom[M2k_Exp_atom.shape[0] // 2, M2k_Exp_atom.shape[1] // 2] = 0

    M2k_Exp = M2k_Exp_atom  #- M2k_Exp_noise

    return M2k_Exp, M2k_Exp_atom, M2k_Exp_noise


def mtf(K_X, K_Y, d, tau, S0, alpha, phi, beta, delta_s):
    """
    Given the spatial frequency, aberration parameters and the phase 
    introduced by the atom scattering process, 
    calculate the imaging response function

    ----------
    parameters

    K_X, K_Y: spacial frequencies
    d: the ratio for converting the spacial frequencies into 
        coordinates at exit pupil plane
    tau: describe the radial transmission decaying of the pupil. 
        $T(r) = T_0 \\exp\\left( -r^2 / tau^2 \\right)$
    S0: spherical aberration, $S_0 r^4$
    alpha, phi: astigmatism, 
        $\\alpha r^2 \\cos\\left(2\\theta - 2\\phi\\right)$
    beta: defocus, $\\beta r^2$
    delta_s: the phase introduced by the atom scattering process

    ------
    return

    Exit pupil function
    """

    R_p, Theta_p = np.abs(K_X + 1j * K_Y) * d, np.angle(K_X + 1j * K_Y)
    p1 = pupil_func(R_p, Theta_p + np.pi, tau, S0, alpha, phi, beta)
    p2 = np.conj(pupil_func(R_p, Theta_p, tau, S0, alpha, phi, beta)) * \
            np.exp(-2*1j*delta_s)
    PSF = (p1 + p2) / (2 * np.cos(delta_s))
    M2k = np.abs(PSF)**2
    return M2k


def fit(M2k_Exp, imgSysData, paras_guess=None, paras_bounds=None):
    """
    Fit the imaging response function using the model provided in 
    Chen-Lung Hung et al. 2011 New J. Phys. 13 075019
    """

    k_x, k_y, K_x, K_y = get_freq(imgSysData["CCDPixelSize"],
                                  imgSysData["magnification"], M2k_Exp.shape)
    d = imgSysData["wavelen"] / (2 * np.pi * imgSysData["NA"])

    def f(M, *args):
        k_x, k_y = M
        A, tau, S0, alpha, phi, beta, delta_s = args
        return A * mtf(k_x, k_y, d, tau, S0, alpha, phi, beta, delta_s)

    if paras_guess == None:
        #                A ,  tau,   S0, alpha,  phi, beta, delta_s
        paras_guess = [ [1 ,  0.8,  -10,  -0.5, -2.3,   17, 0.158], \
                        [1 ,  0.8,    0,     1, -2.3,    0,     0], \
                        [1 ,    1,    1,     1,  1.5,    1,     3], \
                        [9 ,  0.8,    1,   0.4,-0.85,    0,     1], \
                        [1, 0.63, 0.69,    3, -1.6, 0.35,  -1.2]  ]
    elif paras_guess == 'focus':
        paras_guess = [1, 0.8, 0, 1, -2.3, 0, 0]
    elif paras_guess == 'defocus':
        paras_guess = [1, 0.8, -10 - 0.5, -2.3, 17, 3.3]

    if paras_bounds == None:
        paras_bounds = ([-1000, 0.5, -20, -20, -np.pi, -30,
                         -np.pi], [1000, 10, 20, 20, np.pi, 30, np.pi])

    M2k_Exp_cut = np.clip(M2k_Exp, 0, M2k_Exp.max())
    # cut the negative values, which are meaningless

    xdata = np.vstack((K_x.ravel(), K_y.ravel()))
    # prepare the independent variables, i.e., the coordinates
    xdata = np.delete(xdata,
                      (K_x.shape[0] // 2) * K_x.shape[1] + K_x.shape[1] // 2,
                      axis=1)
    ydata = np.delete(M2k_Exp_cut.ravel(),
                      (K_x.shape[0] // 2) * K_x.shape[1] + K_x.shape[1] // 2,
                      axis=0)
    # leave out the center bright point

    rms_min = np.inf
    for pg in paras_guess:
        popt_temp, pcov = curve_fit(f, xdata, ydata, \
                                p0=pg, maxfev=50000, bounds=paras_bounds)

        A_fit, tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit, delta_s_fit = popt_temp

        M2k_Fit = A_fit * mtf(K_x, K_y, d, tau_fit, \
                            S0_fit, alpha_fit, phi_fit, beta_fit, delta_s_fit)

        rms = np.sqrt(np.mean((M2k_Fit - M2k_Exp_cut)**2))
        if rms < rms_min:
            rms_min = rms
            popt = popt_temp

    return M2k_Fit, rms_min, popt, pcov, k_x, k_y, K_x, K_y, d
