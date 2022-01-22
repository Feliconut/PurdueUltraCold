from typing import NamedTuple

import numpy as np

def get_freq(CCDPixelSize, magnification, M2k_mat_shape):
    """
    Calculate the corresponding spacial frequency $k$.
    """

    pixelSize = CCDPixelSize / magnification  # pixel size in object plane, in micron
    px_x = 1.0 * pixelSize  # micron
    px_y = 1.0 * pixelSize  # micron
    sampleNum_y, sampleNum_x = M2k_mat_shape

    # FT from real space to k space  
    k_x = np.fft.fftshift(np.fft.fftfreq(sampleNum_x, px_x)) * 2 * np.pi
    k_y = np.fft.fftshift(np.fft.fftfreq(sampleNum_y, px_y)) * 2 * np.pi

    # TODO figure out what this means
    K_x, K_y = np.meshgrid(k_x, k_y)

    return k_x, k_y, K_x, K_y


def azm_avg(X, Y, Z):
    """
    Do the azimuthal average for a matrix given in Cartesian coordinates.
    
    ----------
    parameters
    
    X: numpy.ndarray, x-coordinate.
    Y: numpy.ndarray, y-coordinate.
    Z; numpy.ndarray, value at (x, y). X, Y and Z should have same shapes.
    
    ------
    return
    
    sorted_r: numpy.ndarray, radial coordinate, sorted from small to large.
    sorted_v: numpy.ndarray, averaged value at r
    """
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError('X, Y and Z have different shapes!')

    row_ind_max = X.shape[0] - 1
    col_ind_max = X.shape[1] - 1

    R = np.sqrt(X**2 + Y**2)

    row_cind = row_ind_max // 2
    col_cind = col_ind_max // 2

    res_dict = {}

    for ii in range(row_cind + 1):
        for jj in range(col_cind + 1):
            res_dict[R[ii, jj]] = []

    for ii in range(row_cind + 1):
        for jj in range(col_cind + 1):
            v_temp = [Z[ii, jj], Z[row_ind_max - ii, jj], \
                      Z[ii, col_ind_max - jj], Z[row_ind_max - ii, col_ind_max - jj]]
            res_dict[R[ii, jj]].append(np.mean(v_temp))

    for ii in range(row_cind + 1):
        for jj in range(col_cind + 1):
            res_dict[R[ii, jj]] = np.mean(res_dict[R[ii, jj]])

    r_list = []
    v_list = []

    for r in res_dict:
        r_list.append(r)
        v_list.append(res_dict[r])

    r_list = np.array(r_list)
    v_list = np.array(v_list)

    sorted_indices = np.argsort(r_list)
    sorted_r = r_list[sorted_indices]
    sorted_v = v_list[sorted_indices]

    if len(sorted_r.tolist()) != len(set(sorted_r.tolist())):
        print("Warning: there're duplicate elements in output!")

    return sorted_r, sorted_v
