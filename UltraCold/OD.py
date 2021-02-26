import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy import log as ln


def from_image_dir(imgDir,
               ramping_param=0,
               trapRegion=(slice(0, 65535), slice(0, 65535)),
               noiseRegion=(slice(0, 65535), slice(0, 65535))):
    """
    Read in images, and calculate optical depth (OD).
    It selects two regions, one with atoms, one without atoms. The latter is
    served as an estimation of noise and to be subtracted in Fourier space.

    ----------
    parameters

    imgDir: string, the directory where the images of atoms are stored.
    numOfImgsInEachRun: int, the number of images you want to include.
    trapRegion: slice, mark the position of the atoms. 
        [(xmin, xmax), (ymin, ymax)] (pixel)
    noiseRegion: slice, mark the position chosen to serve as an estimation 
        of noise. [(xmin, xmax), (ymin, ymax)] (pixel)

    -------
    returns

    atomODs: list, each element of which is a numpy.ndarray, the matrix 
        for optical depth of the region with atoms.
    atomODAvg: numpy.ndarray, the average of all images in `atomODs`.
    noiseODs: list, each element of which is a numpy.ndarray, the matrix 
        for optical depth of the region without atoms.
    noiseODAvg: numpy.ndarray, the average of all images in `noiseODs`.
    imgIndexMin, imgIndexMax: string, the minimum and maximum index of the
        images included
    """

    # get filenames and the minimum and maximum index of the images
    fnames = []
    with open(imgDir + "\\parameters.txt", "r") as paraFile:
        lines = paraFile.readlines()

        for line in lines[1:]:
            img_id, img_ramping_param, *_ = line.split()
            if float(img_ramping_param) == ramping_param:
                fnames.append(imgDir + "\\rawimg_" + img_id)

        imgIndexMin, *_ = lines[1].split()
        imgIndexMax, *_ = lines[-1].split()

    # read in images
    ODs_atom = []
    ODs_noise = []
    for fname in fnames:
        # encoding: big-endian 16-bit unsigned int
        # first digits specify size of the dataset
        f = np.fromfile(fname, '>u2').astype(int)
        xmin, xmax, ymin, ymax = f[0:4]
        img = f[4:].reshape((xmax, ymax))

        # shape of image:
        # Img is 600 * 300 matrix
        # Img = [A ; B] where A and B are 300*300 square matrixes
        # img A is don't have atom (empty), img B has atom
        img_empty = img[xmax // 2:, :]  # img A
        img_atom = img[:xmax // 2, :]  # img B

        I_atom_trap = img_atom[trapRegion]
        I_empty_trap = img_empty[trapRegion]
        I_atom_noise = img_atom[noiseRegion]
        I_empty_noise = img_empty[noiseRegion]

        # see (eqn. 8, Pyragius 2012)
        OD_atom = ln(I_empty_trap / I_atom_trap)
        OD_noise = ln(I_empty_noise / I_atom_noise)

        ODs_atom.append(OD_atom)
        ODs_noise.append(OD_noise)

    # ODs_atom_avg = np.nan_to_num(sum(ODs_atom) / len(ODs_atom))
    # ODs_noise_avg = np.nan_to_num(sum(ODs_noise) / len(ODs_noise))

    # OD_data = (ODs_atom, ODs_atom_avg, ODs_noise, ODs_noise_avg)
    img_index_range = imgIndexMin, imgIndexMax

    return ODs_atom, ODs_noise, img_index_range
    # return OD_data, img_index_range
   
def visualize(atomOD, X=None, Y=None, axes=None, cMap=cm.jet, vRange=None):
    """
    Plot the averaged image (OD) of the 2D thermal atmoic gas
    """
    if axes == None:
        fig_atom = plt.figure(figsize=(6, 4.5))
        ax_atom = fig_atom.add_subplot(111)
    else:
        ax_atom = axes
    
    if X == None or Y == None:
        if vRange == None:
            pc_atom = ax_atom.pcolor(atomOD, cmap=cMap)
        else:
            pc_atom = ax_atom.pcolor(atomOD, cmap=cMap, \
                vmin=max(vRange[0], atomOD.min()), \
                vmax=min(vRange[1], atomOD.max()))
        ax_atom.set_xlabel('$x$ (px)')
        ax_atom.set_ylabel('$y$ (px)')
    else:
        if vRange == None:
            pc_atom = ax_atom.pcolor(X, Y, atomOD, cmap=cMap)
        else:
            pc_atom = ax_atom.pcolor(X, Y, atomOD, cmap=cMap, \
                vmin=max(vRange[0], atomOD.min()), \
                vmax=min(vRange[1], atomOD.max()))
        ax_atom.set_xlabel('$x$ ($\\mu$m)')
        ax_atom.set_ylabel('$y$ ($\\mu$m)')
    plt.colorbar(pc_atom, extend='both')
    ax_atom.set_aspect(1)
    ax_atom.set_title("2D thermal gas (OD)")
    return fig_atom, ax_atom