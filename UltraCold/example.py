import matplotlib.cm as cm
import matplotlib.pyplot as plt

from .OD import from_image_dir


def plot_od_avg(imgDir,
                trapRegion=(slice(0, 65535), slice(0, 65535)),
                noiseRegion=(slice(0, 65535), slice(0, 65535)),
                vRange=[0, 0.5]):

    (_, atomODAvg, *_), _ = from_image_dir(imgDir)

    fig_atom = plt.figure()
    ax_atom = fig_atom.add_subplot(111)

    # Plot OD image
    pc = ax_atom.pcolor(atomODAvg, cmap=cm.jet, vmin=vRange[0], vmax=vRange[1])
    ax_atom.set_aspect(1)
    ax_atom.set_title("Example of OD")
    plt.colorbar(pc)

    # draw a ling around the domain

    img_y, img_x = atomODAvg.shape
    trap_x, trap_y = trapRegion
    noise_x, noise_y = noiseRegion

    xm, *_, xM = range(img_x)[trap_x]
    ym, *_, yM = range(img_y)[trap_y]
    xroute = [xm, xm, xM, xM, xm]
    yroute = [ym, yM, yM, ym, ym]
    ax_atom.plot(xroute, yroute, '-r', linewidth=2)

    xm, *_, xM = range(img_x)[noise_x]
    ym, *_, yM = range(img_y)[noise_y]
    xroute = [xm, xm, xM, xM, xm]
    yroute = [ym, yM, yM, ym, ym]
    ax_atom.plot(xroute, yroute, '-m', linewidth=2)

    plt.show()

    return fig_atom, ax_atom


def do_analysis(img_dir,
                trapRegion,
                noiseRegion,
                imgSysData,
                M2k,
                normalize=False):
    from . import MTF, NPS, Sk
    OD_data, img_index_range = from_image_dir(img_dir,
                                              trapRegion=trapRegion,
                                              noiseRegion=noiseRegion)
    NPS_Exp, *_ = MTF.from_od(OD_data, norm=normalize, imgSysData=imgSysData)

    K_X, K_Y, S, k, S_azmAvg = Sk.from_nps_mtf(NPS_Exp, M2k, imgSysData)
    # *_, K_X, K_Y = get_freq(imgSysData["CCDPixelSize"],
    #                 imgSysData["magnification"], NPS.shape)

    NPS.visualize(NPS_Exp, K_X, K_Y)
    Sk.visualize2d(S, K_X, K_Y)
    Sk.visualize_azmAvg(S_azmAvg, k)
    return K_X, K_Y, S, k, S_azmAvg
