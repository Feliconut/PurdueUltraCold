import matplotlib.cm as cm
import matplotlib.pyplot as plt

from .load_data import read_image


def plot_od_avg(imgDir,
            trapRegion=(slice(0, 65535), slice(0, 65535)),
            noiseRegion=(slice(0, 65535), slice(0, 65535)),
            vRange=[0, 0.5]):

    (_, atomODAvg,*_), _ = read_image(imgDir)

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

    # return fig_atom, ax_atom
