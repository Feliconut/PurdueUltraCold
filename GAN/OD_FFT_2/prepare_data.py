# %%
import operator
from os import getcwd, listdir, chdir

from PIL.Image import new
from cv2 import data
import cv2
chdir('../..')
from os.path import dirname, join
import numpy as np
from numpy.fft import fftshift, fft2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from UltraCold import OD, NPS

# %%
# Modify this according to file position
baseDir = getcwd()
DATA_DIR = join(baseDir, 'DATA')

# %%
src_list, tar_list = [], []
# mean_list = []
CUTOFF = 99.65
included_dataset = []


def odfft(od):
    return abs(fftshift(fft2(fftshift(od))))


def get_vmax(img, q):
    return np.percentile(img.flatten(), q)


for ods, dataset_id in OD.iter_through_dir(mode='group'):

    ods_fft = [odfft(od) for od in ods]
    odfftavg = np.mean(ods_fft, axis=0)
    # adjust the size to 128 * 128

    W, H = np.shape(odfftavg)
    if abs(W - H) / (W + H) > 0.1:
        print(f'id{dataset_id} do not have square-like trap region. Reject.')
        continue

    def cropND(img, bounding):
        start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]

    def boundary_values(img):
        res = []
        res.extend(list(img[0:, 0]))
        res.extend(list(img[0:, -1]))
        res.extend(list(img[0, 0:]))
        res.extend(list(img[-1, 0:]))
        return res

    def crop(img):
        # img = cropND(img, (side, side))
        #Creating a dark square with NUMPY
        s = 300
        f = np.zeros((s, s))
        # f.fill(np.mean(img))
        f.fill(np.mean(boundary_values(img)))

        #Getting the centering position
        ax, ay = (s - img.shape[1]) // 2, (s - img.shape[0]) // 2

        #Pasting the 'image' in a centering position
        f[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img

        return cropND(f, (128, 128))

    ods_fft = [crop(img) for img in ods_fft]
    odfftavg = crop(odfftavg )

    # map the clipped image to [-1,1]
    # standardization
    od_cutoff = sum(
        (get_vmax(od_fft, CUTOFF) for od_fft in ods_fft)) / len(ods)
    fftavg_cutoff = get_vmax(odfftavg, CUTOFF)
    new_src_list = [
        np.clip(od_fft * 2 / od_cutoff - 1, -1, 1) for od_fft in ods_fft
    ]
    new_tar_list = [np.clip(odfftavg * 2 / fftavg_cutoff - 1, -1, 1)
                    ] * len(ods_fft)

    img_iter = zip(ods, new_src_list, new_tar_list)
    while True:
        rawinput = input(
            'C = next image if possible, A = accept current dataset, D = decline current dataset, E = exit'
        )
        if rawinput == 'C' or rawinput == '':
            try:
                od, srcimg, tarimg = next(img_iter)
                # fig=plt.figure(figsize=(4,20),dpi=100)
                plt.subplot(1, 3, 1)
                plt.imshow(od)
                plt.axis('off')
                # plt.colorbar()
                plt.subplot(1, 3, 2)
                plt.imshow(srcimg)
                plt.title(f'id{dataset_id}, od / fft(od) / mean(fft(ods))')
                plt.axis('off')
                # plt.colorbar()
                plt.subplot(1, 3, 3)
                plt.imshow(tarimg)
                plt.axis('off')
                # plt.colorbar()
                plt.show()

            except StopIteration as e:
                print('no more image in this set')
        elif rawinput == 'A':
            src_list.extend(new_src_list)
            tar_list.extend(new_tar_list)
            included_dataset.append(dataset_id)
            print(f'accept dataset {dataset_id}')
            break
        elif rawinput == 'D':
            print(f'decline dataset {dataset_id}')
            break
        elif rawinput == 'E':
            raise KeyboardInterrupt()
print('COMPLETE')
print(f'we have {len(src_list)} images in total')
print(f'included datasets: {",".join(included_dataset)}')

# %%
print('start saving')
np.savez_compressed(join(dirname(__file__), 'data/data128'), src_list,
                    tar_list)
print('finish saving')
# %%
