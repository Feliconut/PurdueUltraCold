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
src_list, tar_list, id_list = [], [], []
# mean_list = []
CUTOFF = 99.65
included_dataset = []


def odfft(od):
    return abs(fftshift(fft2(fftshift(od))))


def get_vmax(img, q=CUTOFF):
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

    ods_fft = list(map(crop, ods_fft))
    odfftavg = crop(odfftavg)

    # map the clipped image to [-1,1]
    # standardization
    od_cutoff = sum((map(get_vmax), ods_fft)) / len(ods)
    fftavg_cutoff = get_vmax(odfftavg)
    new_src_list = list(
        map(lambda od_fft: np.clip(od_fft * 2 / od_cutoff - 1, -1, 1),
            ods_fft))
    new_tar_list = [np.clip(odfftavg * 2 / fftavg_cutoff - 1, -1, 1)
                    ] * len(ods_fft)

    img_iter = zip(ods, new_src_list, new_tar_list)
    while True:
        # rawinput = input(
        #     'C = next image if possible, A = accept current dataset, D = decline current dataset, E = exit'
        # )
        accepted_ids = set(
            '031550,051352,051624,051649,141001,141147,141153,141201,141206,141902,151033,151038,151413,151418,191428,191433,191439,191444,191450,191454,191457,211412,211455,211503,211508,211513,221119,221124,221524,221529'
            .split(','))
        rawinput = 'A' if dataset_id in accepted_ids else 'D'
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
            id_list.extend([dataset_id] * len(ods_fft))
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
def save():
    print('start saving')
    np.savez_compressed(join(dirname(__file__), 'data/data128'), src_list,
                        tar_list, id_list)
    print('finish saving')


# %%
