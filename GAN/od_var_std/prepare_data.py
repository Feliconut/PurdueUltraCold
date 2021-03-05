# %%
from os import getcwd, listdir, chdir
chdir('../..')
# %%
from numpy.core.fromnumeric import mean
from numpy.lib.arraypad import pad
from os.path import dirname, join
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from UltraCold import OD, NPS
from UltraCold.util.labling import approx_trap_region
from UltraCold.util.packing import pack

# %%
# Modify this according to file position
baseDir = getcwd()
DATA_DIR = join(baseDir, 'DATA')

# %%
src_list, tar_list = [], []
mean_list = []
id_list = []

# dataRegion = (slice(0, 256), slice(0, 256))
DATA_SIZE = (64, 64)
SRC_RANGE = (-.4, .4)
TAR_RANGE = (0, .01)


def thresh(img, min=-0.3, max=0.3):
    img[img < min] = min
    img[img > max] = max
    return img


for dataset_id in listdir(DATA_DIR):
    if '.' in dataset_id: continue
    dataset_path = join(DATA_DIR, dataset_id)
    ods, _, _ = OD.from_image_dir(dataset_path)
    print(f'id: {dataset_id}, #img: {len(ods)}')
    if not len(ods): continue

    od_mean = np.mean(ods, axis=0)
    mean_list.append(od_mean)
    od_var = np.mean([(od - od_mean)**2 for od in ods], axis=0)

    # determine noise regions
    try:
        trap_x, trap_y = approx_trap_region(od_mean, 5)
    except:
        print(f'id: {dataset_id} skipped due to bad trap region')
        continue
    noise_regions = pack(od_mean.shape, DATA_SIZE,
                         (trap_x.start, trap_y.start, trap_x.stop -
                          trap_x.start, trap_y.stop - trap_y.start))

    for od in ods:
        for noise_region in noise_regions:
            # Standardize to (0, 1)
            src_list.append(
                (np.clip(od[noise_region], *SRC_RANGE) - SRC_RANGE[0]) /
                (SRC_RANGE[1] - SRC_RANGE[0]))
            tar_list.append(
                (np.clip(od_var[noise_region], *TAR_RANGE) - TAR_RANGE[0]) /
                (TAR_RANGE[1] - TAR_RANGE[0]))

# %%
print('start saving')
np.savez_compressed(join(dirname(__file__), 'data/data64_std'), src_list,
                    tar_list, SRC_RANGE, TAR_RANGE)
print('done saving')
# %%
# %% OD distribution analysis
# all_noise = np.array(tar_list).flatten()
# from scipy.stats import norm
# mu, std = norm.fit(all_noise)
# plt.hist(all_noise, normed=True, bins=200)
# xm, xM = plt.xlim()
# # x = np.linspace(xm+0.8, xM-1, 100)
# # plt.plot(x, norm.pdf(x, mu, std), 'k', linewidth=2)
# plt.title(f'{mu:.2e},{std:.2e}')
# plt.yscale('log')

# %% Some testing code for automatic trap region detection
# img = mean_list[0]
# from matplotlib.patches import Rectangle
# fig, ax = plt.subplots()
# plt.imshow(img)
# (x, y, w, h), *_ = find_rect(img)
# ax.add_patch(Rectangle((x, y), w, h, fill=True, color='red', alpha=0.4))
# plt.show()
