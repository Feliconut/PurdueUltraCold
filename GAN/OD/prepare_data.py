# %%
from os import getcwd, listdir
from os.path import dirname, join
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from UltraCold import OD, NPS

# %%
# Modify this according to file position
baseDir = dirname(__file__)
DATA_DIR = join(baseDir, 'DATA')

# %%
src_list, tar_list = [], []

dataRegion = (slice(0, 256), slice(0, 256))

for dataset_id in listdir(DATA_DIR):
    if '.' in dataset_id: continue
    print(dataset_id)
    dataset_path = join(DATA_DIR, dataset_id)
    ods, _, _ = OD.from_image_dir(dataset_path)
    ods = [od[dataRegion] for od in ods]
    od_mean = np.mean(ods,axis=0)
    for od in ods:
        # (np.c_[od, od_mean])
        src_list.append(od * 100)
        tar_list.append(od_mean * 100)

# %%
print('start saving')
np.savez_compressed('data256_x100',src_list,tar_list)
    
# %%
