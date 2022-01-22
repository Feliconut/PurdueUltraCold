# %%
from importlib import reload
from GAN.OD_FFT_2.model import define_generator, weight_path
import numpy as np
import os
import perform_fitting
reload(perform_fitting)
g = define_generator((128, 128, 1))

weight_paths = [p for p in os.listdir(weight_path) if p.endswith('h5')]
w = weight_paths[-1]
# %%
g.load_weights(weight_path + '/' + w)

# src, pred, tar = np.load(r'GAN\OD_FFT_1\result_valid.npz')
# %%
src, res, tar, ids = np.load(r'GAN\OD_FFT_2\valid_result.npz').values()
# %%
import matplotlib.pyplot as plt
plt.imshow(res[10])
# %%
i = 40
print(f'fitting result for image from {ids[i]}')
perform_fitting.fit(res[i].reshape((128, 128)))
# %%
i = 40
print(f'classical fitting result for image from {ids[i]}')
perform_fitting.fit(tar[i].reshape((128, 128)))
# %%
plt.imshow(res[i])
# %%
plt.imshow(src[i])
# %%
plt.imshow(tar[i])
# %%
