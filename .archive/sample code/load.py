# %%
from img_load import load_bimg
import numpy as np
import matplotlib.pyplot as plt

folderpath = './211412/'
pi = 3.1415926

psat=277
# %%
# intensity at each pixel
result=load_bimg(folderpath, raw=True)
#od=np.mean((-np.log(result['raw1']/result['raw2'])-(result['raw1']-result['raw2'])/psat),axis=0)
# Optical Density 
odimg=-np.log(result['raw1']/result['raw2'])-(result['raw1']-result['raw2'])/psat
np.save('od_image', odimg)
# %%
fig, ax = plt.subplots()
im = ax.imshow(odimg[1])
plt.colorbar(im)
plt.show()
plt.savefig('OD.png') # First optical density graph
# %%
sc = 1 # (temporary) structural factor: pixel area divided by resonance cross section
xc = 150
yc = 150
#3 dim. (index, x, y)
nimg = odimg*sc
nmean = np.mean(nimg,axis=0) # calcualte the average number density distribution
Ntot = np.sum(nmean)
# nimg - nmean: each image is substracted by the mean.
fftimg = np.mean(np.roll(np.roll((np.abs(np.fft.fft2(nimg-nmean))**2),xc,axis=1),yc,axis=2),axis=0)/Ntot
np.save('fft_image', fftimg)

plt.close()
fig, ax = plt.subplots()
im = ax.imshow(fftimg, vmax = 0.3, extent = [0, 2 * pi, 0, 2 *  pi])#
plt.colorbar(im)
plt.savefig('fft.png')