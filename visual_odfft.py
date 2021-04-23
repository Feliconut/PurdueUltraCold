# %%
from UltraCold import OD
from numpy.fft import fft2, fftshift
from numpy import abs
import numpy as np
import matplotlib.pyplot as plt


# %%
def odfft(od):
    return abs(fftshift(fft2(fftshift(od))))


# %%
def odhist(ods, title=None):
    all_noise = np.array(ods).flatten()
    from scipy.stats import norm
    mu, std = norm.fit(all_noise)
    plt.hist(all_noise, density=True, bins=200)
    # xm, xM = plt.xlim()
    # x = np.linspace(xm+0.8, xM-1, 100)
    # plt.plot(x, norm.pdf(x, mu, std), 'k', linewidth=2)
    plt.title(title if title else f'{mu:.2e},{std:.2e}')
    plt.yscale('log')
    # plt.show()


# %% MANUAL ADJUST OF MAXVAL
maxval = 100
for od, dataset_id, i_in_dataset in OD.iter_through_dir(mode='flat'):
    try:
        maxval = float(input())
    except KeyboardInterrupt as e:
        # e.with_traceback()
        break
    except ValueError:
        break
    plt.imshow(odfft(od), vmax=maxval)
    plt.title(f'id{dataset_id}#{i_in_dataset}, MAXVAL={maxval}')
    plt.colorbar()
    # plt.savefig(f'out/id{dataset_id}.{i_in_dataset}-MAX{maxval}.png')
    plt.show()
# %% GENERATE FFT_OD_HSIT
odhist([odfft(od) for od in OD.iter_through_dir(mode='group').__next__()[0]],
       title='FFT(OD) Intensity distribution of dataset 051649')
plt.savefig(f'out/odhist.png')
plt.show()

# %% GENERATE FFT_OD_MEAN
# maxval = 30
# for ods, dataset_id in OD.iter_through_dir(mode='group'):
#     fftavg = np.mean([odfft(od) for od in ods],axis=0)

#     plt.imshow(fftavg, vmax=maxval)
#     plt.title(f'id{dataset_id}#, MAXVAL={maxval}, MEAN')
#     plt.colorbar()
#     plt.savefig(f'out/id{dataset_id}-MAX{maxval}-MEAN.png')
#     plt.show()
#     input()
# %% SET VMAX using PERCENTILE
def get_vmax(img, q):
    return np.percentile(img.flatten(), q)

# %% MANUAL ADJUST OF quantile
q = 99.65
MANUAL = True
for ods, dataset_id in OD.iter_through_dir(mode='group'):
    fftavg = np.mean([odfft(od) for od in ods],axis=0)
    if MANUAL:
        try:
            q = float(input())
        except KeyboardInterrupt as e:
            # e.with_traceback()
            break
        except ValueError:
            break
    maxval = get_vmax(fftavg,q)
    plt.imshow(od)
    plt.imshow(fftavg, vmax=maxval)
    plt.title(f'id{dataset_id}#, q={q}, FFT-MEAN')
    plt.colorbar()
    plt.savefig(f'out/id{dataset_id}-q{q}-FFT-MEAN.png')
    plt.show()
# %%
# FIND THRESH BY MAXIMIZING INFORMATION

