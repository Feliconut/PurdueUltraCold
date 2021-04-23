# %%
from .model import define_discriminator, define_generator, define_gan, load_real_samples, pyplot, train

# %% load image data
from os.path import join, dirname, exists
from os import chdir, mkdir
chdir(join(dirname(__file__)))
if not exists('data'):
    mkdir('data')
if not exists('result'):
    mkdir('result')
src, tar, ids = load_real_samples('data/data128.npz')
assert len(src) == len(tar) == len(ids)
image_shape = src.shape
src, tar = src.reshape((*image_shape, 1)), tar.reshape((*image_shape, 1))

train_ratio = 0.9
dataset_size = len(src)
train_size = int(train_ratio * dataset_size)

src_train = src[:train_size]
tar_train = tar[:train_size]
ids_train = ids[:train_size]
src_valid = src[train_size:]
tar_valid = tar[train_size:]
ids_valid = ids[train_size:]

# %%
from random import shuffle
srctar = list(zip(src_train, tar_train, ids_train))
shuffle(srctar)
src_train, tar_train, ids_train = list(zip(*srctar))
import numpy as np
src_train = np.array(src_train)
tar_train = np.array(tar_train)
ids_train = np.array(ids_train)
#[:500] # training data
# validate_dataset = dataset[500:] # validation data
print('Loaded', src_train.shape, tar_train.shape)
# define input shape based on the loaded dataset
image_shape = src_train.shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# %% train model
dl1, dl2, gl = train(d_model,
                     g_model,
                     gan_model, (src_train, tar_train),
                     n_epochs=4,
                     n_batch=2)
np.savez('loss', dl1=dl1, dl2=dl2, gl=gl)
print('>Saved: loss.npz')
# %%
pyplot.plot(gl)
pyplot.title('generator training loss')
pyplot.savefig('result/loss.png')
# %%
# summarize_performance(2512, g_model, (src, tar))
# g_model.save('model_2512.h5')
# %%
res = g_model.predict(src_valid)
# %%
np.savez_compressed('valid_result',src_valid,res,tar_valid,ids_valid)
