# %%
from matplotlib.pyplot import plot
from UltraCold import MTF, plot_od_avg, read_image

# %%
img_dir = './DATA/211412'

plot_od_avg(img_dir)

# %%

OD_data, img_index_range = read_image(img_dir)

# %%
MTF.from_od(OD_data)

# %%
