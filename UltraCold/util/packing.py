'''A simple shelf-based algorithm for 
packing sample regions into OD image'''

# %%
from itertools import product


def pack(OD_size, noise_size, trap_rect, only_pos=False):
    """
    @brief pack noise regions into OD image, avoiding intersection.
    No intersection among noise regions, or between noise regions and the trap region.
    The current algorithm is crude and not optimal for rectangular noise regions.
    
    @param OD_size (`x size`, `y size`), whre `x size` and `y size` are `int`. Denotes size of OD image.

    @param noise_size similar to OD_size. Denotes size of all noise regions.

    @param trap_rect (`x size`,`y size`, `x spread (width)`, `y spread(height)`). Denotes the trap region.

    @param only_pos `boolean` controls the return value.

    @return list of trap region slicers if `only_pos` is False. list of trap region positions otherwise.
    """
    X, Y = OD_size
    x0, y0 = noise_size
    x, y, dx, dy = trap_rect

    noise_pos = []

    def shelf(xm, ym, xM, yM):
        xs = list(range(xm, xM - x0, x0))
        ys = list(range(ym, yM - y0, y0))
        if xs and ys:
            noise_pos.extend(list(product(xs, ys)))
            return True
        else:
            return False

    if shelf(0, 0, x, Y):
        if shelf(x + dx, 0, X, Y):
            shelf(x, 0, x + dx, y)
            shelf(x, y + dy, x + dx, Y)
        else:
            shelf(x, y + dy, X, Y)
            shelf(x, 0, X, y)
    elif shelf(0, 0, X, y):
        if shelf(0, y + dy, X, Y):
            shelf(x + dx, y, X, y + dy)
        else:
            shelf(x + dx, y, X, Y)
    elif shelf(0, y + dy, X, Y):
        shelf(x + dx, 0, 0, y + dy)
    else:
        shelf(x + dx, 0, X, Y)

    if only_pos:
        return noise_pos
    else:
        return [(slice(x, x + x0), slice(y, y + y0)) for x, y in noise_pos]


# %%
def pack_and_visualize(OD_size, noise_size, trap_rect, noise_pos=[]):
    if not noise_pos:
        noise_pos = pack(OD_size, noise_size, trap_rect, only_pos=True)
    X, Y = OD_size
    x0, y0 = noise_size
    x, y, dx, dy = trap_rect

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)

    plt.xlim(0, X)
    plt.ylim(0, Y)

    ax.add_patch(Rectangle((x, y), dx, dy, color='grey', fill=True, alpha=0.8))

    for xp, yp in noise_pos:
        ax.add_patch(
            Rectangle((xp, yp), x0, y0, color='red', fill=True, alpha=0.5))

    plt.show()
    print(noise_pos)
    return noise_pos


# %% TESTING
# pack_and_visualize((300, 300), (64, 64), (100, 100, 120, 130))
# %%
