from util import load_states
from util import load_controls
from util import load_files
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    n = 16
    n_impulse_2000 = 5

    y = load_controls(n)
    if n_impulse_2000 > 0:
        y_impulse = load_files(n_impulse_2000, 'controls_impulse_2000')
        y = np.vstack((y, y_impulse))

    x = load_states(n)
    if n_impulse_2000 > 0:
        x_impulse = load_files(n_impulse_2000, 'states_impulse_2000')
        x = np.vstack((x, x_impulse))


def clipper(y):
    clipped_y = np.empty(y.shape)
    for i, yi in enumerate(y):
        np.clip(yi[:6], -yi[-3], yi[-3], out=clipped_y[i, :6])
        np.clip(yi[6:16], -yi[-4], yi[-4], out=clipped_y[i, 6:16])
        np.clip(yi[16:23], -yi[-2], yi[-2], out=clipped_y[i, 16:23])
        np.clip(yi[23:30], -yi[-1], yi[-1], out=clipped_y[i, 23:30])
    clipped_y[:, -4:] = y[:, -4:]

    _, bins = np.histogram(np.max(np.abs(y), axis=1), bins=100)
    plt.hist(np.max(np.abs(y), axis=1), bins=bins, color='b', alpha=0.8)
    plt.hist(np.max(np.abs(clipped_y), axis=1),
             bins=bins, color='r', alpha=0.8)

    #np.savetxt('clipped_controls_n_16_n_impulse_2000_5.txt', clipped_y, delimiter=',', fmt='%.7f')


def equalizer(x, y):
    n_bins = 100
    max_y = np.max(np.abs(y[:, :-4]), axis=1)
    _, bins = np.histogram(max_y, bins=n_bins)
    bins[-1] += 1

    idx_y = np.digitize(max_y, bins) - 1
    assert np.min(idx_y) == 0
    assert np.max(idx_y) == n_bins - 1
    bin_dict = {k: [] for k in xrange(n_bins)}
    for i, yi in enumerate(idx_y):
        bin_dict[yi].append(i)

    y_new = np.zeros(y.shape)
    x_new = np.zeros(x.shape)
    sample = np.random.randint(0, n_bins, size=x.shape[0])
    for i, s in enumerate(sample):
        idx = np.random.choice(bin_dict[s])
        x_new[i, :] = x[idx]
        y_new[i, :] = y[idx]
    return x_new, y_new
