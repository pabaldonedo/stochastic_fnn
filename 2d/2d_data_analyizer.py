import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano


def controls_hist(y, labels):

    if not os.path.exists('data_plots'):
        os.makedirs('data_plots')

    fig, axs = plt.subplots(1,3, figsize=(30,10))
    ax = axs.flatten()
    for i in xrange(y.shape[1]):
        ax[i].hist(y[:,i], bins=100, normed=True)
        ax[i].set_xlabel(labels[i])

    plt.savefig('data_plots/controls_hist.png')


def states_hist(x, labels):

    if not os.path.exists('data_plots'):
        os.makedirs('data_plots')

    fig, axs = plt.subplots(5,5, figsize=(30,30))
    ax = axs.flatten()
    for i in xrange(x.shape[1]):
        ax[i].hist(x[:,i], bins=100, normed=True)
        ax[i].set_xlabel(labels[i])

    plt.savefig('data_plots/states_hist.png')


def state_scatter(sampled_x, labels, path):

    if not os.path.exists(path):
        os.makedirs(path)

    fig, ax = plt.subplots(6,5, figsize=(50,50))
    ax = ax.flatten()     
    k = 0
    i = 1
    for ci, cname in enumerate(labels):
        for j, cname2 in enumerate(labels[ci+1:]):
            cj = ci+1+j
            ax[k].scatter(sampled_x[:, ci], sampled_x[:, cj], color='b')
            ax[k].set_xlabel(str(cname), fontsize=15)
            ax[k].set_ylabel(str(cname2), fontsize=15)
            k += 1
            if k%30 == 0:
                plt.savefig('{0}/{1}.png'.format(path, i))
                k = 0
                fig, ax = plt.subplots(6,5, figsize=(50,50))
                ax = ax.flatten()
                i += 1

    plt.savefig('{0}/{1}.png'.format(path, i))


def control_scatter(sampled_y, labels, fname):

    fig, ax = plt.subplots(1,3, figsize=(30,10))
    ax = ax.flatten()     
    k = 0
    for ci, cname in enumerate(labels):
        for j, cname2 in enumerate(labels[ci+1:]):
            cj = ci+1+j
            ax[k].scatter(sampled_y[:, ci], sampled_y[:, cj], color='b')
            ax[k].set_xlabel(str(cname), fontsize=15)
            ax[k].set_ylabel(str(cname2), fontsize=15)
            k += 1

    plt.savefig('{0}.png'.format(fname))

def check_multimodality(x,y):

    n_bins = 20
    x_binned = np.empty(x.shape)
    bins = np.empty((n_bins+1, x.shape[1]))
    for i in xrange(x.shape[1]):
        _, bins[:,i] = np.histogram(x[:,i], bins=n_bins)
        bins[-1,i] = bins[-1,i]+0.01
        centers = 0.5*(bins[:-1, i] + bins[1:, i])
        x_binned[:,i] = centers[np.digitize(x[:,i], bins[:,i]) -1]

    x_transform = np.ascontiguousarray(x_binned).view(np.dtype((np.void, x_binned.dtype.itemsize * x_binned.shape[1])))
    _, idx = np.unique(x_transform, return_index=True)
    x_unique = x_binned[idx]

    #Too few samples! :(

def generate_datset_sample(x, sample_size, fname):

    idx = np.arange(x.shape[0])
    idx = np.random.permutation(idx)
    random_x = x[idx]
    np.savetxt(fname, random_x[:sample_size], fmt='%10.5f', delimiter=',')

def main():
    
    y = np.asarray(pd.read_csv("data/merged_controls.txt",
                               delimiter=',', header=None).values, dtype=theano.config.floatX)

    x = np.asarray(pd.read_csv("data/merged_starting_states.txt",
                               delimiter=',', header=None).values, dtype=theano.config.floatX)


    y_labels = ['control1', 'control2', 'control3']
    x_labels = ['state{0}'.format(i) for i in xrange(x.shape[1])]

    states_hist(x, x_labels)
    controls_hist(y, y_labels)

    random_sample = np.random.permutation(x.shape[0])
    sample_size = 100000
    state_scatter_path = 'data_plots/state_scatter'
    state_scatter(x[:sample_size, :], x_labels, state_scatter_path)

    control_scatter_name = 'data_plots/control_scatter.png'
    control_scatter(y[:sample_size, :], y_labels, control_scatter_name)

if __name__ == '__main__':
    main()