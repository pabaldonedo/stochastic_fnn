import matplotlib.pyplot as plt
import numpy as np
import argparse
from types import IntType


def parse_all_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--train_file", default=None)
    parser.add_argument("-ftest", "--test_file", default=None)
    parser.add_argument("-d", "--dimensions", default=None, type=int)
    parser.add_argument("-m", "--distribution_samples", default=None, type=int)
    parser.add_argument("-n", "--training_samples", default=None, type=int)
    parser.add_argument("-ntest", "--test_samples", default=None, type=int)
    parser.add_argument("-ofile", "--output_file", default=None)
    return parser.parse_args()


def get_reg_value(m, n, dim=None):
    """
    returns the regularizer term for the log likelihood.
    Log likelihood can be decomposed in: unregularized term - constant. We compute the constant an return it
    """
    if dim is None:
        dim = 30

    return n * (np.log(m) + dim / 2. * np.log(2 * np.pi))


def main():
    # Loading parameters
    args = parse_all_args()
    fname = args.train_file
    assert fname is not None, "No file name provided"

    #dim = args.dimensions
    #assert type(dim) is IntType or dim is None, "Unrecognized type for dim value. It must be an integer. Provided: {0!r}".format(dim)

    #m = args.training_samples
    #assert type(m) is IntType, "Unrecognized type for m value. It must be an integer. Provided: {0!r}".format(m)

    n = args.training_samples
    assert type(
        n) is IntType or n is None, "Unrecognized type for n value. It must be an integer. Provided: {0!r}".format(n)

    ofile = args.output_file
    test_fname = args.test_file

    ntest = args.test_samples
    if test_fname is not None:
        assert type(
            ntest) is IntType or ntest is None, "Unrecognized type for ntest value. It must be an integer. Provided: {0!r}".format(ntest)

    fig, ax = plt.subplots(figsize=(15, 10))
    colors = ['b', 'r']

    # Train log likelihood ocmputations
    #train_reg_value = get_reg_value(m, n, dim=dim)
    train_log_likelihood = np.genfromtxt(fname, delimiter=',')
    #train_log_likelihood[:,1] -= train_reg_value

    ax.plot(train_log_likelihood[:, 0],
            train_log_likelihood[:, 1], color=colors[0])

    # For showing mean log likelihood values in the right y-axis
    if n is not None:
        ax2 = ax.twinx()
        ax2.plot(train_log_likelihood[:, 0], train_log_likelihood[
                 :, 1] * 1. / n, color=colors[0], alpha=0)

    # If test available plot also test log likelihood
    if test_fname is not None:
        #        test_reg_value = get_reg_value(m, ntest, dim=dim)
        test_log_likelihood = np.genfromtxt(test_fname, delimiter=',')
       # test_log_likelihood[:, 1] -= test_reg_value
        ax.plot(test_log_likelihood[:, 0],
                test_log_likelihood[:, 1], color=colors[1])
        if ntest is not None:
            ax2.plot(test_log_likelihood[:, 0], test_log_likelihood[
                     :, 1] * 1. / ntest, color=colors[1], alpha=0)
        legend_names = ['train', 'test']
    else:
        legend_names = ['train']

    ax.legend(legend_names, loc=4)
    ax.set_xlabel('epochs')
    ax.set_ylabel('log_likelihood')
    if n is not None:
        ax2.set_ylabel('Mean log_likelihood')

    # Show or save to file
    if ofile is None:
        plt.show()
    else:
        plt.savefig(ofile)


if __name__ == '__main__':
    main()
