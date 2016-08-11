import pandas as pd
import numpy as np
import theano


def main():

    controls_file = 'data/no_fallen_merged_controls.txt'
    states_file = 'data/no_fallen_merged_starting_states.txt'

    y = np.asarray(pd.read_csv(
        controls_file, delimiter=',',
        header=None).values, dtype=theano.config.floatX)

    x = np.asarray(pd.read_csv(
        states_file, delimiter=',',
        header=None).values, dtype=theano.config.floatX)

    """
    # DOWN

    idx_down = np.arange(x.shape[0])
    y_bone1 = x[:, 1] < -1.
    y_bone2 = x[:, 7] < -1.
    y_bones = np.logical_or(y_bone1, y_bone2)

    th_limits = np.array([[-2.4, -1.2], [-1.5, -0.9], [-2.5, -2], [-2., -1.4]])

    bone1 = np.logical_and(x[:, 4] > th_limits[0, 0],
                           x[:, 4] < th_limits[0, 1])
    bone2 = np.logical_and(x[:, 10] > th_limits[1, 0],
                           x[:, 10] < th_limits[1, 1])
    bone3 = np.logical_and(x[:, 16] > th_limits[2, 0],
                           x[:, 16] < th_limits[2, 1])
    bone4 = np.logical_and(x[:, 22] > th_limits[3, 0],
                           x[:, 22] < th_limits[3, 1])

    bone12 = np.logical_and(bone1, bone2)
    bone34 = np.logical_and(bone3, bone4)
    
    idx_down = idx_down[np.logical_and(bone12, bone34)]
    idx_down = idx_down[y_bones]
    x_down = x[idx_down]
    y_down = y[idx_down]
    """

    # Stand still
    th_limits = np.array([[-.4, .4], [-.4, .4], [-.3, .3], [-.3, .3]])

    idx_up = np.arange(x.shape[0])
    bone1 = np.logical_and(x[:, 4] > th_limits[0, 0],
                           x[:, 4] < th_limits[0, 1])
    bone2 = np.logical_and(x[:, 10] > th_limits[1, 0],
                           x[:, 10] < th_limits[1, 1])
    bone3 = np.logical_and(x[:, 16] > th_limits[2, 0],
                           x[:, 16] < th_limits[2, 1])
    bone4 = np.logical_and(x[:, 22] > th_limits[3, 0],
                           x[:, 22] < th_limits[3, 1])

    bone12 = np.logical_and(bone1, bone2)
    bone34 = np.logical_and(bone3, bone4)

    idx_up = idx_up[np.logical_and(bone12, bone34)]

    x_up = x[idx_up]
    y_up = y[idx_up]

    # In between
    idx_critical = np.arange(x.shape[0])
    idx_critical = np.setdiff1d(idx_critical, idx_up)
    x_critical = x[idx_critical]
    y_critical = y[idx_critical]

    np.savetxt('x_critical.txt', x_critical, delimiter=',', fmt='%.6f')
    np.savetxt('y_critical.txt', y_critical, delimiter=',', fmt='%.6f')

    np.savetxt('x_up.txt', x_up, delimiter=',', fmt='%.6f')
    np.savetxt('y_up.txt', y_up, delimiter=',', fmt='%.6f')


if __name__ == '__main__':
    main()
