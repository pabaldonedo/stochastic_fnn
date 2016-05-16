import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Polygon

def load_states(n):

    x = np.genfromtxt("data/states_1_len_61.txt", delimiter=',')
    for i in xrange(2, n+1):
        tmp = np.genfromtxt("data/states_{0}_len_61.txt".format(i), delimiter=',')
        x = np.vstack((x, tmp))
    return x

def load_controls(n):

    x = np.genfromtxt("data/controls_1_len_61.txt", delimiter=',')
    for i in xrange(2, n+1):
        tmp = np.genfromtxt("data/controls_{0}_len_61.txt".format(i), delimiter=',')
        x = np.vstack((x, tmp))
    return x

def boxplot(x, step=None, names=None, bone_names=None, path="./"):
    if step is None:
        step = x.shape[1]
    for i in xrange(int(np.ceil(x.shape[1]*1./step))):
        plt.figure()
        fig, ax = plt.subplots()

        bp = plt.boxplot(x[:,step*i:step*(i+1)])
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='red', marker='+')

        # Add a horizontal grid to the plot, but make it very light in color
        # so we can use it for reading data values but not be distracting
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                       alpha=0.5)
        # Hide these grid behind plot objects
        ax.set_axisbelow(True)
        if bone_names is not None:
            ax.set_title('State vector for bone {0}'.format(bone_names[i]))
        ax.set_xlabel('Variable')
        ax.set_ylabel('Value')

        if names is not None:
            xtickNames = plt.setp(ax, xticklabels=names[step*i:step*(i+1)])
        numBoxes = len(bp['boxes'])
        medians = list(range(numBoxes))
        #plt.setp(xtickNames, rotation=45, fontsize=8)
        means = np.mean(x[:,step*i:step*(i+1)], axis=0)

        for j, box in enumerate(bp['boxes']):
            boxX = []
            boxY = []
            for k in range(5):
                boxX.append(box.get_xdata()[k])
                boxY.append(box.get_ydata()[k])
            boxCoords = list(zip(boxX, boxY))
            
            boxPolygon = Polygon(boxCoords, facecolor='royalblue')
            ax.add_patch(boxPolygon)
            # Now draw the median lines back over what we just filled in
            med = bp['medians'][j]
            medianX = []
            medianY = []

            for k in range(2):
                medianX.append(med.get_xdata()[k])
                medianY.append(med.get_ydata()[k])
                plt.plot(medianX, medianY, 'r')
                medians[j] = medianY[0]
            # Finally, overplot the sample averages, with horizontal alignment
            # in the center of each box
            plt.plot([np.mean(med.get_xdata())], means[j],
                     color='darkkhaki', marker='*', markeredgecolor='darkkhaki')



        # Due to the Y-axis scale being different across samples, it can be
        # hard to compare differences in medians across the samples. Add upper
        # X-axis tick labels with the sample medians to aid in comparison
        # (just use two decimal places of precision)
        bottom, top = ax.axes.get_ylim()
        pos = np.arange(numBoxes) + 1
        upperLabels = [str(np.round(s, 2)) for s in medians]
        upperLabelsMean = [str(np.round(s, 2)) for s in means]

        for tick, label in zip(range(numBoxes), ax.get_xticklabels()):
         
            ax.text(pos[tick], top - (top*0.05), upperLabels[tick],
                     horizontalalignment='center', size='x-small', weight='bold',
                     color='r')
            ax.text(pos[tick], top - (top*0.1), upperLabelsMean[tick],
                     horizontalalignment='center', size='x-small', weight='bold',
                     color='darkkhaki')


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        # Finally, add a basic legend
        plt.figtext(0.1, 0.015, '*', color='darkkhaki',
                    weight='roman', size='medium')
        plt.figtext(0.115, 0.013, ' Average Value', color='black', weight='roman',
                    size='x-small')

        fig.savefig('{0}/{1}'.format(path, i))


def main():

    n = 7
    x = load_states(n)
    y = load_controls(n)
    bone_features = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'r0', 'r1', 'r2', 'r3', 'wx', 'wy', 'wz']
    names = bone_features*15 +\
            ['left foot', 'right foot']
    bone_names = ['Hips', 'Spine1', 'Neck', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg',
                'RightLeg', 'RightFoot', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm',
                'RightForeArm', 'RightHand', 'Contact To Ground']

    boxplot(x, step=len(bone_features), names=names, bone_names=bone_names, path='tmp/states')
    boxplot(y, path='tmp/controls')
    


if __name__ == '__main__':
    main()