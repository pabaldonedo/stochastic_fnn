import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cmx
import matplotlib.colors as colors
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

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def bone_evolution_random_sample(x, n_show_bones,seq_len, bone_features, bone_names, path):
    bone_idx = np.random.randint(x.shape[0]/seq_len, size=n_show_bones)
    cmap = get_cmap(len(bone_features))

    for b, bname in enumerate(bone_names):
        fig, ax = plt.subplots(len(bone_features), 1, figsize=(30,20))
        for fi, f in enumerate(bone_features):
            for bi, bidx in enumerate(bone_idx):
                col = cmap(bi)
                ax[fi].plot(np.arange(seq_len), x[bidx*seq_len:bidx*seq_len+seq_len, b*len(bone_features)+fi], color=col)

        plt.savefig('{0}/{1}.png'.format(path, bname))

def bone_evolution_mean(x, seq_len, bone_features, bone_names, path):
    cmap = get_cmap(len(bone_features))

    mux = np.zeros((seq_len, x.shape[1]))
    stdx = np.zeros((seq_len, x.shape[1]))
    minx = np.zeros((seq_len, x.shape[1]))
    maxx = np.zeros((seq_len, x.shape[1]))
    for s in xrange(seq_len):
        mux[s] = np.mean(x[s::seq_len], axis=0)
        stdx[s] = np.std(x[s::seq_len], axis=0)
        minx[s] = np.min(x[s::seq_len], axis=0)
        maxx[s] = np.max(x[s::seq_len], axis=0)
    
    
    for b, bname in enumerate(bone_names):
        fig, ax = plt.subplots(len(bone_features), 1, figsize=(60,50))
        for fi, f in enumerate(bone_features):
            ax[fi].errorbar(np.arange(seq_len), mux[:, b*len(bone_features)+fi], yerr=stdx[:, b*len(bone_features)+fi].T, color='b', ecolor='b')
            ax[fi].errorbar(np.arange(seq_len), mux[:, b*len(bone_features)+fi], yerr=np.vstack((-1*minx[:, b*len(bone_features)+fi]+mux[:, b*len(bone_features)+fi], maxx[:, b*len(bone_features)+fi]-mux[:, b*len(bone_features)+fi])), ecolor='r', color='b')
            ax[fi].set_title(f)
        plt.savefig('{0}/{1}.png'.format(path, bname))
        
def bone_evolution_scatter(x, seq_len, bone_features, bone_names, path):
    
    for b, bname in enumerate(bone_names):

        for fi, f in enumerate(bone_features):
            fig, ax = plt.subplots(5,6, figsize=(60,60))
            ax = ax.flatten()            
            for s in xrange(int(np.floor(seq_len/2))):
                ax[s].scatter(x[2*s::seq_len, b*len(bone_features) + fi], x[2*s+1::seq_len, b*len(bone_features) + fi], color='b')
                ax[s].set_xlabel(str(2*s))
                ax[s].set_ylabel(str(2*s+1))
                print "ONE SCATTER DONE"
            plt.savefig('{0}/{1}_{2}.png'.format(path, bname, str(f)))
            print "DONE"

def bone_scatter(x,bone_features, bone_names, path):
    
    for b, bname in enumerate(bone_names):
        k = 0
        fig, ax = plt.subplots(10,10, figsize=(50,50))
        ax = ax.flatten()      
        for fi, f in enumerate(bone_features):
            for j, f2 in enumerate(bone_features[fi+1:]):
                fj = fi+1+j
                ax[k].scatter(x[:,b*len(bone_names)+fi], x[:,b*len(bone_names)+fj], color='b')
                ax[k].set_xlabel(str(f), fontsize=15)
                ax[k].set_ylabel(str(f2), fontsize=15)
                k += 1
                print "ONE SCATTER DONE"
        plt.savefig('{0}/{1}.png'.format(path, bname))
        print "DONE"


def control_evolution_scatter(x, seq_len, path):
    
        for i in xrange(x.shape[1]):
            fig, ax = plt.subplots(5,6, figsize=(60,60))
            ax = ax.flatten()            
            for s in xrange(int(np.floor(seq_len/2))):
                ax[s].scatter(x[2*s::seq_len, i], x[2*s+1::seq_len, i], color='b')
                ax[s].set_xlabel(str(2*s))
                ax[s].set_ylabel(str(2*s+1))
                print "ONE SCATTER DONE"
            plt.savefig('{0}/control_{1}.png'.format(path, i))
            print "DONE"

def main():

    n = 7

    x = load_states(n)
    #y = load_controls(n)
    print "DATA LOADED"
    #x = np.random.randn(61*20,197)
    bone_features = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'r0', 'r1', 'r2', 'r3', 'wx', 'wy', 'wz']
    names = bone_features*15 +\
            ['left foot', 'right foot']
    bone_names = ['Hips', 'Spine1', 'Neck', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg',
                'RightLeg', 'RightFoot', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm',
                'RightForeArm', 'RightHand', 'Contact To Ground']

    #boxplot(x, step=len(bone_features), names=names, bone_names=bone_names, path='tmp/states')
    #boxplot(y, path='tmp/controls')
    # bone_evolution_mean(x, 61,  bone_features, bone_names[:-1], 'bone_plots/sequence_mean')
    #bone_evolution_scatter(x, 61, bone_features, bone_names[:-1], 'bone_plots/sequence_evolution')
    #control_evolution_scatter(y, 61, 'bone_plots/sequence_evolution')
    bone_scatter(x, bone_features, bone_names, 'bone_plots/bone_scatter')

if __name__ == '__main__':
    main()
