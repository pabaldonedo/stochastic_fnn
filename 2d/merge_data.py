import re
import numpy as np


def raw_merged():
    n = [13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45, 51, 54]
    controls_fnames = ['data/raw_data/controls_{0}.txt'.format(i) for i in n]
    controls_ofname = 'data/merged_controls.txt'
    states_fnames = ['data/raw_data/starting_states_{0}.txt'.format(i) for i in n]
    states_ofname = 'data/merged_starting_states.txt'

    valid_line = re.compile("[0-9.\-]+")

    with open(controls_ofname, 'w') as of:
        for fname in controls_fnames:
            with open(fname, 'r') as f:
                for i, line in enumerate(f):
                    if not valid_line.match(line):
                        continue
                    of.write(re.sub(r"\s([0-9\-])", r",\1", line))
            print "File {0} processing done".format(fname)

    with open(states_ofname, 'w') as of:
        for fname in states_fnames:
            with open(fname, 'r') as f:
                for i, line in enumerate(f):
                    if not valid_line.match(line):
                        continue
                    of.write(re.sub(r"\s([0-9\-])", r",\1", line))
            print "File {0} processing done".format(fname)

def no_fallen_merged():
    n = [13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45, 51, 54]
    controls_fnames = ['data/raw_data/controls_{0}.txt'.format(i) for i in n]
    controls_ofname = 'data/no_fallen_merged_controls.txt'
    states_fnames = ['data/raw_data/starting_states_{0}.txt'.format(i) for i in n]
    states_ofname = 'data/no_fallen_merged_starting_states.txt'

    valid_line = re.compile("[0-9.\-]+")

    buffer_state = ""
    buffer_control = ""
    cleaned_state = True

    with open(controls_ofname, 'w') as cof:
        with open(states_ofname, 'w') as sof:
            for sfi, sfname in enumerate(states_fnames):
                with open(controls_fnames[sfi], 'r') as cf:
                    with open(sfname, 'r') as sf:
                        for i, sline in enumerate(sf):
                            cline = cf.readline()
                            if sline == 'Reset\r\n':
                                if cleaned_state:
                                    sof.write(buffer_state)
                                    cof.write(buffer_control)
                                buffer_state = ""
                                buffer_control = ""
                                cleaned_state = True
                                continue
                            if not valid_line.match(sline):
                                print "INVALID"
                                import ipdb
                                ipdb.set_trace()
                                continue

                            parsed_state = re.sub(r"\s([0-9\-])", r",\1", sline)
                            parsed_control = re.sub(r"\s([0-9\-])", r",\1", cline)
                            buffer_state += parsed_state
                            buffer_control += parsed_control

                            np_state = np.fromstring(parsed_state, sep=',')
                            cleaned_state =  cleaned_state and np.logical_and(np_state[1] > -1.,
                                                                                np_state[7] >-1.)

                    print "File {0} processing done".format(sfname)




def main():

    no_fallen_merged()


if __name__ == '__main__':
    main()
