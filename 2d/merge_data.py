import re


def main():

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


if __name__ == '__main__':
    main()
