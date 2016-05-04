import re
import argparse
import os


def parse_all_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fname", default=None)
    return parser.parse_args()

def main():

    args = parse_all_args()
    fname = args.fname
    name, extension = os.path.splitext(fname)
    ofname =  name + '_processing' + extension
    seq_len = -1
    n_lines = 0
    valid_line = re.compile("[0-9.\-]+")

    #number of blocks:
    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            if line == "\n" and seq_len == -1:
                seq_len = i
                continue
            if not valid_line.match(line):
                continue
            n_lines += 1
    n_sequences = n_lines / seq_len
    read_lines = 0
    with open(fname, 'r') as f:
        with open(ofname, 'w') as of:
            for i, line in enumerate(f):
                if line == "\n" and n_lines == -1:
                    n_lines = i
                    continue
                if not valid_line.match(line):
                    continue
                read_lines += 1
                if read_lines > n_sequences*seq_len:
                    break
                of.write(re.sub(r"\s([0-9\-])", r",\1", line))

    os.rename(ofname, name + '_len_{0}'.format(seq_len) + extension)

if __name__ == '__main__':
    main()

