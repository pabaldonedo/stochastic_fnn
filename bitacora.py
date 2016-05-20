import logging
import argparse


def parse_all_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fname", default='cuaderno_de_bitacora.txt')
    parser.add_argument("-i", "--info", default=False, action='store_true')
    parser.add_argument("-w", "--warning", default=False, action='store_true')
    parser.add_argument("-c", "--critical", default=False, action='store_true')
    parser.add_argument("-m", "--message", default=None)
    return parser.parse_args()


def main():
    args = parse_all_args()
    fname = args.fname
    info = args.info
    warning = args.warning
    critical = args.critical
    message = args.message

    if info:
        lvl = logging.INFO
    if warning:
        lvl = logging.WARNING
    if critical:
        lvl = logging.CRITICAL

    if not (info or warning or critical):
        lvl = logging.INFO

    logging.basicConfig(level=logging.INFO, filename=fname,
                    format="%(asctime)-15s %(levelname)-8s %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S")

    logging.log(lvl, message)

if __name__ == '__main__':
    main()