from argparse import ArgumentParser
from test_creation import TestCreation


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename",
                        help="input file name (WITHOUT PATH)",
                        metavar="FILE", type=str)
    parser.add_argument("-t", "--test", dest="testnumber", default='T0',
                        help="number of the test to create as desc in config",
                        type=str, metavar="TEST")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    print(f'Creating test: \n - number: {args.testnumber}')
    print(f' - input file: {args.filename}')
    testcreation = TestCreation()
    test_file = testcreation.create_test(args.testnumber, args.filename)
    print(f'Test file created test type {args.testnumber}: \n {test_file}.')
