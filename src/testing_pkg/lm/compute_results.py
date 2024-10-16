from argparse import ArgumentParser
from testing_pkg.lm.evaluator_class import Evaluator


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename",
                        help="input file name (path in config)",
                        metavar="FILE", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # print(f'Evaluating test: \n - file: {args.filename} \n ')
    eval = Evaluator(args.filename)
    testname = eval.testparams['name']
    testype = eval.testparams['type']
    if 'ckpt' in args.filename:
        model = args.filename.split('.')[-1].split('-')[-1]
    else:
        # not a checkpoint
        model = args.filename.split('.')[-1]

    # print(f'Detected test name {eval.testname}, type {eval.testtype}')

    eval.evaluate()
    # print(f'Result file created: \n {eval.result_file}.')
    # print(f'Eval file created: \n {eval.eval_file}.')
    # print(f'Wrong file created: \n {eval.wrong_file}.')
    # print(f'Convergence graph png created: \n {eval.graph_file}.')

    if eval.accuracy2 != -1 or eval.accuracy3 != -1:
        print(f'{testname}_1, {testype}, {model}, {eval.accuracy1:.2f}')
    else:
        print(f'{testname}, {testype}, {model}, {eval.accuracy1:.2f}')

    if eval.accuracy2 != -1:
        print(f'{testname}_2, {testype}, {model}, {eval.accuracy2:.2f}')
    if eval.accuracy3 != -1:
        print(f'{testname}_3, {testype}, {model}, {eval.accuracy3:.2f}')
