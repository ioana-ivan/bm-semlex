from numpy import mean
from testing_pkg.test.test_creation import TestCreation, TestType, TestRecord
import codecs
import yaml
from datetime import datetime
from itertools import zip_longest


class TestRecordResult(TestRecord):
    def __init__(self, *args):

        if len(args) == 1:  # list
            super().__init__(args[0][:-1])
            self.score = float(args[0][-1])
        else:
            super().__init__(args[:-1])
            self.score = float(args[-1])

    def __str__(self):
        return f"{super().__str__()}\t{self.score}"


class Evaluator:
    def __init__(self, file_to_eval):
        self.file_to_eval = file_to_eval
        self.configfile = "src//testing_pkg//lm//eval_config.yaml"
        self.config_dict = {}
        self.lines_except_neg = 2
        self.default_neg = 1
        self.testparams = {}

        self.eval_list1 = []
        self.eval_list2 = []
        self.eval_list3 = []

        self.wrong_list1 = []
        self.wrong_list2 = []
        self.wrong_list3 = []

        self.delta_list1 = []
        self.delta_list2 = []
        self.delta_list3 = []

        self.accuracy1 = -1
        self.accuracy2 = -1
        self.accuracy3 = -1

        self.msg1 = ''
        self.msg2 = ''
        self.msg3 = ''

        # pre-processing
        self.load_config()
        self.__load_output_filenames()
        self.__extract_test_type()

    def __extract_test_type(self):
        testname = self.file_to_eval.split("_")[0]

        testcreation = TestCreation()
        testcreation.load_test_params_from_config(testname)
        self.testparams = testcreation.test_parameters

    def __get_nb_neg(self):
        """
        Determines the number of negative examples in the test.

        Returns:
            int: The number of negative examples.
        """
        return self.testparams.get('nb_neg', self.default_neg)

    def evaluate(self):
        nb_neg = self.__get_nb_neg()
        nb_lines_test = nb_neg + self.lines_except_neg

        testtype = TestType(int(self.testparams['type']))

        eval_methods = {
            TestType.type_sub: self.eval_type1,
            TestType.type_relc: self.eval_type1,
            TestType.type_ref: self.eval_type1,
            TestType.type_relnc_rand: self.eval_type5,
            TestType.type_relc_rand: self.eval_type6,
            # TestType.type_relc_rand: self.eval_type9,
            TestType.type_sub_rand: self.eval_type6,
            # TestType.type_sub_rand: self.eval_type9,
        }
        if testtype in eval_methods:
            eval_methods[testtype](nb_lines_test)
        else:
            raise Exception(f"Test type {testtype} not implemented")

        self.__write_stats()

    def load_config(self):
        with codecs.open(self.configfile, encoding='utf-8', mode='r') as f:
            self.config_dict = yaml.safe_load(f)

    def __load_output_filenames(self):
        wrong_dir = self.config_dict['paths']['wrong_dir']
        eval_dir = self.config_dict['paths']['eval_dir']
        result_dir = self.config_dict['paths']['result_dir']
        delta_dir = self.config_dict['paths']['delta_dir']

        self.wrong_file1 = wrong_dir + self.file_to_eval + self.config_dict['suffixes']['wrong1']
        self.wrong_file2 = wrong_dir + self.file_to_eval + self.config_dict['suffixes']['wrong2']
        self.wrong_file3 = wrong_dir + self.file_to_eval + self.config_dict['suffixes']['wrong3']

        self.result_file1 = result_dir + self.file_to_eval + self.config_dict['suffixes']['result1']
        self.result_file2 = result_dir + self.file_to_eval + self.config_dict['suffixes']['result2']
        self.result_file3 = result_dir + self.file_to_eval + self.config_dict['suffixes']['result3']

        self.eval_file1 = eval_dir + self.file_to_eval + self.config_dict['suffixes']['eval1']
        self.eval_file2 = eval_dir + self.file_to_eval + self.config_dict['suffixes']['eval2']
        self.eval_file3 = eval_dir + self.file_to_eval + self.config_dict['suffixes']['eval3']

        self.eval_file1 = eval_dir + self.file_to_eval + self.config_dict['suffixes']['eval1']
        self.eval_file2 = eval_dir + self.file_to_eval + self.config_dict['suffixes']['eval2']
        self.eval_file3 = eval_dir + self.file_to_eval + self.config_dict['suffixes']['eval3']

        self.delta_file1 = delta_dir + self.file_to_eval + self.config_dict['suffixes']['delta1']
        self.delta_file2 = delta_dir + self.file_to_eval + self.config_dict['suffixes']['delta2']
        self.delta_file3 = delta_dir + self.file_to_eval + self.config_dict['suffixes']['delta3']

        for file in [self.wrong_file1, self.wrong_file2, self.wrong_file3,
                     self.eval_file1, self.eval_file2, self.eval_file3,
                     self.delta_file1, self.delta_file2, self.delta_file3,
                     self.result_file1, self.result_file2, self.result_file3]:
            with codecs.open(file, encoding='utf-8', mode='w') as wfile:
                wfile.write('')

    def __write_to_delta(self, delta_list, delta_file):
        with codecs.open(delta_file, encoding='utf-8', mode='w') as dfile:
            for delta in delta_list:
                dfile.write(f"{delta}\n")

    def __write_to_eval(self, eval_list, eval_file):
        with codecs.open(eval_file, encoding='utf-8', mode='w') as efile:
            for result in eval_list:
                efile.write(f"{result}\n")

    def __write_to_wrong(self, wrong_list, wrong_file):
        with codecs.open(wrong_file, encoding='utf-8', mode='w') as wfile:
            for (syn, oth, negative_min) in wrong_list:
                wfile.write(syn)
                wfile.write('\n')
                wfile.write(oth)
                wfile.write('\n')
                wfile.write(negative_min)
                wfile.write('\n')

    def __update_accuracy(self, eval_list, accuracy):
        if eval_list is self.eval_list1:
            self.accuracy1 = accuracy
        elif eval_list is self.eval_list2:
            self.accuracy2 = accuracy
        elif eval_list is self.eval_list3:
            self.accuracy3 = accuracy

    def __write_stats(self):
        dict_eval = {
            'condition1': (self.eval_list1, self.eval_file1, self.wrong_list1, self.wrong_file1, self.result_file1,
                           self.delta_list1, self.delta_file1, self.msg1),
            'condition2': (self.eval_list2, self.eval_file2, self.wrong_list2, self.wrong_file2, self.result_file2,
                           self.delta_list2, self.delta_file2, self.msg2),
            'condition3': (self.eval_list3, self.eval_file3, self.wrong_list3, self.wrong_file3, self.result_file3,
                           self.delta_list3, self.delta_file3, self.msg3)
        }

        for _, (eval_list,
                eval_file,
                wrong_list,
                wrong_file,
                result_file,
                delta_list,
                delta_file,
                message) in dict_eval.items():
            if not eval_list:
                continue

            accuracy = sum(eval_list) / len(eval_list) * 100
            self.__update_accuracy(eval_list, accuracy)

            self.__write_to_eval(eval_list, eval_file)
            self.__write_to_delta(delta_list, delta_file)
            self.__write_to_wrong(wrong_list, wrong_file)
            self.__write_to_summary(accuracy, result_file, message)

    def __write_to_summary(self, acc, result_file, message):
        today = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        model_name = '/'.join(self.file_to_eval.split('.')[-2:])
        result = f'{message}: {acc:.2f} \n'

        with codecs.open(result_file, encoding='utf-8', mode='w') as rfile:
            rfile.write(f'Evaluation done on: [{today}] \n')
            rfile.write(f'Model: {model_name} \n')
            rfile.write(f'Filename: {self.file_to_eval} \n')
            rfile.write(f'Nb of examples: {len(self.eval_list1)} \n')
            rfile.write(result)

    def eval_type1(self, nb_lines_test):
        self._eval_common_legacy(
            nb_lines_test,
            condition=lambda ref, syn, other: syn.score < other.score,
            message='Accuracy (syn < other)'
        )

    def _eval_common_legacy(self, nb_lines_test, condition, message):
        input_dir = self.config_dict['paths']['input_dir']
        file_to_eval_path = input_dir + self.file_to_eval

        with codecs.open(file_to_eval_path, encoding='utf-8', mode='r') as ifile:
            for next_lines in zip_longest(*[ifile] * nb_lines_test):
                ref = TestRecordResult(next_lines[0].strip().split('\t'))
                syn = TestRecordResult(next_lines[1].strip().split('\t'))
                other = TestRecordResult(next_lines[2].strip().split('\t'))

                if condition(ref, syn, other):
                    self.eval_list1.append(1)
                else:
                    self.eval_list1.append(0)
                    self.wrong_list1.append((str(ref), str(syn), str(other)))

    def eval_type5(self, nb_lines_test):
        """
        Evaluates the test type 5, where both incorrect and correct syn are recognized as such.
        The condition on the negative items is the minimum score.
        """
        self._eval_common(
            nb_lines_test,
            cond_neg=lambda negatives: min([n.score for n in negatives]),
            condition1=lambda syn, other, cond_neg: syn.score < cond_neg and other.score < cond_neg,
            condition2=lambda syn, other, cond_neg: syn.score < cond_neg,
            condition3=lambda syn, other, cond_neg: other.score < cond_neg,
            delta2=lambda syn, other, cond_neg: syn.score - cond_neg,
            delta3=lambda syn, other, cond_neg: other.score - cond_neg,
            message1='Accuracy (syn < neg and oth < neg)',
            message2='Accuracy (syn < neg)',
            message3='Accuracy (oth < neg)'
        )

    def eval_type6(self, nb_lines_test):
        """
        Evaluates the test type 6, where the correct syn must be better than the incorrect syn and random words.
        The condition on the negative items is the minimum score.
        """
        self._eval_common(
            nb_lines_test,
            cond_neg=lambda negatives: min([n.score for n in negatives]),
            condition1=lambda syn, other, cond_neg: syn.score < cond_neg and syn.score < other.score,
            condition2=lambda syn, other, cond_neg: syn.score < cond_neg,
            condition3=lambda syn, other, cond_neg: syn.score < other.score,
            delta2=lambda syn, other, cond_neg: syn.score - cond_neg,
            delta3=lambda syn, other, cond_neg: syn.score - other.score,
            message1='Accuracy (syn < oth and syn < neg)',
            message2='Accuracy (syn < neg)',
            message3='Accuracy (syn < oth)'
        )

    def eval_type7(self, nb_lines_test):
        """
        Evaluates the test type 6, where the correct syn must be better than the incorrect syn and random words.
        The condition on the negative items is the average score.
        """
        self._eval_common(
            nb_lines_test,
            cond_neg=lambda negatives: mean([n.score for n in negatives]),
            condition1=lambda syn, other, cond_neg: syn.score < cond_neg and syn.score < other.score,
            condition2=lambda syn, other, cond_neg: syn.score < cond_neg,
            condition3=lambda syn, other, cond_neg: syn.score < other.score,
            message1='Accuracy (syn < oth and syn < neg)',
            message2='Accuracy (syn < neg)',
            message3='Accuracy (syn < oth)'
        )

    def eval_type8(self, nb_lines_test):
        """
        Evaluates the test type 6, where we compute the rank of the correct synonym among the random words.
        No condition on negative items.
        """
        self._eval_common(
            nb_lines_test,
            cond_neg=lambda negatives: [n.score for n in negatives],
            condition1=lambda syn, other, cond_neg: syn.score < syn.score,
            condition2=lambda syn, other, cond_neg: self.get_rank(syn.score, cond_neg),
            condition3=lambda syn, other, cond_neg: syn.score < syn.score,
            message1='Accuracy (syn < oth and syn < neg)',
            message2='Accuracy (syn < neg)',
            message3='Accuracy (syn < oth)'
        )

    def eval_type9(self, nb_lines_test):
        """
        Evaluates the test type 6, where we compare the correct synonym with the random words two by two.
        No condition on negative items.
        """
        self._eval_common(
            nb_lines_test,
            cond_neg=lambda negatives: [n.score for n in negatives],
            condition1=lambda syn, other, cond_neg: syn.score < syn.score,
            condition2=lambda syn, other, cond_neg: self.get_2_by_2(syn.score, cond_neg),
            condition3=lambda syn, other, cond_neg: syn.score < syn.score,
            message1='Accuracy (syn < oth and syn < neg)',
            message2='Accuracy (syn < neg)',
            message3='Accuracy (syn < oth)'
        )

    def get_rank(self, number, num_list):
        # Sort the list
        sorted_list = sorted(num_list)

        # Find the rank
        rank = 1  # Start rank from 1
        for num in sorted_list:
            if number > num:
                rank += 1
            else:
                break

        return rank

    def get_2_by_2(self, number, num_list):
        # Sort the list
        acc = 0
        for num in num_list:
            if number < num:
                acc += 1
        return acc / len(num_list)

    def __extract_test_records(self, next_lines):
        syn = TestRecordResult(next_lines[0].strip().split('\t'))
        other = TestRecordResult(next_lines[1].strip().split('\t'))
        negatives = []
        for neg in next_lines[2:]:
            res = TestRecordResult(neg.strip().split('\t'))
            if res.flag != '0':
                raise ValueError(f"Negative example should have flag 0, got {res.flag}")
            negatives.append(res)

        return syn, other, negatives

    def _eval_common(self, nb_lines_test, cond_neg, condition1, condition2, condition3,
                     delta1=None, delta2=None, delta3=None,
                     message1=None, message2=None, message3=None):

        input_dir = self.config_dict['paths']['input_dir']
        file_to_eval_path = input_dir + self.file_to_eval

        self.msg1 = message1
        self.msg2 = message2
        self.msg3 = message3

        with codecs.open(file_to_eval_path, encoding='utf-8', mode='r') as ifile:
            for next_lines in zip_longest(*[ifile] * nb_lines_test):

                syn, other, negatives = self.__extract_test_records(next_lines)

                negative_min = min(negatives, key=lambda n: n.score)

                # make dictionaries for each condition and the corresponding files
                dict_cond = {
                    'condition1': (condition1, delta1, self.eval_list1, self.wrong_list1, self.delta_list1),
                    'condition2': (condition2, delta2, self.eval_list2, self.wrong_list2, self.delta_list2),
                    'condition3': (condition3, delta3, self.eval_list3, self.wrong_list3, self.delta_list3)
                }

                for _, (condition, delta, eval_list, wrong_list, delta_list) in dict_cond.items():

                    result = int(condition(syn, other, cond_neg(negatives)))
                    # result = condition(syn, other, cond_neg(negatives))
                    eval_list.append(result)

                    if delta is not None:
                        delta_list.append(delta(syn, other, cond_neg(negatives)))

                    if result == 0:
                        wrong_list.append((str(syn), str(other), str(negative_min)))
