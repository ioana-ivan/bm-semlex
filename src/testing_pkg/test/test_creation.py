import codecs
import yaml
import random

from testing_pkg.test.data_classes import TestElements, TestRecord, TestType, Flag
from testing_pkg.test.data_utils import adapt_plural, adapt_punctuation, adapt_article


class TestCreation():

    def __init__(self):
        """
        Initializes the TestCreation class.
        Attributes:
            test_elements_list (list): A list to store test elements.
            test_records_list (list): A list to store test records.
            configfile (str): Path to the configuration file.
            config (dict): Dictionary to store configuration settings.
            test_parameters (dict): Dictionary to store test parameters.
            create_methods (dict): Dictionary mapping test types to their respective creation methods.
        """
        self.test_elements_list = []
        self.test_records_list = []
        self.configfile = "src\\testing_pkg\\test\\test_config.yaml"
        self.config_dict = {}
        self.test_parameters = {}

        self.create_methods = {
            TestType.type_sub: self.__create_type_sub,
            TestType.type_relnc: self.__create_type_relnc,
            TestType.type_relc: self.__create_type_relc,
            TestType.type_ref: self.__create_type_ref,
            TestType.type_relnc_rand: self.__create_type_relnc_rand,
            TestType.type_relc_rand: self.__create_type_relc_rand,
            TestType.type_sub_rand: self.__create_type_sub_rand,
        }

    def create_test(self, config_test_name, base_filename):
        """
        Creates a test based on the provided test number and filename.

        Args:
            config_test_name (int): The name of the test to create as defined in config.
            base_filename (str): The name of the base file containing the data used to create the test.

        Returns:
            None
        """

        # load test creation config
        self.load_config()

        # load test data from the intermediate base file
        self.__get_test_data(base_filename)

        # load test parameters from the config file
        self.load_test_params_from_config(config_test_name)

        # get the type of test to be created
        test_type = TestType(int(self.test_parameters['type']))

        try:
            create_method = self.create_methods[test_type]
        except KeyError:
            raise Exception(f"Test type {test_type} not implemented")

        create_method()

        return self.__write_to_file(config_test_name, base_filename)

    def __get_test_data(self, basefile):
        """
        Loads test elements from a specified base file and appends them to the test elements list.

        Args:
            basefile (str): The name of the base file to load test elements from.
        Raises:
            FileNotFoundError: If the specified file does not exist.
            IOError: If there is an error reading the file.
        """
        input_dir = self.config_dict['paths']['input_dir']
        filepath = input_dir + basefile
        print(f"Loading input file: {filepath}")

        try:
            with codecs.open(filepath, encoding='utf-8', mode='r') as ifile:
                for line in ifile:
                    elements = line.strip('\n').split('\t')
                    test_element = TestElements(elements)
                    self.test_elements_list.append(test_element)
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except IOError as e:
            print(f"Error reading file: {e}")

    def load_config(self):
        """
        Loads the configuration from a YAML file specified by `self.configfile`.

        This method attempts to open the configuration file in read mode with UTF-8 encoding,
        and then loads its contents into the `self.config` attribute using `yaml.safe_load`.

        Raises:
            Exception: If there is an error opening or reading the configuration file,
                       an exception is caught and an error message is printed.
        """
        try:
            with codecs.open(self.configfile, encoding='utf-8', mode='r') as f:
                self.config_dict = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file [{self.configfile}]: {e}")

    def load_test_params_from_config(self, name):
        """
        Loads test parameters from the configuration based on the given test name.
        Args:
            name (str): The name of the test for which parameters are to be loaded.
        Raises:
            Exception: If test parameters for the given name are not found in the configuration.
        """
        if not self.config_dict:
            self.load_config()

        dict_test = {}
        config_tests = self.config_dict['tests']
        for dict_test in config_tests:
            if dict_test['name'] == name:
                self.test_parameters = dict_test

        if not self.test_parameters:
            raise Exception(f"Test parameters for {name} not found")

    def __write_to_file(self, testname, basefile):
        testfile = self.config_dict['paths']['test_dir'] + testname + "_" + basefile
        with codecs.open(testfile, encoding='utf-8', mode='w') as pfile:
            for item in self.test_records_list:
                pfile.write(str(item.itemnb))
                pfile.write('\t')
                pfile.write(str(item.flag))
                pfile.write('\t')
                pfile.write(str(item.word))
                pfile.write('\t')
                pfile.write(str(item.context))
                pfile.write('\n')

        return testfile

    def __post_process(self, sentence, index_string, target):
        sentence = adapt_article(sentence, index_string, target)
        sentence = adapt_punctuation(sentence)
        return sentence

    def __associate_flag(self, src, synonym, other):
        flag_dict = {}
        flag_dict[src] = Flag.SOURCE.value
        flag_dict[synonym] = Flag.SYNONYM.value
        flag_dict[other] = Flag.OTHER.value
        return flag_dict

    def __associate_flag_rand(self, src, synonym, other):
        flag_dict = {}
        flag_dict[synonym] = Flag.SOURCE.value
        flag_dict[other] = Flag.SYNONYM.value
        return flag_dict

    def __index_in_sentence(self, example, index):
        word_list = example.split(' ')
        index_str = 0
        sentence_length = len(word_list)

        for i in range(sentence_length):
            if i != index and i != sentence_length - 1:
                index_str += len(word_list[i]) + 1
            else:
                break
        return index_str

    def __substitute(self, word, example, index):
        ex_word_list = example.split(' ')

        ex_word_list[index] = word

        example = ' '.join(ex_word_list)
        return example

    def __load_negative_elements(self, negative):
        """
        Loads negative elements from a file if provided.

        Args:
            negative (str): Path to the file containing words to be used in negative examples.

        Returns:
            list: A list of words.
        """
        negative_elements = []
        if negative:
            try:
                with codecs.open(negative, encoding='utf-8', mode='r') as f:
                    for line in f:
                        negative_elements.append(line.strip('\n'))
            except FileNotFoundError:
                print(f"Negative examples file not found: {negative}")
            except IOError as e:
                print(f"Error reading negative examples file: {e}")
        return negative_elements

    def __choose_negative(self, negative_elements, test_elements, nb_neg, seed_offset):
        """
        Generates a list of negative examples excluding the target words.

        Args:
            negative_elements (list): List of negative elements.
            test_elements (TestElements): The current test elements.
            nb_neg (int): Number of negative examples to select.
            seed_offset (int): Offset for the random seed.

        Returns:
            list: A list of selected negative examples.
        """
        if not negative_elements:
            return []

        # Exclude target, synonym and other from negative words list

        # Copy is needed to preserve the original list and also the seed
        negative_elements_copy = negative_elements.copy()
        for word in [test_elements.src, test_elements.synonym, test_elements.other]:
            if word in negative_elements_copy:
                negative_elements_copy.remove(word)

        random.seed(self.config_dict['constants']['seed'] + seed_offset)
        neg_list = random.choices(negative_elements_copy, k=nb_neg)
        if len(neg_list) != nb_neg:
            raise ValueError("Wrong number of negative examples")
        return neg_list

    def __create_test_records(self, special_instr, associate_flag, prompt=None, negative=None, nb_neg=0):
        """
        Helper method to create test records.

        Args:
            special_instr (function): A function that defines the specific processing logic for each type.
            associate_flag (function): A function that associates flags to words.
            prompt (str, optional): The prompt template used for generating context.
            negative (str, optional): Path to the file containing negative examples.
            nb_neg (int, optional): Number of negative examples to select.

        Returns:
            None
        """
        negative_elements = self.__load_negative_elements(negative)

        for i, test_elements in enumerate(self.test_elements_list):
            # Choose negative words
            neg_list = self.__choose_negative(negative_elements, test_elements, nb_neg, i + 1)

            # Get string-type index using list-type index for post-processing
            index_str = self.__index_in_sentence(test_elements.example, test_elements.index)

            # Associate flag to each target word {word : flag}
            flag_element_dict = associate_flag(
                test_elements.src,
                test_elements.synonym,
                test_elements.other
            )

            # Add negative examples to the flag dictionary if applicable
            for other in neg_list:
                flag_element_dict[other] = Flag.OTHER.value

            # For each (target word, synonym, other_synonym) in the triple
            for word, flag in flag_element_dict.items():

                context = special_instr(test_elements, word, index_str, flag, prompt)

                test_record = TestRecord(
                    str(i + 1),
                    str(flag),
                    word,
                    context)

                self.test_records_list.append(test_record)

    def __create_type_sub(self):
        """
        Create test of type substitution.
        """
        def special_instr(test_elements, word, index_str, flag, prompt):
            example_sub = self.__substitute(
                word,
                test_elements.example,
                test_elements.index)

            context = self.__post_process(
                example_sub,
                index_str,
                word)
            return context

        self.__create_test_records(special_instr, self.__associate_flag)

    def __create_type_relnc(self):
        """
        Creates test of type relation without context.
        {src} is evaluated in prompt.
        """
        prompt = self.test_parameters['prompt']

        def special_instr(test_elements, word, index_str, flag, prompt):
            src = test_elements.src.capitalize() # NOQA
            context = eval(f"f'{prompt}'")
            return context

        self.__create_test_records(special_instr, self.__associate_flag, prompt=prompt)

    def __create_type_relc(self):
        """
        Creates test of type relation with context.
        {src} and {example} are evaluated in prompt.
        """
        prompt = self.test_parameters['prompt']

        def special_instr(test_elements, word, index_str, flag, prompt):
            src = test_elements.src.capitalize() # NOQA
            example = self.__post_process(test_elements.example, index_str, word) # NOQA
            context = eval(f"f'{prompt}'")
            return context

        self.__create_test_records(special_instr, self.__associate_flag, prompt=prompt)

    def __create_type_ref(self):
        """
        Creates test of type reference.
        {src} and {example} are evaluated in prompt.
        """
        prompt = self.test_parameters['prompt']

        def process_func(test_elements, word, index_str, flag, prompt):
            src = test_elements.src.capitalize() # NOQA
            example = self.__post_process( # NOQA
                test_elements.example,
                index_str,
                word)

            prompt = adapt_plural(prompt, word)
            context = eval(f"f'{prompt}'")
            return context

        self.__create_test_records(process_func, self.__associate_flag, prompt=prompt)

    def __create_type_relnc_rand(self):
        """
        Creates test of type relation without context with random negative examples.
        {src} and {example} are evaluated in prompt.
        """
        prompt = self.test_parameters['prompt']
        negative = self.test_parameters['negative']
        nb_neg = self.test_parameters['nb_neg']

        def process_func(test_elements, word, index_str, flag, prompt):
            src = test_elements.src.capitalize() # NOQA
            example = self.__post_process( # NOQA
                test_elements.example,
                index_str,
                word)

            context = eval(f"f'{prompt}'")
            return context

        self.__create_test_records(process_func, self.__associate_flag_rand, prompt=prompt, negative=negative,
                                   nb_neg=nb_neg)

    def __create_type_relc_rand(self):
        """
        Creates test of type relation with context with both other_syn and random negative examples.
        {src} and {example} are evaluated in prompt.
        """
        prompt = self.test_parameters['prompt']
        negative = self.test_parameters['negative']
        nb_neg = self.test_parameters['nb_neg']

        def process_func(test_elements, word, index_str, flag, prompt):
            src = test_elements.src.capitalize() # NOQA
            example = self.__post_process( # NOQA
                test_elements.example,
                index_str,
                word)

            if 'This' in prompt:
                prompt = adapt_plural(prompt, word)
            context = eval(f"f'{prompt}'")
            return context

        self.__create_test_records(process_func, self.__associate_flag_rand, prompt=prompt, negative=negative,
                                   nb_neg=nb_neg)

    def __create_type_sub_rand(self):
        """
        Creates test of type substitution with both other_syn and random negative examples.
        """
        negative = self.test_parameters['negative']
        nb_neg = self.test_parameters['nb_neg']

        def process_func(test_elements, word, index_str, flag, prompt):
            example_sub = self.__substitute(word, test_elements.example, test_elements.index)
            context = self.__post_process(example_sub, index_str, word)
            return context

        self.__create_test_records(process_func, self.__associate_flag_rand, negative=negative, nb_neg=nb_neg)
