import codecs

import numpy


def adapt_article(sentence, index, target):
    if target[0].lower() in ['a', 'e', 'i', 'o', 'u']:
        # if vowel and 'a' is present, replace with 'an'
        if sentence[index-3:index] == ' a ':
            sentence = sentence[:index-1] + 'n' + sentence[index-1:]
    else:
        # if consonant and 'an' is present, replace with 'a'
        if sentence[index-4:index] == ' an ':
            sentence = sentence[:index-2] + sentence[index-1:]

    return sentence


def adapt_punctuation(sentence):
    sentence = sentence.replace(' ,', ',')
    sentence = sentence.replace(' .', '.')
    sentence = sentence.replace(' !', '!')
    sentence = sentence.replace(' ?', '?')
    sentence = sentence.replace(' :', ':')
    sentence = sentence.replace(' ;', ';')
    sentence = sentence.replace('( ', '(')
    sentence = sentence.replace(' )', ')')
    sentence = sentence.replace('[ ', '[')
    sentence = sentence.replace(' ]', ']')
    sentence = sentence.replace('{ ', '{')
    sentence = sentence.replace(' }', '}')

    # special rule for apostrophe
    sentence = sentence.replace(" 's", "'s")
    sentence = sentence.replace(" 'd", "'d")
    sentence = sentence.replace(" 'n", "'n")
    sentence = sentence.replace(" 't", "'t")
    sentence = sentence.replace(" 'm", "'m")
    sentence = sentence.replace(" 'll", "'ll")
    sentence = sentence.replace(" 're", "'re")
    sentence = sentence.replace(" 've", "'ve")
    sentence = sentence.replace(" n't", "n't")

    # special rule for opening brackets [ `` ] -> [ "]
    sentence = sentence.replace("`` ", '"')

    # special rule for closing brackets [ '' ] -> [" ]
    sentence = sentence.replace(" ''", '"')

    return sentence


def index_in_sentence(sentence, index):
    index_str = 0
    sentence_length = len(sentence)
    for i in range(sentence_length):
        if i != index and i != sentence_length - 1:
            index_str += len(sentence[i]) + 1
        else:
            break
    return index_str


def post_process(sentence, index_string, target):

    # adapt article
    sentence = adapt_article(sentence, index_string, target)

    # adapt punctuation
    sentence = adapt_punctuation(sentence)
    return sentence


if __name__ == "__main__":
    ex_pairs_triples_dir = "output/synonyms/pairs_triples_examples/"
    test_dir = "output/synonyms/test/human/"

    input_filename = "semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv"

    input_file = ex_pairs_triples_dir + input_filename
    output_file = test_dir + "t1_pp_man_" + input_filename + ".new"
    output_ans_file = test_dir + "t1_pp_man_" + input_filename + ".ans" + ".new"

    i = 0
    blind = True
    which = -1
    numpy.random.seed(160)

    with codecs.open(input_file, encoding='utf-8', mode='r') as efile:
        with codecs.open(output_file, encoding='utf-8', mode='w') as pfile:
            with codecs.open(output_ans_file, 
                             encoding='utf-8', mode='w') as afile:
                for line in efile:
                    i += 1
                    elems = line.strip('\n').split('\t')
                    # print(elems)

                    src = elems[0]
                    synonym = elems[1]
                    other = elems[2]
                    index = int(elems[3])
                    example = elems[4].split(' ')

                    index_str = index_in_sentence(example, index)

                    replacements = []
                    replacements.append(src)

                    if blind:
                        # randomly choose which sentence to appear first
                        which = numpy.random.randint(0, 2, 1)
                        # print(f"Which is {which}")
                        if which == 0:
                            rest = [synonym, other]
                        else:
                            rest = [other, synonym]
                        [replacements.append(word) for word in rest]
                    else:
                        # always synonym first
                        replacements.append(synonym)
                        replacements.append(other)

                    for word in replacements:

                        if word == src:
                            flag = 2
                        elif word == synonym:
                            flag = 1
                        else:
                            flag = 0

                        example[index] = f'[{word}]'
                        context = post_process(' '.join(example), index_str,
                                               word)

                        pfile.write(str(i))
                        pfile.write('\t')

                        if not blind:
                            # write answer to current file
                            pfile.write(str(flag))
                            pfile.write('\t')
                        else:
                            # write answers to separate file
                            if flag == 1:  # true synonym - only write once
                                if which == 0:
                                    # print(f"In if which is {which}, write 2")
                                    afile.write("2")
                                else:
                                    # print(f"In if which is {which} write 3")
                                    afile.write("3")
                                afile.write('\n')
                                
                        pfile.write(word)
                        pfile.write('\t')
                        pfile.write(context)
                        pfile.write('\n')
