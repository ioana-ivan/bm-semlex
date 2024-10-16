import codecs
import importlib
from nltk.corpus import semcor
from nltk.tree import Tree
from nltk.corpus.reader.wordnet import Lemma
import numpy
import old.extract_concept_from_dataset as ext
from nltk.corpus import wordnet

importlib.reload(ext)

'''
import nltk
nltk.download('wordnet')
nltk.download('semcor')
'''


def actual_form(element):
    if isinstance(element, Tree):
        leaves = element.leaves()
        if len(leaves) > 1:
            # more than one child, complex word
            form = '_'.join(leaves)
        else:
            form = leaves[0]
    else:
        form = element[0]
    return form


def actual_sentence(tagged_sentence):
    sentence = []

    for element in tagged_sentence:
        form = actual_form(element)
        sentence.append(form)

    return ' '.join(sentence)


def write_to_file(triples, file):
    with codecs.open(file, encoding='utf-8', mode='w') as ofile:
        for triple, triple_sent in triples.items():
            list_triple = list(triple)
            list_triple_sent = list(triple_sent)
            # print(list_triple, list_triple_sent)
            for i in range(len(list_triple)):
                ofile.write(list_triple[i])
                ofile.write('\t')
                ofile.write(list_triple_sent[i])
                ofile.write('\n')


def most_frequent_lemma(lemma_list):
    freq_dict = {}
    for lemma in lemma_list:
        count = lemma.count()
        freq_dict[lemma] = count
    # print(freq_dict)

    if not freq_dict:
        return None

    return max(freq_dict, key=lambda key: freq_dict[key])


def choose_synonym(target, strategy):
    # same synset as target synset
    target_synset = target.synset()
    synset_lemmas = target_synset.lemmas()

    synonym_lemmas = list(set(synset_lemmas) - {target})

    filtered_lemmas = synonym_conditions(synonym_lemmas, target)

    if not filtered_lemmas:
        return None

    # print(f'Choosing synonym from synset_lemmas {filtered_lemmas}')
    return choose_strategy(strategy, filtered_lemmas)


def choose_synonyms(target):
    # same synset as target synset
    target_synset = target.synset()
    synset_lemmas = target_synset.lemmas()

    synonym_lemmas = list(set(synset_lemmas) - {target})

    filtered_lemmas = synonym_conditions(synonym_lemmas, target)

    if not filtered_lemmas:
        return None

    # print(f'Filtered synset_lemmas {filtered_lemmas}')
    return filtered_lemmas


def choose_other(target, synonyms, pos, strategy):
    # get other synsets
    synsets = other_synsets(target, pos)
    other_lemmas = []
    filtered_lemmas = []

    # print(f'Synonyms {synonyms}')

    if not isinstance(synonyms, list):
        synonyms = [synonyms]

    for synonym in synonyms:
        for synset in synsets:
            lemmas = synset.lemmas()
            # FILTER synsets : synonym form in other_syn synset
            if form_in_synset(synonym.name(), synset):
                continue
            for lemma in lemmas:
                if lemma not in other_lemmas:
                    other_lemmas.append(lemma)

        filtered_lemmas = synonym_conditions(other_lemmas, target)

        filtered_lemmas = other_conditions(filtered_lemmas, synonym)

    # print(f'Choosing other from synset_lemmas {filtered_lemmas}')
    return choose_strategy(strategy, filtered_lemmas)


def choose_others(target, synonyms, pos):
    # get all other possible synonyms
    synsets = other_synsets(target, pos)
    other_lemmas = []
    filtered_lemmas = []

    # print(f'Synonyms {synonyms}')

    if not isinstance(synonyms, list):
        synonyms = [synonyms]

    for synonym in synonyms:
        for synset in synsets:
            lemmas = synset.lemmas()

            # FILTER synsets : synonym form in other_syn synset
            if form_in_synset(synonym.name(), synset):
                continue

            # add all to list
            filtered_lemmas = synonym_conditions(lemmas, target)

            filtered_lemmas = other_conditions(filtered_lemmas, synonym)

            if filtered_lemmas not in other_lemmas:
                other_lemmas.append(filtered_lemmas)

    # print(f'Filtered synset_lemmas other {other_lemmas}')
    return other_lemmas


def choose_strategy(strategy, lemmas):
    if strategy == 'first':
        return lemmas[0]
    elif strategy == 'freq':
        return most_frequent_lemma(lemmas)
    else:
        print('Unknown strategy to select lemma')
        return None


def compound_word(form):
    if '_' in form:
        return True
    else:
        return False


def number(form):
    if form.isdigit():
        return True
    else:
        return False


def synonym_conditions(lemmas, target):
    # filter compound words
    lemmas = list(filter(lambda lemma:
                         not compound_word(lemma.name()),
                         lemmas))

    # filter numbers
    lemmas = list(filter(lambda lemma:
                         not number(lemma.name()),
                         lemmas))

    # filter same form as target incl. caps
    lemmas = list(filter(lambda lemma:
                         not lemma.name().lower() == target.name().lower(),
                         lemmas))

    # filter all containing caps
    lemmas = list(filter(lambda lemma:
                         lemma.name().islower(),
                         lemmas))

    return lemmas


def other_conditions(lemmas, synonym):
    # filter same form as synonym incl. caps
    lemmas = list(filter(lambda lemma:
                         not lemma.name().lower() == synonym.name().lower(),
                         lemmas))

    # FILTER other synonym form in synonym synset
    lemmas = list(filter(lambda lemma:
                         not form_in_synset(lemma.name(), synonym.synset()),
                         lemmas))

    # filter supersense(synonym) == supersense(other)
    lemmas = list(filter(lambda lemma:
                         not (supersense(lemma) == supersense(synonym)),
                         lemmas))

    # filter if synonym and other have common hypernyms
    lemmas = list(filter(lambda lemma:
                         not (common_hypernyms(lemma, synonym)),
                         lemmas))

    # filter all containing caps
    lemmas = list(filter(lambda lemma:
                         lemma.name().islower(),
                         lemmas))

    return lemmas


def supersense(lemma):
    return lemma.synset().lexname()


def common_hypernyms(lemma1, lemma2):
    l1_hypernyms = set(lemma1.synset().hypernyms())
    l2_hypernyms = set(lemma2.synset().hypernyms())
    # print(f'l1_hypernyms {l1_hypernyms} -> ls_hypernyms {l2_hypernyms}')

    if list(l1_hypernyms.intersection(l2_hypernyms)):
        # print(f'Filtered {list(l1_hypernyms.intersection(l2_hypernyms))}')
        return True
    return False


def other_synsets(lemma, pos):
    lemma_name = str(lemma.name())

    # get all possible synsets
    synset_list = wordnet.synsets(lemma_name, pos=pos)

    # exclude synset of target word lemma
    synset_list = list(set(synset_list) - {lemma.synset()})

    return synset_list


def write_triple_example(triples, file):
    with codecs.open(file, encoding='utf-8', mode='w') as ofile:
        for triple, ind_sent_list in triples.items():
            # print(f'triple {triple} -> values {ind_sent_list}')
            triple_l = list(triple)

            for ind_sent in ind_sent_list:
                for i in range(len(triple_l)):
                    ofile.write(triple_l[i])
                    ofile.write('\t')

                ofile.write(str(ind_sent[0]))
                ofile.write('\t')
                ofile.write(ind_sent[1])
                ofile.write('\n')


def form_in_synset(form, synset):
    synset_lemmas = synset.lemmas()
    synset_forms = [lemma.name() for lemma in synset_lemmas]
    if form in synset_forms:
        return True
    return False


def istagged(elem):
    if isinstance(elem, Tree) and \
       isinstance(elem.label(), Lemma):
        return True


def target_conditions(element):
    # tagged word conditions
    form = actual_form(element)
    lemma = element.label()

    # condition: no compound words
    if lemma.name() == 'group':
        return False

    # condition: form == lemma
    if lemma.name() != form:
        return False

    # condition: filter by pos
    if lemma.synset().pos() != pos:
        return False

    # condition: exclude any not lowercase
    if not lemma.name().islower():
        return False

    return True


def write_triples(list_triples, file):
    with codecs.open(file, encoding='utf-8', mode='w') as ofile:
        for triple in list_triples:
            triple = list(triple)
            ofile.write(triple[0])
            ofile.write('\t')
            ofile.write(triple[1])
            ofile.write('\t')
            ofile.write(triple[2])
            ofile.write('\n')


def write_triple_lists(triples, file):
    with codecs.open(file, encoding='utf-8', mode='w') as ofile:
        for triple, ind_sent_list in triples.items():
            triple_l = list(triple)
            # 3 cases: tuple of one lemma, tuple of lemmas, tuple of tuples
            for i in range(len(triple_l) - 1):
                item = triple_l[i]

                # print(f'Is a tuple of lemmas {item}')
                lemma_list = []

                for i in range(len(item)):
                    lemma_list.append(item[i].name())

                # ofile.write(str(lemma_list))
                write_list(ofile, lemma_list)
                ofile.write('\t')

            # tuple of tuples
            tuple_tuples = triple_l[-1]

            for i in range(len(tuple_tuples)):
                tuple_lemmas = tuple_tuples[i]
                lemma_name_list = []
                for j in range(len(tuple_lemmas)):
                    lemma_name_list.append(tuple_lemmas[j].name())

                # ofile.write(str(lemma_list))

                write_list(ofile, lemma_name_list)

                if i == (len(tuple_tuples) - 1):
                    ofile.write('\t')
                else:
                    ofile.write(' - ')

            for ind_sent in ind_sent_list:
                ofile.write(str(ind_sent[0]))
                ofile.write('\n')
                ofile.write(ind_sent[1].replace(triple_l[0][0].name(),
                                                f'[{triple_l[0][0].name()}]'))
                ofile.write('\n')


def write_list(ofile, list):
    for i in range(len(list)):
        ofile.write(list[i])
        if i < (len(list) - 1):
            ofile.write(', ')


def limit_dataset(tagged_sentences,
                  chunks,
                  sentences,
                  random_examples,
                  nb_examples_max):

    if random_examples:
        selected_indices = numpy.random.choice(range(len(tagged_sentences)),
                                               nb_examples_max)

        tagged_sentences = [tagged_sentences[i] for i in selected_indices]
        chunks = [chunks[i] for i in selected_indices]
        sentences = [sentences[i] for i in selected_indices]
    elif nb_examples_max > 0:
        # get tagged and normal sentences from semcor
        tagged_sentences = semcor.tagged_sents(tag='sem')[:nb_examples_max]
        chunks = semcor.chunk_sents()[:nb_examples_max]
        sentences = semcor.sents()[:nb_examples_max]
        # print('Total nb of sentences: ', len(tagged_sentences))

    return tagged_sentences, chunks, sentences


def print_triple_freq_count(lemma, synonym, other_syn):
    print(f'Triple {lemma.name()}: {lemma.count()} -> \
      {synonym.name()}: {synonym.count()} -> \
      {other_syn.name()}: {other_syn.count()}')


def select_triples(tagged_sentences, chunks, sentences, strategy_syn, strategy_other, reuse_examples):
    triples = {}
    for i in range(len(tagged_sentences)):
        index = -1
        tagged_sentence = tagged_sentences[i]
        chunk_sentence = chunks[i]
        sentence = ' '.join(sentences[i])

        # filter small sentences
        if len(tagged_sentence) < min_sentence_length:
            continue

        for j in range(len(tagged_sentence)):
            element = tagged_sentence[j]
            index += len(chunk_sentence[j])

            # significant word
            if istagged(element) and target_conditions(element):

                lemma = element.label()

                # choose synonym
                synonym = choose_synonym(lemma,
                                         strategy_syn)

                if synonym is None:
                    continue

                # choose other synonym
                other_syn = choose_other(lemma, synonym, pos, strategy_other)

                if other_syn is None:
                    continue

                print_triple_freq_count(lemma, synonym, other_syn)

                # add to triples
                triple_t = tuple([lemma.name(),
                                 synonym.name(),
                                 other_syn.name()])
                
                if triple_t not in triples:
                    triples[triple_t] = [(index, sentence)]
                else:
                    if reuse_examples:
                        list_values = triples[triple_t]
                        list_values.append((index, sentence))

    return triples


def select_triples_lists(tagged_sentences, chunks, sentences, strategy_syn,
                         strategy_other, reuse_examples):
    triples = {}
    for i in range(len(tagged_sentences)):
        index = -1
        tagged_sentence = tagged_sentences[i]
        chunk_sentence = chunks[i]
        sentence = ' '.join(sentences[i])

        # filter small sentences
        if len(tagged_sentence) < min_sentence_length:
            continue

        for j in range(len(tagged_sentence)):
            element = tagged_sentence[j]
            index += len(chunk_sentence[j])

            # significant word
            if istagged(element) and target_conditions(element):

                lemma = element.label()

                # all synonyms
                synonyms = choose_synonyms(lemma)

                if synonyms is None or len(synonyms) == 0 or \
                   (len(synonyms) == 1 and not synonyms[0]):
                    continue

                # choose other synonyms
                other_syns = choose_others(lemma,
                                           synonyms,
                                           pos)

                # print(f'{lemma} -> {other_syns}')
                if other_syns is None or len(other_syns) == 0 or \
                   (len(other_syns) == 1 and not other_syns[0]):
                    continue

                # add lists
                other_syns_list = []
                for lemma_list in other_syns:
                    # print(f'Looking at lemma list : {lemma_list}')
                    if lemma_list:
                        other_syns_list.append(tuple(lemma_list))

                triple_t = tuple([(lemma,),
                                  tuple(synonyms),
                                  tuple(other_syns_list)])

                if triple_t not in triples:
                    triples[triple_t] = [(index, sentence)]
                else:
                    if reuse_examples:
                        list_values = triples[triple_t]
                        list_values.append((index, sentence))

    return triples


def triples_filename(out_dir, strategy_syn, strategy_other, pos, example_limit, seed, random_examples):
    return (out_dir
            + 'semcor_wordnet_'
            + strategy_syn[0]
            + strategy_other[0]
            + '_'
            + pos + '_'
            + str(example_limit)
            + ('_rand_' + str(seed) if random_examples else '')
            + '.csv')


if __name__ == "__main__":
    # mode in ['triples', 'examples', 'lists']
    mode = 'examples'

    # filtering conditions
    min_sentence_length = 10  # in chunks
    pos = 'n'

    if mode == 'lists':
        strategy_other = 'all'
        strategy_syn = 'all'
    else:
        strategy_other = 'freq'
        strategy_syn = 'freq'

    # overwrite strategy
    # strategy_syn = 'freq'
    # strategy_other = 'freq'

    # dataset conditions
    random_examples = True
    example_limit = 30000
    seed = 161
    numpy.random.seed(seed)
    reuse_examples = False  # use an ex more than once if multiple tags

    # get semcor dataset
    tagged_sentences = semcor.tagged_sents(tag='sem')
    chunks = semcor.chunk_sents()
    sentences = semcor.sents()

    # limit dataset
    tagged_sentences, chunks, sentences = limit_dataset(tagged_sentences,
                                                        chunks,
                                                        sentences,
                                                        random_examples,
                                                        example_limit)

    print(f'Nb tagged sentences {len(tagged_sentences)}')

    # select triples
    if mode == 'lists':
        triples = select_triples_lists(tagged_sentences, chunks, sentences,
                                       strategy_syn,
                                       strategy_other,
                                       reuse_examples)
    else:
        triples = select_triples(tagged_sentences, chunks, sentences,
                                 strategy_syn,
                                 strategy_other,
                                 reuse_examples)

    if mode == 'triples':
        out_dir = 'output/synonyms/pairs_triples/raw/'
        out_file = triples_filename(out_dir,
                                    strategy_syn,
                                    strategy_other,
                                    pos,
                                    example_limit,
                                    seed,
                                    random_examples)

        write_triples(triples, out_file)
        print(f'Write triples to file {out_dir}{out_file}')

    elif mode == 'examples':
        out_dir = 'output/synonyms/pairs_triples_examples/'
        out_file = triples_filename(out_dir,
                                    strategy_syn,
                                    strategy_other,
                                    pos,
                                    example_limit,
                                    seed,
                                    random_examples)

        write_triple_example(triples, out_file)
        print(f'Write triples examples to file {out_dir}{out_file}')

    elif mode == 'lists':
        out_dir = 'output/synonyms/pairs_triples_lists/'
        out_file = triples_filename(out_dir,
                                    strategy_syn,
                                    strategy_other,
                                    pos,
                                    example_limit,
                                    seed,
                                    random_examples)

        write_triple_lists(triples, out_file)
        print(f'Write triples lists to file {out_dir}{out_file}')