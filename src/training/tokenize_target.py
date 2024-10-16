import codecs
from itertools import zip_longest
from hf_olmo import OLMoTokenizerFast

dir = 'output/synonyms/pairs_triples_examples/'
file = 'semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv'
filename = dir + file

output = 'output/synonyms/tokens/' + file + '_tokens_delta_abs.txt'

equal_nb_tokens = 0
syn_higher = 0

with codecs.open(filename, encoding='utf-8', mode='r') as ifile:
    with codecs.open(output, encoding='utf-8', mode='w') as ofile:
        for line in ifile:
            items = line.split('\t')
            syn = items[1]
            oth = items[2]

            tokenizer = OLMoTokenizerFast.from_pretrained(
                                                        "allenai/OLMo-1B",
                                                        trust_remote_code=True,
                                                        device='auto')
            syn_tokens = tokenizer.tokenize(syn)
            oth_tokens = tokenizer.tokenize(oth)

            #res = len(syn_tokens) - len(oth_tokens)  # delta

            res = abs(len(syn_tokens) - len(oth_tokens))  # delta abs

            #if len(syn_tokens) != len(oth_tokens):  # difference
            #    res = 1
            #else:
            #    res = 0

            #if len(syn_tokens) > len(oth_tokens):  # syn higher
            #    res = 1
            #else:
            #    res = 0

            print(res)

            ofile.write(str(res) + '\n')

            if len(syn_tokens) == len(oth_tokens):
                equal_nb_tokens += 1

            # how many syn is higher ?
            if len(syn_tokens) > len(oth_tokens):
                syn_higher += 1


print(f'Equal number of tokens: {equal_nb_tokens}')
print(f'Synonym number of tokens higher: {syn_higher}')

