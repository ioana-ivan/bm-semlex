import codecs
from itertools import zip_longest


if __name__ == "__main__":
    curated_dir = "output/synonyms/pairs_triples_lists/"
    curated_filename = "semcor_wordnet_aa_n_5000_rand_160.csv.curated.200"

    output_dir = "output/synonyms/pairs_triples_examples/"
    output_filename = curated_filename + ".csv"

    curated_file = curated_dir + curated_filename
    output_file = output_dir + output_filename

    nb_lines_grouped = 2

    with codecs.open(curated_file, encoding='utf-8', mode='r') as ifile:
        with codecs.open(output_file, encoding='utf-8', mode='w') as ofile:
            for next_lines in zip_longest(*[ifile] * nb_lines_grouped):

                triple_ind = next_lines[0].strip()
                example = next_lines[1].strip()

                target_word = triple_ind.split('\t')[0]
                synonym = triple_ind.split('\t')[1]

                example = example.replace(f'[{target_word}]', target_word)

                ofile.write(triple_ind)
                ofile.write('\t')
                ofile.write(example)
                ofile.write('\n')
