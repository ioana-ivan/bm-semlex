import codecs
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn


def nouns_to_file(limit, pos, output_file):

    synsets = wn.all_synsets(pos=pos)

    for synset in synsets:
        for lemma in synset.lemmas():
            if ('_' not in lemma.name() and '-' not in lemma.name() 
               and lemma.name().islower()):
                output_file.write(lemma.name() + '\n')
                limit -= 1
                if limit == 0:
                    return 0


if __name__ == "__main__":
    pos = 'n'
    limit = 10000
    filedir = "output/synonyms/negative/"
    filename = pos + "_" + str(limit) + ".txt"
    file = filedir + filename
    output_file = codecs.open(file, encoding='utf-8', mode='w')

    nouns_to_file(limit, pos, output_file)