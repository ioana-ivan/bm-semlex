
def adapt_plural(prompt, word):
    if word in ['comforts',
                'conveniences',
                'belongings',
                'laurels',
                'means',
                'grounds',
                'spirits',
                'words']:
        return prompt.replace('This', 'These')
    return prompt


def adapt_article(sentence, index, target):
    if target[0].lower() in ['a', 'e', 'i', 'o', 'u']:
        # if vowel and 'a' is present, replace with 'an'
        if sentence[index - 3:index] == ' a ':
            sentence = sentence[:index - 1] + 'n' + sentence[index - 1:]
    else:
        # if consonant and 'an' is present, replace with 'a'
        if sentence[index - 4:index] == ' an ':
            sentence = sentence[:index - 2] + sentence[index - 1:]
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
