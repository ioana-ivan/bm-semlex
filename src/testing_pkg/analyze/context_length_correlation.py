import codecs
from matplotlib import pyplot as plt
import numpy as np
import scipy
from hf_olmo import OLMoTokenizerFast
from transformers import LlamaTokenizer


def get_tokenizer(model):
    if model == "allenai.OLMo-1B":
        tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-1B",
                                                      trust_remote_code=True,
                                                      device='auto')
    elif model == "allenai.OLMo-7B":
        tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B",
                                                      trust_remote_code=True,
                                                      device='auto')
    elif model == "llm360.Amber":
        tokenizer = LlamaTokenizer.from_pretrained("llm360/Amber")

    return tokenizer


def get_tokens_distance(tokenizer, sentence, target_word):
    index_str = sentence.find(' is a synonym of')
    #print(f'[{sentence[:index_str]}]')
    #print(sentence[:index_str].split(' '))
    
    # extract word before index from sentence
    target_word = sentence[:index_str].split(' ')[-1]
    #print(target_word.lower())

    if target_word == '':
        raise ValueError(f"Error: Target word {target_word} not found in sentence {sentence}.")
    #print(target_word, index_str)
    #print(sentence)
    sentence = sentence[:index_str - len(target_word)]
    #print(sentence)

    index = sentence.find(target_word)
    if index == -1:
        index = sentence.find(target_word.lower())
        if index == -1:
            print(f"Error: Target word {target_word} not found in sentence {sentence}.")
    sentence = sentence[index + len(target_word):]

    return tokenizer.tokenize(sentence)


def corr_and_plot(file1, file2, model):
    # read file
    tokenizer = get_tokenizer(model)
    delta_pp = []
    acc = []
    with codecs.open(file1, 'r', 'utf-8') as f:
        for line in f:
            delta_pp.append(float(line.strip()))
            if (float(line.strip()) <= 0.0):
                acc.append(1)
            else:
                acc.append(0)

    line_tokens = []
    with codecs.open(file2, 'r', 'utf-8') as f:
        # read 12 lines at a time
        for next_lines in zip(*[f] * 12):
            line = next_lines[0]
            sentence = line.strip().split('\t')[3]
            target_word = line.strip().split('\t')[2]

            tokens = tokenizer.tokenize(sentence)

            # tokens = get_tokens_distance(tokenizer, sentence, target_word)

            line_tokens.append(len(tokens))

    print(f"Mean accuracy: {sum(acc) / len(acc)}")
    score = sum(acc) / len(acc)

    if len(line_tokens) != len(delta_pp):
        raise ValueError(f"Error: Each file must contain exactly 200 items. Found {len(line_tokens)} and {len(delta_pp)}.")

    # correlations
    correlation_p = scipy.stats.pearsonr(line_tokens, delta_pp)
    print(f'Scipy Pearson correlation: {correlation_p[0]:.3f}, pvalue: {correlation_p[1]:.2f}')

    correlation_s = scipy.stats.spearmanr(line_tokens, delta_pp)
    print(f'Scipy Spearman correlation: {correlation_s.correlation:.3f}, pvalue: {correlation_s.pvalue:.2f}')

    # compute linear regression
    m, b = np.polyfit(line_tokens, delta_pp, 1)
    # color points green if delta_pp > 0, red otherwise
    colors = ['green' if delta < 0 else 'red' for delta
              in delta_pp]

    return line_tokens, delta_pp, m, b, colors, correlation_p, score


def generate_file_paths(test, model):
    return [f"{result_file_dir}{test}_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv.{model}.delta3",
            f"{eval_file_dir}{test}_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv.{model}"]


if __name__ == "__main__":
    result_file_dir = "output/synonyms/result/delta/"
    eval_file_dir = "output/synonyms/eval/"

    tests = ['T11', 'T12', 'T13']
    # tests = ['T11']
    models = ['allenai.OLMo-1B', 'allenai.OLMo-7B', 'llm360.Amber']
    # models = ['allenai.OLMo-1B', 'allenai.OLMo-7B']

    # file1 = result_file_dir + f"{test}_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv.{model}.delta3"
    # file2 = eval_file_dir + f"{test}_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv.{model}"

    tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-1B",
                                                  trust_remote_code=True,
                                                  device='auto')

    file_list = []
    fig, axs = plt.subplots(len(tests), len(models), figsize=(15, 15), sharex=True, sharey=True)

    for i, test in enumerate(tests):
        for j, model in enumerate(models):
            file_list = generate_file_paths(test, model)
            line_tokens, acc, m, b, colors, corr_pearson, score = corr_and_plot(file_list[0], file_list[1], model)

            axs[i][j].scatter(line_tokens, acc, color=colors)
            axs[i][j].plot(line_tokens, m * np.array(line_tokens) + b, color='black')
            axs[i][j].set_title(f"{test}, {model.split('.')[1]}, c={corr_pearson[0]:.3f}, p={corr_pearson[1]:.2f}, acc={score*100:.2f}")

            if i == len(tests) - 1 and j == 1:
                axs[i][j].set_xlabel('Number of tokens')
            if i == 1 and j == 0:
                axs[i][j].set_ylabel('Delta pp')
    plt.show()
