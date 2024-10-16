import codecs
from matplotlib import pyplot as plt
from numpy import mean
import scipy.stats as stats


def read_accuracy_file(file_path):
    accs = []
    with codecs.open(file_path, encoding='utf-8', mode='r') as ifile:
        for line in ifile:
            if line.strip() == '':
                continue
            acc = float(line.strip())
            accs.append(acc)
    return accs


def compute_t_test(file1, file2):
    accs1 = read_accuracy_file(file1)
    accs2 = read_accuracy_file(file2)

    #print(f'Sanity check: {mean(accs1)} {mean(accs2)}')

    # Sanity check
    if len(accs1) != 200 or len(accs2) != 200:
        raise ValueError(f"Error: Each file must contain exactly 200 items. Found {len(accs1)} and {len(accs2)}.")

    # Compute t-test
    #print(f"Variation accs1 {stats.variation(accs1)} accs2 {stats.variation(accs2)}")
    result = stats.ttest_rel(accs1, accs2)
    #print(f'Confidence interval: {result.confidence_interval(confidence_level=0.95)}')
    return accs1, accs2, result.statistic, result.pvalue


if __name__ == "__main__":
    acc_file_dir = "output/synonyms/result/eval/"

    # between models

    '''
    controlled = 'eval3'
    test = 'T12_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv'

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    for i, (model1, model2) in enumerate([('allenai.OLMo-1B', 'allenai.OLMo-7B'), ('allenai.OLMo-1B', 'llm360.Amber'),
                                          ('allenai.OLMo-7B', 'llm360.Amber')]):
        accuracy_file1 = acc_file_dir + f'{test}.{model1}.{controlled}'
        accuracy_file2 = acc_file_dir + f'{test}.{model2}.{controlled}'
        accs1, accs2, t, p = compute_t_test(accuracy_file1, accuracy_file2)
        # print(f"{model1.split('.')[1]}\t{model2.split('.')[1]}\t{t}\t{p}")
        print(f"{model1.split('.')[1]}, {model2.split('.')[1]}, {t:.2f}, {p:.3f}")

        # make a figure containing three histograms, each for one pair pf models
        n1, bins1, patches1 = axs[i].hist(accs1, bins=50, label=model1)
        n2, bins2, patches2 = axs[i].hist(accs2, bins=50, label=model2, alpha=0.5)
        axs[i].set_title(f'{model1.split(".")[1]} vs {model2.split(".")[1]}')
        axs[i].set_xlabel('Delta pp')
        axs[i].set_ylabel('Frequency')

    # plt.show()

    '''
    # between conditions, same model
    # models = ['allenai.OLMo-1B', 'allenai.OLMo-7B', 'llm360.Amber']
    models = ['allenai.OLMo-1B', 'allenai.OLMo-7B', 'llm360.Amber']
    condition = ['eval3', 'eval3']

    for model in models:
        accuracy_file1 = acc_file_dir + f'T7_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv.{model}.{condition[0]}'
        accuracy_file2 = acc_file_dir + f'T8_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv.{model}.{condition[1]}'
        accs1, accs2, t, p = compute_t_test(accuracy_file1, accuracy_file2)
        print(f"{t:.2f}, {p:.3f}")
    
