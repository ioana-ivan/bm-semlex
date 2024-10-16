import codecs
from itertools import zip_longest
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
import pandas as pd

sys.path.insert(0, "C:\\These\\Code\\bm-semlex\\src\\")
from testing_pkg.lm.evaluator_class import TestRecordResult
from sklearn.metrics import matthews_corrcoef


acc_file_dir = "output/synonyms/result/"
tokens_file_dir = "output/synonyms/tokens/"

accuracy_file = acc_file_dir + 'eval_T1_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv.allenai.OLMo-1B'
accuracy_file2 = acc_file_dir + 'eval_T3_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv.allenai.OLMo-1B'
accuracy_file3 = acc_file_dir + 'eval_T4_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv.allenai.OLMo-1B'

tokens_file = tokens_file_dir + 'semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv_tokens_delta.txt'

accs = []
delta_tokens = []

for acc_file in [accuracy_file, accuracy_file2, accuracy_file3]:
    with codecs.open(acc_file, encoding='utf-8', mode='r') as ifile:
        for line in ifile:
            if line == '\n':
                continue
            acc = float(line.strip().split('\t')[1])
            #print(acc)
            accs.append(acc)

    with codecs.open(tokens_file, encoding='utf-8', mode='r') as ifile:
        for line in ifile:
            delta_tokens.append(float(line.strip()))

# sanity check
if len(accs) != len(delta_tokens):
    print(f"Error: itemnb mismatch, accs {len(accs)}, delta {len(delta_tokens)}")
    exit(1)

same = 0
for i in range(len(accs)):
    if accs[i] == delta_tokens[i]:
        same += 1

print(f'Agreement {same / len(accs):.2f}')

# correlations
correlation_p = scipy.stats.pearsonr(accs, delta_tokens)
print(f'Scipy Pearson correlation: {correlation_p[0]:.3f}, pvalue: {correlation_p[1]:.2f}')

correlation_s = scipy.stats.spearmanr(accs, delta_tokens)
print(f'Scipy Spearman correlation: {correlation_s.correlation:.3f}, pvalue: {correlation_s.pvalue:.2f}')

correlation_b = scipy.stats.pointbiserialr(accs, delta_tokens)
print(f'Scipy Point biserial correlation: {correlation_b.correlation:.3f}, pvalue: {correlation_b.pvalue:.2f}')

correlation_k = scipy.stats.kendalltau(accs, delta_tokens)
print(f'Scipy Kendall correlation: {correlation_k.correlation:.3f}, pvalue: {correlation_k.pvalue:.2f}')

matt_corrcoeff = sklearn.metrics.matthews_corrcoef(accs, delta_tokens)
print(f'Sklearn Matthews correlation: {matt_corrcoeff:.3f}')

# compute table
#print(accs)
#print(delta_tokens)

# create pandas dataframe with two columns, accuracy and delta_tokens unique values
unique_accs = list(set(accs))
unique_delta_tokens = list(set(delta_tokens))

#print(unique_accs)
#print(unique_delta_tokens)

# make all possible pairs between unique values of accs and delta_tokens
comb = [(x, y) for x in unique_delta_tokens for y in unique_accs]
print(comb)

df = pd.DataFrame(comb, columns=['delta_tokens', 'accuracy'])

df['values'] = 0

# count occurence of unique items in delta_tokens
count = {x: delta_tokens.count(x) for x in unique_delta_tokens}
print(count) 

zeros = 0
ones = 0

for i in range(len(accs)):
    acc_item = accs[i]
    delta_item = delta_tokens[i]

    if acc_item == 0:
        zeros += 1
    else:
        ones += 1

    # increment value in dataframe where accuracy = acc_item and delta_tokens = delta_item
    value = df.loc[(df['accuracy'] == acc_item) & (df['delta_tokens'] == delta_item), 'values']
    value = value + 1
    df.loc[(df['accuracy'] == acc_item) & (df['delta_tokens'] == delta_item), 'values'] = value
   
print(df)

# check if values in 'values' column add up to 600
print(f"total {df['values'].sum()}")
print(f'zeros {zeros}')
print(f'ones {ones}')
print(f"accuracy moyen {ones/df['values'].sum()}")

# divide values if delta_tokens is zeros by zeros, if not by ones, and so on using count dictionary

for value in unique_delta_tokens:
    df.loc[df['delta_tokens'] == value, 'values'] = df.loc[df['delta_tokens'] == value, 'values'] / count[value]
# df['values'] = np.where(df['delta_tokens'] == 0, df['values'] / count[0], df['values'] / count[1])
df['values'] = df['values'].round(2)

print(df) 

# compute linear regression
m, b = np.polyfit(delta_tokens, accs, 1)

# make 2 correlation graphs : Pearson and Spearman
# make Pearson correlation graph
plt.scatter(delta_tokens, accs)
plt.plot(delta_tokens, m * np.array(delta_tokens) + b, color='red')
plt.xlabel('delta_tokens')
plt.ylabel('accuracy')
plt.title(f'Pearson: {correlation_p[0]:.2f} Spearman: {correlation_s.correlation:.2f}')
# save plot
#plt.savefig('correlation_accs_tokens.png')
# show plot

#plt.show()