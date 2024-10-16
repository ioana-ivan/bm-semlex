import codecs
from itertools import zip_longest
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import pandas as pd

sys.path.insert(0, "C:\\These\\Code\\bm-semlex\\src\\")
from testing_pkg.lm.evaluator_class import TestRecordResult

test_file_dir = "output/synonyms/eval/"

test_file_1 = test_file_dir + 'T3_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv.allenai.OLMo-1B'
test_file_2 = test_file_dir + 'T4_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv.allenai.OLMo-1B'
nb_lines_test = 3

deltas_1 = []
deltas_2 = []

with codecs.open(test_file_1, encoding='utf-8', mode='r') as ifile:
    for next_lines in zip_longest(*[ifile] * nb_lines_test):
        ref = TestRecordResult(next_lines[0].strip().split('\t'))
        syn = TestRecordResult(next_lines[1].strip().split('\t'))
        other = TestRecordResult(next_lines[2].strip().split('\t'))

        itemnb = ref.testitem.itemnb
        delta = syn.score - other.score
        deltas_1.append((itemnb, delta))

with codecs.open(test_file_2, encoding='utf-8', mode='r') as ifile:
    for next_lines in zip_longest(*[ifile] * nb_lines_test):
        ref = TestRecordResult(next_lines[0].strip().split('\t'))
        syn = TestRecordResult(next_lines[1].strip().split('\t'))
        other = TestRecordResult(next_lines[2].strip().split('\t'))

        itemnb = ref.testitem.itemnb
        delta = float(syn.score - other.score)
        deltas_2.append((itemnb, delta))

# sanity check
for i in range(len(deltas_1)):
    if deltas_1[i][0] != deltas_2[i][0]:
        print("Error: itemnb mismatch")
        exit(1)

deltas_1_list = [delta for _, delta in deltas_1]
#print(deltas_1_list[:10])

# normalize deltas_1_list
#deltas_1_list = [delta / max(deltas_1_list) for delta in deltas_1_list]
#print('norm:', deltas_1_list[:10])

deltas_2_list = [delta for _, delta in deltas_2]
#print(deltas_2_list[:10])

# normalize deltas_2_list
#deltas_2_list = [delta / max(deltas_2_list) for delta in deltas_2_list]
#print('norm:', deltas_2_list[:10])

#merge both lists in one containing all values from both
all_deltas = deltas_1_list + deltas_2_list

# correlations
correlation_p = scipy.stats.pearsonr(deltas_1_list, deltas_2_list)
print(f'Scipy Pearson correlation: {correlation_p[0]:.3f}, pvalue: {correlation_p[1]:.2f}')

correlation_s = scipy.stats.spearmanr(deltas_1_list, deltas_2_list)
print(f'Scipy Spearman correlation: {correlation_s.correlation:.3f}, pvalue: {correlation_s.pvalue:.2f}')

# compute linear regression
m, b = np.polyfit(deltas_1_list, deltas_2_list, 1)

# make 2 correlation graphs : Pearson and Spearman
# make Pearson correlation graph
#colors = iter([plt.cm.Set2(i) for i in [5, 6, 7]])
# colors = iter([plt.cm.tab20(i) for i in [0, 2, 10, 12]]) # main only
# colors = iter([plt.cm.tab20(i) for i in [10, 12, 14]]) # rel only
# axs[k].set_prop_cycle(cycler('color', plt.cm.tab20.colors))
# set color to plot

'''
plt.scatter(deltas_1_list[:201], deltas_2_list[:201])
plt.scatter(deltas_1_list[201:401], deltas_2_list[201:401])
plt.scatter(deltas_1_list[401:], deltas_2_list[401:])
plt.plot(deltas_1_list, m * np.array(deltas_1_list) + b, color='red')
plt.xlabel('Perplexity delta sub')
plt.ylabel('Perplexity delta rel_ctx')
plt.title(f'Pearson: {correlation_p[0]:.2f} Spearman: {correlation_s.correlation:.2f}')
# save plot
# plt.savefig('correlation_sub_rel_O1O7A.png')
# show plot
plt.show()
'''

deltas_list = list(zip(deltas_1_list, deltas_2_list))
df = pd.DataFrame(deltas_list)
print(len(deltas_1_list))

x_name = u"Relation (pp Δ)"
y_name = u"Reference (pp Δ)"

df.columns = [x_name, y_name]

sns.set_theme()
sns.set_style("whitegrid")
sns.color_palette(palette='Set2')

plt.xlim(min(all_deltas) - 10, max(all_deltas) + 10)
plt.ylim(min(all_deltas) - 10, max(all_deltas) + 10)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.set_title('Pearson: {:.2f} Spearman: {:.2f}'.format(correlation_p[0], correlation_s.correlation))
#ax.set_title('Pearson: {:.2f} p-value: {:.2f}'.format(correlation_p[0], correlation_p[1]))

sns.regplot(
  data=df,
  x=x_name,
  y=y_name,
  #scatter_kws={"color": "gray"},
  line_kws={"color": "red"}
)

plt.show()

items = [float(item) for item, _ in deltas_1]
#print(items)

# make matrix from 3 arrays
items = np.array(items)
deltas_1_list = np.array(deltas_1_list)
deltas_2_list = np.array(deltas_2_list)

# combine into matrix
matrix = np.vstack((items, deltas_1_list, deltas_2_list))
#print(matrix)

# make a correltion matrix
correlation_matrix = np.corrcoef(matrix).round(decimals=2)
#print(correlation_matrix)

# make a heatmap
#plt.matshow(correlation_matrix)
#plt.show()