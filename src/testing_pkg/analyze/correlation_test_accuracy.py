import codecs
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
import pandas as pd

# sys.path.insert(0, "C:\\These\\Code\\bm-semlex\\src\\")

acc_file_dir = "output/synonyms/result/eval/"

accuracy_file1 = acc_file_dir + 'T12_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv.allenai.OLMo-1B.eval3'
accuracy_file2 = acc_file_dir + 'T13_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv.allenai.OLMo-1B.eval3'
accs1 = []
accs2 = []

with codecs.open(accuracy_file1, encoding='utf-8', mode='r') as ifile:
    for line in ifile:
        if line == '\n':
            continue
        acc = float(line.strip())
        #print(acc)
        accs1.append(acc)

with codecs.open(accuracy_file2, encoding='utf-8', mode='r') as ifile:
    for line in ifile:
        if line == '\n':
            continue
        acc = float(line.strip())
        #print(acc)
        accs2.append(acc)

# sanity check
if len(accs1) != len(accs2):
    print(f"Error: itemnb mismatch, accs {len(accs1)}, delta {len(accs2)}")
    exit(1)

same = 0
for i in range(len(accs1)):
    if accs1[i] == accs2[i]:
        same += 1

print(f'Agreement {same / len(accs1):.2f}')

# correlations
correlation_p = scipy.stats.pearsonr(accs1, accs2)
print(f'Scipy Pearson correlation: {correlation_p[0]:.3f}, pvalue: {correlation_p[1]:.2f}')

correlation_s = scipy.stats.spearmanr(accs1, accs2)
print(f'Scipy Spearman correlation: {correlation_s.correlation:.3f}, pvalue: {correlation_s.pvalue:.2f}')

correlation_b = scipy.stats.pointbiserialr(accs1, accs2)
print(f'Scipy Point biserial correlation: {correlation_b.correlation:.3f}, pvalue: {correlation_b.pvalue:.2f}')

correlation_k = scipy.stats.kendalltau(accs1, accs2)
print(f'Scipy Kendall correlation: {correlation_k.correlation:.3f}, pvalue: {correlation_k.pvalue:.2f}')

#matt_corrcoeff = sklearn.metrics.matthews_corrcoef(accs1, accs2)
#print(f'Sklearn Matthews correlation: {matt_corrcoeff:.3f}')

# compute table
#print(accs)
#print(accs2)

# create pandas dataframe with two columns, accuracy and accs2 unique values
unique_accs1 = list(set(accs1))
unique_accs2 = list(set(accs2))

#print(unique_accs1)
#print(unique_accs2)

# make all possible pairs between unique values of accs and accs2
comb = [(x, y) for x in unique_accs1 for y in unique_accs2]
print(comb)

df = pd.DataFrame(comb, columns=['accs1', 'accs2'])

df['values'] = 0
df['count'] = 0

# count occurence of unique items in second test
count1 = {x: accs1.count(x) for x in unique_accs1}
print(f"Zeros and ones count for first test {count1}, acc first test {count1[1]/len(accs1)}")

#count2 = {x: accs2.count(x) for x in unique_accs2}
#print(f"Zeros and ones count for second test {count2}, acc second test {count2[1]/len(accs2)}")

zeros = 0
ones = 0

for i in range(len(accs1)):
    acc1_item = accs1[i]
    acc2_item = accs2[i]

    if acc1_item == 0:
        zeros += 1
    else:
        ones += 1

    # increment value in dataframe where accs1 = acc1_item and accs2 = acc2_item
    value = df.loc[(df['accs1'] == acc1_item) & (df['accs2'] == acc2_item), 'count']
    value = value + 1
    df.loc[(df['accs1'] == acc1_item) & (df['accs2'] == acc2_item), 'count'] = value
   
print(df)

# check if values in 'values' column add up to 600
print(f"total {df['count'].sum()}")
print(f'zeros {zeros}')
print(f'ones {ones}')
print(f"accuracy moyen {ones/df['count'].sum()}")

# divide values if accs1 is zero by zeros count, if not by ones, and so on using count dictionary

for value in unique_accs1:
    df.loc[df['accs1'] == value, 'values'] = df.loc[df['accs1'] == value, 'count'] / count1[value]
# df['values'] = np.where(df['accs2'] == 0, df['values'] / count[0], df['values'] / count[1])
df['values'] = df['values'].round(2)

print(df) 

# compute linear regression
m, b = np.polyfit(accs2, accs1, 1)

# make 2 correlation graphs : Pearson and Spearman
# make Pearson correlation graph
plt.scatter(accs2, accs1)
plt.plot(accs2, m * np.array(accs2) + b, color='red')
plt.xlabel('accs2')
plt.ylabel('accuracy')
plt.title(f'Pearson: {correlation_p[0]:.2f} Spearman: {correlation_s.correlation:.2f}')
# save plot
#plt.savefig('correlation_accs_tokens.png')
# show plot

#plt.show()