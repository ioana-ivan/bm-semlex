import codecs
from itertools import zip_longest

input_file1 = "output/synonyms/test/human/t1_pp_man_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv"
input_file2 = "output/synonyms/test/human/t1_pp_man_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv.ans"
output_file = "2_or_3.txt"

# get answers
list_ans = []
with codecs.open(input_file2, encoding='utf-8', mode='r') as efile:
    for line in efile:
        line_stuffs = line.split(', ')
        for item in line_stuffs:
            if item != '':
                answer = item.split('=')[1]
                list_ans.append(answer)
                print(f'Ans {answer}')

i = 0
with codecs.open(input_file1, encoding='utf-8', mode='r') as ifile:
    with codecs.open(output_file, encoding='utf-8', mode='w') as pfile:
        for next_lines in zip_longest(*[ifile] * 3):
            ref = next_lines[0].strip().split('\t')[1]
            first = next_lines[1].strip().split('\t')[1]
            second = next_lines[2].strip().split('\t')[1]
            print(f"Answer {list_ans[i]}")

            if first == list_ans[i]:
                pfile.write("2")
            elif second == list_ans[i]:
                pfile.write("3")
            else:
                raise ValueError(f"Something weird happened at lines {ref} {first} {second}")
            pfile.write('\n')

            i += 1


