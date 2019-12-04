import pandas as pd
import math

test_file = open('data/sts-test.csv', 'r')
records = []
for line in test_file:
    line = line.split('\t')
    records.append([line[5], line[6], line[4]])



test_set = pd.DataFrame.from_records(data=records, columns=['sent1', 'sent2', 'gold'])
results = pd.read_csv('output/xlnet_8958_Relu.txt', names=['preds'])

print(test_set.shape, results.shape)
print(results.head())
test_set = pd.concat([test_set, results], axis=1)
test_set['dif'] = test_set.apply(lambda x: math.fabs(float(x['gold']) - float(x['preds'])), axis=1)
print(test_set[['dif', 'gold', 'preds']].head())
test_set.sort_values(by='dif', ascending=False, inplace=True)

test_set.to_csv('output/xlnet_error_analysis.tsv', index=False, sep='\t')