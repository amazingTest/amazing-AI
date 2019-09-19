

import pandas as pd

csv_data = pd.read_csv('./data/sentiment_chinese.txt')
labels = list(csv_data['labels'])
texts = list(csv_data['text'])

print(texts)
print(len(list(filter(lambda x: x.strip() == '', texts))))
print(len(list(filter(lambda x: x not in ['positive', 'negative'], labels))))








