texts = []
labels = []

with open('./data/sentiment labelled sentences/amazon_cells_labelled.txt') as data:
    for line in data.readlines():
        texts.append(line.strip()[:-1].replace('\t', ''))
        labels.append(int(line.strip()[-1]))


with open('./data/sentiment labelled sentences/imdb_labelled.txt') as data:
    for line in data.readlines():
        texts.append(line.strip()[:-1].replace('\t', ''))
        labels.append(int(line.strip()[-1]))

with open('./data/sentiment labelled sentences/yelp_labelled.txt') as data:
    for line in data.readlines():
        texts.append(line.strip()[:-1].replace('\t', ''))
        labels.append(int(line.strip()[-1]))


if __name__ == '__main__':
    # 数据测试
    print(len(texts) == len(labels))
    print(all(list(map(lambda x: isinstance(x, int), labels))))
    print(all(list(map(lambda x: isinstance(x, str) and x, texts))))







