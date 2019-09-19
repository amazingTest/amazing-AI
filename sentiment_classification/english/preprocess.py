# -*- coding: utf-8 -*-

import jieba
import numpy as np
from gensim.models import word2vec
import pandas as pd


# 创建停用词list
def stop_words_list(file_path):
    stopwords = [line.strip() for line in open(file_path, 'r', encoding='utf-8').readlines()]
    return stopwords


# 对句子进行分词
def seg_sentence(sentence, get_right_part_from_last_comma=False, user_dic_path="data/user_dic.txt", stop_words_path="data/stop_words.txt"):
    jieba.load_userdict(user_dic_path)
    chinese_comma_index = 0
    english_comma_index = 0
    try:
        chinese_comma_index = sentence.strip().rindex('，')
        english_comma_index = sentence.strip().rindex(',')
    except BaseException:
        pass
    right_comma_index = max(chinese_comma_index, english_comma_index)
    sentence_seged = jieba.cut(sentence.strip()[right_comma_index:])\
        if get_right_part_from_last_comma else jieba.cut(sentence.strip())
    stopwords = stop_words_list(stop_words_path)  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr.strip()


def vectorize(questions, word2vec_path='G:/MachineLearning/word2vec/word2vec_wx'):
    # questions = map(seg_sentence, questions)
    # 导入训练好的词向量
    model = word2vec.Word2Vec.load(word2vec_path)
    # 将X词向量X_vector
    X_vector = []
    max_len = 0
    num = 0

    test1 = []

    for x_sentence in questions:
        x_sentence = x_sentence.replace('  ', '')
        x_word = x_sentence.split(' ')
        x_sentvec = [model[w] for w in x_word if w in model.wv.vocab]
        test1.append(len(x_sentvec) == len(x_word))

        if len(x_sentvec) > max_len:
            max_len = len(x_sentvec)
        X_vector.append(x_sentvec)
        num += 1
    # 计算词向量的维数
    word_dim = len(X_vector[0][0])
    # print(word_dim)
    sentend = np.zeros((word_dim,), dtype=np.float32)
    for sentvec in X_vector:
        if len(sentvec) < max_len:
            # 补零.
            for i in range(max_len - len(sentvec)):
                sentvec.append(sentend)
    print(max_len)
    print(len([i for i in test1 if i]) / len(test1))
    return X_vector


def get_texts_and_labels():

    df = pd.read_csv(r'G:/MachineLearning/DataSet/sentiment_classification/english/englishSentiment.xls.csv',
                     encoding='ISO-8859-1',
                     names=['label', '1', '2', '3', '4', 'text'])

    return vectorize(df['text']), df['label']


# def word_count(words, word_num=50):
#     dic = {}
#     for word in words:
#         if word not in dic.keys():
#             dic[word] = 1
#         else:
#             dic[word] += 1
#     word_count = sorted(dic.items(), key=lambda x: x[1], reverse=True)
#     if word_num < 1:
#         return None
#     else:
#         return word_count[0: int(word_num)]
#
#
# def placeholder_words_set(data_path, label, title='question', sheet_name='Sheet1',
#                           placeholder_words_path='./data/placeholder_words_{}.txt'):
#     df = pd.read_excel(data_path, sheet_name=sheet_name)
#     df[title] = df["question"].map(lambda x: seg_sentence(x))
#     placeholder_words_list = []
#     words = []
#     for i in df.loc[(df.pattern == label), title]:
#         for j in i.split(' '):
#             if str(j.strip()) is not '':
#                 words.append(j)
#     num = 0
#     for i in [i[1] for i in utils.word_count(words, return_all=True)]:
#         if i < int(len(words) / len(set(words))):
#             placeholder_words_list.append(utils.word_count(words, return_all=True)[num][0])
#         num += 1
#     file = open(placeholder_words_path.format(list(mapping.values()).index(label)), 'w', encoding='gbk')
#     file.writelines([i.strip()+'\n' for i in set(placeholder_words_list)])


if __name__ == '__main__':
    q, p = get_texts_and_labels()
    print(len(list(q)))

