import os
from keras.models import load_model
from gensim.models import word2vec
import numpy as np
from Utils import utils
from retentionRateWithSearchKeywords import preprocess
# os.chdir('D:\PycharmProjects\chatbot\chatbot')
chat_model = load_model('./model/model_best.h5')
wordVector_model = word2vec.Word2Vec.load('D:/MachineLearning/word2vec/word2vec_wx')


def get_que_vector(input_sentence):
    question = input_sentence.strip()
    que_list = preprocess.seg_sentence(question).split(' ')
    print('分词、去停用词结果: ' + str(que_list))
    que_vector = [wordVector_model[w] for w in que_list if w in wordVector_model.wv.vocab]
    # print(len(que_vector))
    # 获得单词的维度
    word_dim = 0
    try:
        word_dim = len(que_vector[0])
    except BaseException:
        print('很抱歉这题我不会，下一题')
    sentend = np.zeros((word_dim,), dtype=np.float32)
    if len(que_vector) > 23:
        que_vector[24:] = []
    else:
        for i in range(23 - len(que_vector)):
            que_vector.append(sentend)
    que_vector = np.array([que_vector])
    # que_vector = que_vector.reshape(len(que_vector), -1)
    return que_vector


if __name__ == '__main__':
    while True:
        question = input('输入问题: ')
        question = question.strip()
        que_list = preprocess.seg_sentence(question).split(' ')
        print('分词、去停用词结果: ' + str(que_list))
        que_vector = [wordVector_model[w] for w in que_list if w in wordVector_model.wv.vocab]
        # print(len(que_vector))
        # 获得单词的维度
        try:
            word_dim = len(que_vector[0])
        except BaseException:
            print('很抱歉这题我不会，下一题')
            continue
        sentend = np.zeros((word_dim,), dtype=np.float32)
        if len(que_vector) > 23:
            que_vector[24:] = []
        else:
            for i in range(23 - len(que_vector)):
                que_vector.append(sentend)
        que_vector = np.array([que_vector])
        # que_vector = que_vector.reshape(len(que_vector), -1)
        # print(que_vector.shape)
        # 预测答句
        predictions = chat_model.predict(que_vector)
        # print(predictions[0])
        # return [mapping[value] for value in utils.get_max_dic(predictions[0]).values()]
        print((predictions[0]))
