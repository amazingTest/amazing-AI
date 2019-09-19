# -*- coding: utf-8 -*-
import time
from sklearn.model_selection import train_test_split
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping
# from imblearn.over_sampling import SMOTE
from src.sentiment_classification.english import preprocess
# 优化方法选用Adam(其实可选项有很多，如SGD)
from keras.optimizers import Adam
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import keras

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('— val_f1: %f — val_precision: %f — val_recall %f' % (_val_f1, _val_precision, _val_recall))
        return


_metrics = Metrics()

INPUT_DIM = 256
TIME_STEPS = 54
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False

# 有很多新数据时再使用
# for label in list(mapping.values()):
#     print('正在录入占位符文本: ' + label)
#     preprocess.placeholder_words_set(data_path=r'./data/疑问句标注.xlsx', label=label)


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='sigmoid')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = keras.layers.multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


questions, patterns = preprocess.get_texts_and_labels()
X_train, X_test, y_train, y_test = train_test_split(questions, patterns, test_size=0.2, stratify=patterns,
                                                    random_state=555)

# 过采样
# oversampler = SMOTE(ratio='auto', random_state=np.random.randint(100), k_neighbors=5, m_neighbors=10, kind='regular')

X_train = np.asarray(X_train, dtype='float64')
X_test = np.asarray(X_test, dtype='float64')
y_train = np.asarray(y_train, dtype='float64')
y_test = np.asarray(y_test, dtype='float64')

# 过采样，输入一维数据
# os_X_train, os_y_train = oversampler.fit_sample(X_train.reshape(len(X_train), -1), y_train)
# os_X_train = np.array(os_X_train)
# 转换为one-hot-label
# os_y_train = keras.utils.to_categorical(os_y_train)
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
# 过采样后转回二维数据(输入LSTM)
# os_X_train = os_X_train.reshape((os_X_train.shape[0], -1, 256))
# print(os_X_train.shape)
# print(os_y_train.shape)
# X_train = X_train.reshape(len(X_train), -1)
# X_test = X_test.reshape(len(X_test), -1)
# print(X_test.shape)
# print(os_y_train.shape)
# print(y_test.shape)
# 初始化一个模型, 拿全连接试试水
# model = Sequential()
# model.add(Dense(units=1000, input_shape=(X_train.shape[1], )))
# model.add(Activation('sigmoid'))
# model.add(Dropout(0.5))
# model.add(Dense(units=1000, input_shape=(X_train.shape[1], )))
# model.add(Activation('elu'))
# model.add(Dropout(0.5))
# model.add(Dense(units=y_train.shape[1], activation='sigmoid'))
# # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(lr=4e-4, decay=5e-6),
#               metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=2, mode='min', min_delta=0.5)
inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
drop_out_1 = Dropout(0.35)(inputs)
lstm_out = Bidirectional(LSTM(54, return_sequences=True), merge_mode='concat')(drop_out_1)
attention_mul = attention_3d_block(lstm_out)
attention_flatten = Flatten()(attention_mul)
output = Dense(units=y_train.shape[1], activation='sigmoid')(attention_flatten)
model = Model(inputs=inputs, outputs=output)
model.compile(loss='categorical_crossentropy',
               optimizer=Adam(lr=2.55e-4, decay=5e-6),
               metrics=['accuracy'])
# 训练、保存模型
model.fit(X_train, y_train, nb_epoch=50, validation_data=(X_test, y_test), shuffle=True, callbacks=[es, _metrics])
model.save('./model/model{}.h5'.format(str(int(time.time()))))

