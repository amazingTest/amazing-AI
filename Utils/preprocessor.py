import numpy as np
from gensim.models import word2vec
import jieba


class Preprocessor:

    '''

        >>> placeholder_words_test_file = open('placeholder_words_test.txt', 'w', encoding='utf-8')
        >>> placeholder_words_test_file.writelines(['你好\\n', '您好\\n', '你们好\\n' ])
        >>> placeholder_words_test_file.close()
        >>> user_words_test_file = open('user_words_test.txt', 'w', encoding='utf-8')
        >>> user_words_test_file.writelines(['超级人工咨询\\n', '超级智能咨询\\n', '超级人工智能咨询\\n' ])
        >>> user_words_test_file.close()
        >>> stop_words_test_file = open('stop_words_test.txt', 'w', encoding='utf-8')
        >>> stop_words_test_file.writelines(['，\\n', '。\\n', '？\\n' ])
        >>> stop_words_test_file.close()
        >>> word2vec_model_path = 'G:/MachineLearning/word2vec/word2vec_wx'
        >>> pre_processor = Preprocessor(user_words_path='./user_words_test.txt', stop_words_path='./stop_words_test.txt',\
                            placeholder_words_path='./placeholder_words_test.txt',\
                            word2vec_model_path=word2vec_model_path)
        >>> stop_words_list, placeholder_words_list = pre_processor.get_words_lists()
        >>> stop_words_list
        ['，', '。', '？']
        >>> placeholder_words_list
        ['你好', '您好', '你们好']
        >>> sentence = '你好，您好，你们好。请问你们需要超级人工咨询，超级智能咨询，超级人工智能咨询吗？'
        >>> sentence_words = pre_processor.cut_sentence_to_words(sentence)
        >>> sentence_words
        [' ', ' ', ' ', '请问', '你们', '需要', '超级人工咨询', '超级智能咨询', '超级人工智能咨询', '吗']
        >>> tensor = pre_processor.sentences2tensor([sentence])
        >>> len(tensor[0]) == len(sentence_words)
        True
        >>> filled_sentence_end_tensor = pre_processor.get_sentences_tensor([sentence, sentence+sentence],\
                                                                            is_fill_sentence_end=True)
        >>> len(filled_sentence_end_tensor[0]) == len(filled_sentence_end_tensor[1])
        True
        >>> import os
        >>> os.remove('./placeholder_words_test.txt') if os.path.exists('./placeholder_words_test.txt') else None
        >>> os.remove('./stop_words_test.txt') if os.path.exists('./stop_words_test.txt') else None
        >>> os.remove('./user_words_test.txt') if os.path.exists('./user_words_test.txt') else None

    '''

    def __init__(self, word2vec_model_path=None, user_words_path=None, stop_words_path=None, placeholder_words_path=None,
                 use_sentence_latter_part_of_last_comma=False):

        import warnings

        try:
            self.word2vec_model = word2vec.Word2Vec.load(word2vec_model_path) if word2vec_model_path else None
        except BaseException as e:
            raise TypeError('set up word2vec model failed:  %s' % e)
        if not self.word2vec_model:
            warnings.warn('u probably need a word2vec model, u can set it to self.word2vec_model')
        self.user_words_path = user_words_path
        self.stop_words_path = stop_words_path
        self.placeholder_words_path = placeholder_words_path
        self.use_sentence_latter_part_of_last_comma = use_sentence_latter_part_of_last_comma

    # 将句子列表中所有句子张量化(默认均衡句子词向量维度)
    def get_sentences_tensor(self, sentences, is_fill_sentence_end=True):
        '''

        :param sentences: sentence list<list>
        :param is_fill_sentence_end: whether to make dimension of sentence consistent with each other
        :return: tensor <list>
        '''
        tensor = self.sentences2tensor(sentences)

        tensor = Preprocessor.fill_sentence_end(tensor, word_dim=len(tensor[0][0]),
                                                sentence_dim=max([len(sent_vec) for sent_vec in tensor])) if\
                                                is_fill_sentence_end else tensor
        return tensor

    # 句子列表 -> 张量
    def sentences2tensor(self, sentences):

        '''

        :param sentences: sentences u want to change to tensor
        :return:  tensor <list>
        '''

        for index, item in enumerate(sentences):
            sentences[index] = self.cut_sentence_to_words(item)

        tensor = []
        for sentence in sentences:
            sent_vec = [self.word2vec_model[w] if w in self.word2vec_model.wv.vocab else
                        self.word2vec_model[' '] for w in sentence]
            tensor.append(sent_vec)

        return tensor

    # 切割句子
    def cut_sentence_to_words(self, sentence):

        '''

        :param sentence: sentence that u want to cut in words<str>
        :return: words <list>
        '''

        if not isinstance(sentence, str):
            try:
                sentence = str(sentence)
            except BaseException:
                raise TypeError('sentence must able to be string!')

        stop_words, placeholder_words = self.get_words_lists()
        sentence = Preprocessor.get_latter_part_of_last_comma(sentence) \
            if self.use_sentence_latter_part_of_last_comma else sentence
        sentence = Preprocessor.get_no_space_str(sentence)

        jieba.load_userdict(self.user_words_path) if self.user_words_path else None

        cutted_sentence = jieba.cut(sentence)

        returned_words = []
        for index, word in enumerate(cutted_sentence):
            if word in placeholder_words:
                returned_words.append(' ')
                continue
            returned_words.append(word) if word not in stop_words else None

        return returned_words

    # 根据文件目录获取词列表
    def get_words_lists(self):

        '''

        :return: stop_words<list>, placeholder_words<list>
        '''

        stop_words = [line.strip() for line in open(self.stop_words_path, 'r', encoding='utf-8').readlines()] \
            if self.stop_words_path else []
        placeholder_words = [line.strip() for line in open(self.placeholder_words_path, 'r', encoding='utf-8')
                             .readlines()] if self.placeholder_words_path else []
        return stop_words, placeholder_words

    # 获取句子对相似度
    def get_sentences_pair_similarity(self, base_sentence, compared_sentences):
        pass

    # 获取词组对相似度
    def get_words_pair_similarity(self, base_word, compared_words):
        pass

    # 均衡句子维度
    @staticmethod
    def fill_sentence_end(tensor, word_dim, sentence_dim):
        sentence_end = np.zeros((word_dim,), dtype=np.float32)

        # 补零
        for t in tensor:
            if len(t) < sentence_dim:
                for i in range(sentence_dim - len(t)):
                    t.append(sentence_end)
        return tensor

    # 去除字符串所有空格
    @staticmethod
    def get_no_space_str(s):
        s = s.replace(' ', '').replace('\t', '').replace('\n', '')
        return s

    # 截取字符串中 最后逗号 后面的部分，如不存在，则返回原始字符串。
    @staticmethod
    def get_latter_part_of_last_comma(s):
        s = s.strip()
        chinese_comma_index = 0
        english_comma_index = 0
        try:
            chinese_comma_index = s.rindex('，')
        except ValueError:
            pass
        try:
            english_comma_index = s.rindex(',')
        except ValueError:
            pass
        right_comma_index = max(chinese_comma_index, english_comma_index)
        return s[right_comma_index:]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    # from bert_serving.client import BertClient
    # import numpy as np
    #
    # bc = BertClient()
    # test = bc.encode(['广东品骏快递有限公司佛山分公司', '深圳品骏快递有限公司佛山分公司'])
    #
    # from gensim.models import word2vec
    # model = word2vec.Word2Vec.load('D:/MachineLearning/word2vec/word2vec_wx')
    # print(model[' '])
    # print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5))
    # restrict_vocab = 10000
    # print(model.most_similar(positive='热', topn=5))  # 直接给入词
    # print(model.doesnt_match("breakfast cereal dinner lunch".split()))
    # print(model.similarity(test[0], test[1]))



