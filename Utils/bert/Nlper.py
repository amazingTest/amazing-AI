import requests
import json
import numpy as np


class Nlper:

    def __init__(self, bert_client):
        self.bert_client = bert_client

    def get_text_similarity(self, base_text, compared_text, algorithm='cosine', magic_cut=True):
        if isinstance(algorithm, str) and algorithm.lower() == 'cosine':
            arrays = self.bert_client.encode([base_text, compared_text])
            magic_array = (arrays[0] + arrays[1]) / np.pi
            arrays = [arrays[0] - magic_array, arrays[1] - magic_array] if magic_cut else arrays
            norm_1 = np.linalg.norm(arrays[0])
            norm_2 = np.linalg.norm(arrays[1])
            dot_product = np.dot(arrays[0], arrays[1])
            similarity = round(0.5 + 0.5 * (dot_product / (norm_1 * norm_2)), 2)
            return similarity


if __name__ == '__main__':
    pass
    # from bert_serving.client import BertClient
    # bc = BertClient(ip='127.0.0.1',timeout=5)
    # nlper = Nlper()
    # print(nlper.get_text_similarity('ok', '成功', magic_cut=True))




