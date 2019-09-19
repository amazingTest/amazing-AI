from flask import Flask
from flask import request
from sentencepattern_attention import playmodel, config
from keras.models import load_model
from utils import utils
import tensorflow as tf
app = Flask(__name__)

mapping = config.mapping

chat_model = load_model('D:/MachineLearning/sentencepattern_attention/model/model8596.h5')
graph = tf.get_default_graph()

default_return = '''<h1>问题句式分类器</h1>
                    <form action="/" method="post">
                    <p><input name="sentence" placeholder="请输入你的问题"></p>
                    <p><button type="submit">点击就送屠龙宝刀</button></p>
                    </form>'''

@app.route('/', methods=['GET'])
def Home():
    return default_return

@app.route('/', methods=['POST'])
def getpatterns():
    global graph
    with graph.as_default():
        # 需要从request对象读取表单内容：
        if request.form['sentence'] and request.form['sentence'].strip() is not '':
            que_vector = playmodel.get_que_vector(request.form['sentence'])
            # print(que_vector)
            if len(que_vector[0]) is not 0:
                predictions = chat_model.predict(que_vector)
                print(predictions[0])
                max_dic = utils.get_max_dic(predictions[0], boundary_value=0.0, num=1)
                print(max_dic)
                pre_result = [mapping[value] for value in max_dic.values()]
                result = '<h3>{}</h3>'.format(pre_result)
                return default_return + result
            else:
                return default_return + '<h3>请重新输入<h3>'
        else:
            return default_return + '<h3>请重新输入<h3>'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)