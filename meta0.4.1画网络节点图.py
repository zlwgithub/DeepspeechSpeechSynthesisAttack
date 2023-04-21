#     tensorboard --logdir=D:\pythonProject\语音合成攻击deepspeech0.4.1\logmeta

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.python.platform import gfile
#从文件格式为.meta文件加载模型
graph = tf.compat.v1.get_default_graph()
graphdef = graph.as_graph_def()
_ = tf.train.import_meta_graph("./deepspeech-0.4.1-checkpoint/model.v0.4.1.meta")
summary_write = tf.compat.v1.summary.FileWriter("./logmeta" , graph)
