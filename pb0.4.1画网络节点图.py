#     tensorboard --logdir=D:\pythonProject\语音合成攻击deepspeech0.4.1\logpb

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.python.platform import gfile

model = './deepspeech-0.4.1-models/output_graph.pb'
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('./logpb', graph)
