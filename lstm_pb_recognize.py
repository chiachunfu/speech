import tensorflow as tf
import numpy as np
import data
import time
from constants import c

# Parameters for index to string
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1

file_name = c.INFERENCE.NPZ_PATH
pb_PATH = c.INFERENCE.PB_PATH

batch_size=1
num_layers=1
num_classes=28
num_features = c.LSTM.FEATURES
num_hidden = c.LSTM.HIDDEN


datafile = np.load(file_name)
wav_inputs = np.expand_dims(datafile['data_in'],axis=0)
data_len = np.asarray([datafile['seq_len'][0]])

with tf.gfile.FastGFile(pb_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Set output tensor
    y = sess.graph.get_tensor_by_name('SparseToDense:0')
    start = time.time()
    labels = sess.run(y, feed_dict={'InputData:0':wav_inputs,
                                    'SeqLen:0':data_len})
    print(time.time()-start)
    str_decoded = ''.join([chr(x) for x in labels[0] + FIRST_INDEX])
    str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
    print(str_decoded)
