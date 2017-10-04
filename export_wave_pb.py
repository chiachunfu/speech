import sugartensor as tf
import numpy as np
import librosa
import tensorflow as tfw
from tensorflow.python.framework import graph_util

from model import *
import data


batch_size = 1     # batch size
voca_size = data.voca_size
x = tf.placeholder(dtype=tf.sg_floatx, shape=(batch_size, None, 20))
# sequence length except zero-padding
seq_len = tf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)
# encode audio feature
logit = get_logit(x, voca_size)
# ctc decoding
decoded, _ = tf.nn.ctc_beam_search_decoder(logit.sg_transpose(perm=[1, 0, 2]), seq_len, merge_repeated=False)
# to dense tensor
y = tf.add(tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values), 1, name="output")

with tf.Session() as sess:
     tf.sg_init(sess)
     saver = tf.train.Saver()
     saver.restore(sess, tf.train.latest_checkpoint('asset/train'))

graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

with tf.Session() as sess:
     tf.sg_init(sess)
     saver = tf.train.Saver()
     saver.restore(sess, tf.train.latest_checkpoint('asset/train'))
     # Output model's graph details for reference.
     tf.train.write_graph(sess.graph_def, '/root/speech-to-text-wavenet/asset/train', 'graph.txt', as_text=True)
     # Freeze the output graph.
     output_graph_def = graph_util.convert_variables_to_constants(sess,input_graph_def,"output".split(","))
     # Write it into .pb file.
     with tfw.gfile.GFile("/root/speech-to-text-wavenet/asset/train/wavenet_model.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())
