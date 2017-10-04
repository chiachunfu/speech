import tensorflow as tf
from tensorflow.python.framework import graph_util
from constants import c

model_folder ='Path/to/your/model/folder'

num_features = c.LSTM.FEATURES
num_hidden = c.LSTM.HIDDEN
batch_size=1
num_layers=1
num_classes=28

# Construct the graph. For detail comments, please see lstm_ctc.py
inputs = tf.placeholder(tf.float32, [batch_size, None, num_features], name='InputData')
targets = tf.sparse_placeholder(tf.int32, name='LabelData')
seq_len = tf.placeholder(tf.int32, [None], name='SeqLen')

cell = tf.contrib.rnn.LSTMCell(num_hidden)
stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,state_is_tuple=True)
outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32, time_major =False)
shape = tf.shape(inputs)
batch_s, max_time_steps = shape[0], shape[1]
outputs = tf.reshape(outputs, [-1, num_hidden])
W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
b = tf.Variable(tf.constant(0., shape=[num_classes]))
logits = tf.matmul(outputs, W) + b
logits = tf.reshape(logits, [batch_s, -1, num_classes])
logits = tf.transpose(logits, (1, 0, 2))
decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)
y = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values)

# Need output node name to convert checkpoint into protocol buffers.
output_node_names = "SparseToDense"

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    # Restore checkpoint (load weights and biases)
    saver.restore(sess, tf.train.latest_checkpoint(model_folder))

graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

with tf.Session() as sess:
     tf.global_variables_initializer().run()
     saver = tf.train.Saver()
     saver.restore(sess, tf.train.latest_checkpoint(model_folder))

     # Output model's graph details for reference.
     tf.train.write_graph(sess.graph_def, model_folder, 'graph_lstm.txt', as_text=True)
     # Freeze the output graph.
     output_graph_def = graph_util.convert_variables_to_constants(sess,input_graph_def, \
                                                            output_node_names.split(","))
     # Write it into .pb file.
     with tf.gfile.GFile("/Users/chiachunfu/Desktop/INSIGHT/Project/lstm_model.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())
