from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import numpy as np
import csv
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import os
import tempfile


class DataSet(object):

    def __init__(self, input, labels, dtype=tf.float32):
        """Construct a DataSet."""

        self._input = input
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = 0

    @property
    def input(self):
        return self._input

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._input = self._input[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._input[start:end], self._labels[start:end]

    def set_examples(self, num):
        self._num_examples = num


def read_data_sets(dtype=tf.float32):
    print "Begin Reading Data"
    class DataSets(object):
        pass
    data_sets = DataSets()

    train_input = []
    train_labels = []
    test_input = []
    test_labels = []
    count = 0

    filename_queue = tf.train.string_input_producer(["data/learn.csv"])

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    default_values = [[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]]
    limit_bal,sex,education1,education2,education3,education4,marriage1,marriage2,marriage3,age,pay_1,pay_2,pay_3,pay_4,pay_5,pay_6, bill_amt1,bill_amt2,bill_amt3,bill_amt4,bill_amt5,bill_amt6,pay_amt1,pay_amt2,pay_amt3,pay_amt4,pay_amt5,pay_amt6,default = tf.decode_csv(value, record_defaults=default_values)
    features = tf.pack([limit_bal,sex,education1,education2,education3,education4,marriage1,marriage2,marriage3,age,pay_1,pay_2,pay_3,pay_4,pay_5,pay_6, bill_amt1,bill_amt2,bill_amt3,bill_amt4,bill_amt5,bill_amt6,pay_amt1,pay_amt2,pay_amt3,pay_amt4,pay_amt5,pay_amt6])

    print "Starting train data"

    with tf.Session() as sess:
    # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(20060):
            # Retrieve a single instance:
            example, label = sess.run([features, default])
            train_input.append(example)
            if int(label) == 0:
                label = [1,0]
            else:
                label = [0,1]
            train_labels.append(label)
            #print sess.run([features, default])

        coord.request_stop()
        coord.join(threads)

    filename_queue = tf.train.string_input_producer(["data/test.csv"])

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    default_values = [[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]]
    limit_bal,sex,education1,education2,education3,education4,marriage1,marriage2,marriage3,age,pay_1,pay_2,pay_3,pay_4,pay_5,pay_6, bill_amt1,bill_amt2,bill_amt3,bill_amt4,bill_amt5,bill_amt6,pay_amt1,pay_amt2,pay_amt3,pay_amt4,pay_amt5,pay_amt6,default = tf.decode_csv(value, record_defaults=default_values)
    features = tf.pack([limit_bal,sex,education1,education2,education3,education4,marriage1,marriage2,marriage3,age,pay_1,pay_2,pay_3,pay_4,pay_5,pay_6, bill_amt1,bill_amt2,bill_amt3,bill_amt4,bill_amt5,bill_amt6,pay_amt1,pay_amt2,pay_amt3,pay_amt4,pay_amt5,pay_amt6])

    print "Starting test data"

    with tf.Session() as sess:
    # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(9940):
        # Retrieve a single instance:
            example, label = sess.run([features, default])
            test_input.append(example)
            if int(label) == 0:
                count+=1
                label = [1,0]
            else:
                label = [0,1]
            test_labels.append(label)

        coord.request_stop()
        coord.join(threads)

    train_input = np.array(train_input)
    train_labels = np.array(train_labels)
    test_input = np.array(test_input)
    test_labels = np.array(test_labels)

    data_sets.train = DataSet(train_input, train_labels, dtype=dtype)
    data_sets.test = DataSet(test_input, test_labels, dtype=dtype)

    return data_sets

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

credit = read_data_sets()
credit.train.set_examples(20060)
credit.test.set_examples(9940)

def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)

trX, trY, teX, teY = credit.train.input, credit.train.labels, credit.test.input, credit.test.labels

X = tf.placeholder("float", [None, 28])
Y = tf.placeholder("float", [None, 2])

w_h = init_weights([28, 50])
w_h2 = init_weights([50, 50])
w_o = init_weights([50, 2])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)


    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, Y: teY,
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))


# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    for i in range(1000):
        batch_xs, batch_ys = credit.train.next_batch(100)
        #batch_ys = np.reshape(batch_ys, [-1, 2])
        if i%100 == 0:
            sess.run(accuracy, feed_dict={x: credit.train.input, y_: credit.train.labels})
            print("step %d, whole training accuracy %g"%(i, trainacc))

        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
        print sess.run(accuracy, feed_dict={x: credit.train.input, y_: credit.train.labels})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print sess.run(accuracy, feed_dict={x: credit.test.input, y_: credit.test.labels})
    sess.close()
