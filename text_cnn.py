#-*- encoding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework import is_tensor

class TextCNN(object):
    """
    A CNN network for sentence classification.
    Use an embedding layer, followed by a convolutional, max-pooling and softmax layers
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self._sequence_length = sequence_length
        self._num_classes = num_classes
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._filter_sizes = filter_sizes
        self._num_filters = num_filters
        self._input_x = tf.placeholder(tf.int32, [None, self._sequence_length], name="input_x")
        self._input_y = tf.placeholder(tf.float32, [None, self._num_classes], name="input_y")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self._l2_reg_lambda = l2_reg_lambda
        self.network()

    @property
    def input_x(self):
        return self._input_x
    
    @input_x.setter
    def input_x(self, x):
        if is_tensor(x):
            self._input_x = x
        else:
            tf.logging.error("Input_x is not a tensor")

    @property
    def input_y(self):
        return self._input_y
    
    @input_y.setter
    def input_y(self, y):
        if is_tensor(y):
            self._input_y = y
        else:
            tf.logging.error("Input_y is not a tensor")

    @property
    def dropout_keep_prob(self):
        return self._dropout_keep_prob

    @dropout_keep_prob.setter
    def dropout_keep_prob(self, prob):
        if is_tensor(prob):
            self._dropout_keep_prob = prob
        else:
            tf.logging.error("Dropout_keep_prob is not a tensor")

    @property
    def loss(self):
        return self._loss
    
    @property
    def accuracy(self):
        return self._accuracy

    def network(self):
        # embedding layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            self._word_embedding = tf.Variable(tf.random_uniform([self._vocab_size, self._embedding_size],-1.0,1.0), name="word_embedding")
            self._embedding_char = tf.nn.embedding_lookup(self._word_embedding, self._input_x)
            self._embedding_char_flat = tf.expand_dims(self._embedding_char, -1)
            
        # convolutional, max-pooling layers
        pooled_outputs = []
        for i,filter_size in enumerate(self._filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # convolutional layer
                filter_shape = [filter_size, self._embedding_size, 1, self._num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name = "W")
                b = tf.Variable(tf.constant(0.1, shape = [self._num_filters]), name = "b")
                conv = tf.nn.conv2d(self._embedding_char_flat, W, strides = [1,1,1,1], padding = "VALID", name="conv")
                # apply no-liner function
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # maxpooling
                pooled = tf.nn.max_pool(h, ksize=[1, self._sequence_length-filter_size+1,1,1],
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="pooling")
                pooled_outputs.append(pooled)

        # Combine all pooling output features
        num_filters_total = self._num_filters * len(self._filter_sizes)
        self._h_pooled = tf.concat(pooled_outputs, 3)
        self._h_pooled_flat = tf.reshape(self._h_pooled, [-1, num_filters_total])

        # add dropout
        with tf.name_scope("dropout"):
            self._h_drop = tf.nn.dropout(self._h_pooled_flat, self._dropout_keep_prob)
        
        # add output layer
        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total,self._num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self._num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self._scores = tf.nn.xw_plus_b(self._h_drop, W, b, name="score")
            self._predictions = tf.argmax(self._scores, 1, name="predictions")

        # add loss layer
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self._scores, labels=self._input_y)
            self._loss = tf.reduce_mean(losses) + self._l2_reg_lambda * l2_loss

        # calcu accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self._predictions, tf.argmax(self._input_y, 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"), name="accuracy")
