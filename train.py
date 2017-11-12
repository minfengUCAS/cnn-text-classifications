#-*-encoding:utf-8-*-

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers

from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# Data Loading Parameters
tf.flags.DEFINE_float("dev_samples_percentage",0.1,"Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default:128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, val in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), val))
print("")

# Data Preparation
input_x, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in input_x])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(input_x)))

# shuffle data
np.random.seed(0)
index_perm = np.random.permutation(np.arange(len(y)))
x_shuffle = x[index_perm]
y_shuffle = y[index_perm]

dev_num = int(FLAGS.dev_samples_percentage*len(y))
train_x = x_shuffle[:-1*dev_num]
train_y = y_shuffle[:-1*dev_num]
eval_x = x_shuffle[-1*dev_num:]
eval_y = y_shuffle[-1*dev_num:]

print("Vocabulary size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Eval split: {:d}/{:d}".format(len(train_y), len(eval_y)))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(sequence_length = max_document_length,
                num_classes = train_y.shape[1],
                vocab_size = len(vocab_processor.vocabulary_),
                embedding_size = FLAGS.embedding_dim,
                filter_sizes = map(int,FLAGS.filter_sizes.split(",")),
                num_filters = FLAGS.num_filters,
                l2_reg_lambda = FLAGS.l2_reg_lambda)
        
        # Define Training procedure
        global_step = tf.Variable(tf.constant(0), trainable=False, name="global_step")
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vals = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vals, global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summary = []
        for grad, val in grads_and_vals:
            if grad is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(val.name), grad)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(val.name), tf.nn.zero_fraction(grad))
                grad_summary.append(grad_hist_summary)
                grad_summary.append(sparsity_summary)
        grad_summary_merged = tf.summary.merge(grad_summary)
        
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Write to {}\n".format(out_dir))
        
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        accuracy_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        
        # Train Summaries
        train_summary_op = tf.summary.merge([grad_summary, loss_summary, accuracy_summary])
        train_summary_dir = os.path.abspath(os.path.join(out_dir, "summary", "train"))
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Eval Summaries
        eval_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        eval_summary_dir = os.path.abspath(os.path.join(out_dir, "summary", "eval"))
        eval_summary_writer = tf.summary.FileWriter(eval_summary_dir, sess.graph)
        
        # Checkpoint directionary
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoint"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialization all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            feed_dict = {cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
            _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, 
                cnn.loss, cnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}\tStep:{}\t Loss:{:g}\tAcc:{:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def eval_step(x_batch, y_batch, writer=None):
            feed_dict = {cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0}

            step, summaries, loss, accuracy = sess.run([global_step, eval_summary_op, cnn.loss, cnn.accuracy], 
                    feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}\tStep:{}\tLoss:{:g}\tAcc:{:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        batches = data_helpers.batch_iter(list(zip(train_x,train_y)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop, For each batch
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                eval_step(eval_x, eval_y, writer = eval_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                print("\nSave Checkpoint:")
                path = saver.save(sess, checkpoint_prefix, global_step = current_step)
                print("Saved model checkpoint to {}\n".format(path))

