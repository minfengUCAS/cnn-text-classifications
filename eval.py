#-*- encoding:utf-8 -*-

import tensorflow as tf
import numpy as np
import datetime
import time
import os
import data_helpers
from tensorflow.contrib import learn
from text_cnn import TextCNN
import csv

# Parameter
# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Print Parameters
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for key,val in sorted(FLAGS.__flags.items()):
    print("{}:{}".format(key,val))
print("")

if FLAGS.eval_train:
    x_raw, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y = np.argmax(y, axis=1)
else:
    x_raw = ["wonderful", "terrible"]
    y = [1,0]

# Map data into vocabulary
vocab_path = os.path.abspath(os.path.join(FLAGS.checkpoint_dir,"..", "vocab"))
vocab_preprocessor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_text = np.array(list(vocab_preprocessor.transform(x_raw)))

# Evaluation
check_point_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    sess_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=sess_config)
    with sess.as_default():
        # Load the saved mete graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(check_point_file))
        saver.restore(sess, check_point_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Get prediction tensor
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_text), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_text_batch in batches:
            batch_predictions = sess.run([predictions],{input_x: x_text_batch,
                dropout_keep_prob: 1.0})
            all_predictions= np.concatenate([all_predictions, batch_predictions[0]])

# Print Accuracy if y is not None
if y is not None:
    correct_prediction = float(sum(all_predictions == y))
    print("Total number of text examples: {}".format(len(y)))
    print("Accuracy: {}".format(correct_prediction/float(len(y))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
output_dir = os.path.join(FLAGS.checkpoint_dir,"..", "prediciton.csv")
print("Saving evaluatio to {}".format(output_dir))
with open(output_dir, "w") as write_f:
    csv.writer(write_f).writerows(predictions_human_readable)

