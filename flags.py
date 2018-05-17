from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("gpu_id", "0", "which gpu to use.")
flags.DEFINE_integer("batch_path_size", 2, "how many paths for each batch")
flags.DEFINE_integer("num_pair_per_path", 400, "how many pair in each path")
flags.DEFINE_integer("embedding_size", 128, "size of embedding vector")
flags.DEFINE_integer("skip_window", 10, "size of skip window")
flags.DEFINE_integer("num_skips", 20, "How many times to reuse an input to generate a label - useless")
flags.DEFINE_integer("num_class", 39, "how many classes in the dataset")
flags.DEFINE_integer("num_neg_samples", 64, "how many negative samples")
flags.DEFINE_float("lambda_supervised", 0.1, "trade-off for supervised regularization")
flags.DEFINE_float("supervised_learning_rate", 0.25, "learning rate for supervised classification")
flags.DEFINE_float("semisupervised_learning_rate", 0.1, "learning rate for semi-supervised learning")
flags.DEFINE_bool("use_feature", False, "whether use external features")
flags.DEFINE_integer("emb_steps", 10000, "how many epochs for learning embedding")
flags.DEFINE_integer("clf_steps", 100, "how many epochs for learning classifier")
flags.DEFINE_string("embedding_path", "/hdd2/graph_embedding/customized/results/exp_blogcatalog_semi_avg_label10_3/", "location for the embedding")
