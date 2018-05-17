import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile
import pickle

import numpy as np
import cPickle
import random
import math
import h5py
from multiprocessing import cpu_count
import scipy.io as sio
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from six import iteritems
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as skshuffle
from numpy import genfromtxt
from sklearn.metrics import accuracy_score, f1_score

from collections import defaultdict as dd
import tensorflow as tf
from tensorflow.python.ops.init_ops import glorot_uniform_initializer, RandomUniform, RandomNormal

from gensim.models import Word2Vec
from gensim.models.word2vec import Vocab

import flags
#import word2vec

flags = tf.flags
FLAGS = flags.FLAGS


np.random.seed(1)
random.seed(1)

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id 
num_pair_per_path = FLAGS.num_pair_per_path

def set_hparams_from_args(args):
    if not args:
	return


def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
	G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in iteritems(G)}

# # Load Data
matfile = '/hdd2/graph_embedding/deepwalk/example_graphs/blogcatalog.mat'
mat = sio.loadmat(matfile)
A = mat['network']
g = sparse2graph(A)
labels_matrix = mat['group']
labels_count = labels_matrix.shape[1]


# file_list = ['/hdd2/graph_embedding/customized/tmp/citeseer.embeddings.walks']
file_list = ['/hdd2/graph_embedding/customized/blogcatalog.embeddings.walks.0']
dataset = genfromtxt(file_list[0], delimiter=' ')

def get_num_vacabulary(dataset):
    word_count = 0
    for d in dataset:
        word_count = max(word_count, max(d))
    return int(word_count)

vocabulary_size = get_num_vacabulary(dataset) + 1


words = dataset.flatten()
words = [str(int(w)) for w in words]



def build_dataset(words):
    count = []
#     count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = int(word) #len(dictionary)
    data = list()
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        data.append(index)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)



## Read the data, the trn_x is the position in the embedding matrix
data_splited_filename = '/hdd2/graph_embedding/customized/blogcatalog_splited_p10.pickle'
with open(data_splited_filename, 'rb') as handle:
    unserialized_data = pickle.load(handle)
trn_idx, trn_y, tst_idx, tst_y = (unserialized_data['trn_idx'],
				  unserialized_data['trn_y'],
				  unserialized_data['tst_idx'],
				  unserialized_data['tst_y'])
#trn_idx = trn_idx.reshape(trn_idx.shape[0], 1)
#tst_idx = tst_idx.reshape(tst_idx.shape[0], 1)
#trn_idx, trn_y, tst_idx, tst_y, = get_labeled_instant()


# In[399]:

path_index = 0
batch_path_size = FLAGS.batch_path_size

batch_size = batch_path_size * num_pair_per_path
def generate_batch(batch_path_size, num_skips, skip_window):
    global path_index
    batch_size = batch_path_size * num_pair_per_path
    batch = []
    labels = []
    path_list = []
    path_index_list = []
    w_p2p = np.zeros([batch_size, batch_path_size])
#     span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=batch_path_size)
    for i in range(batch_path_size):
        len_path = len(dataset[path_index])
        path_list.append(dataset[path_index])
        for l in range(skip_window, len_path - skip_window): # [ skip_window target skip_window ]
            for m in range(l - skip_window, l + skip_window + 1):
                if m < 0 or m >= len_path or m == l:
                    continue
                batch.append(dictionary[str(int(dataset[path_index][l]))])
                labels.append(dictionary[str(int(dataset[path_index][m]))])
                path_index_list.append(i)

        w_p2p[i * num_pair_per_path : (i+1) * num_pair_per_path, i] = 1

        path_index = (path_index + 1) % len(dataset)
    return (np.asarray(batch, dtype = np.int32),
            np.asarray(labels, dtype = np.int32).reshape([len(labels), 1]),
            np.asarray(path_list, dtype=np.float32),
            np.asarray(path_index_list, dtype=np.float32),
            w_p2p)


# Reproduce Gensim weights initialization
def seeded_vector(seed_string, vector_size):
    """Create one 'random' vector (but deterministic by seed_string)"""
    # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
    once = np.random.RandomState(hash(seed_string) & 0xffffffff)
    return (once.rand(vector_size) - 0.5) / vector_size

# read trained embedding matrix
def read_train_matrix(embeddings_file):
    trained_embeddings = np.loadtxt(embeddings_file)
    return trained_embeddings

features_list = []
for idx in range(vocabulary_size):
    str_node = reverse_dictionary[idx]
    features_list.append(seeded_vector(str_node + str(1), FLAGS.embedding_size))
features_matrix = np.asarray(features_list)
#features_matrix = read_train_matrix('/hdd2/graph_embedding/customized/results/deepwalk_unsupervised/blog_embeddings_iter710000.txt')


def Average_Paths(X, _weight, _bias):
    path_avg_output = tf.reduce_mean(X, axis=1)

    scale_output = tf.nn.softmax(tf.matmul(_weight, tf.transpose(path_avg_output)) + _bias)

#     scale_output = tf.nn.softmax(scale_output)

#     print('softmax_output.shape:')
#     print(softmax_output.shape)
    # Linear activation
    return scale_output, path_avg_output[-1], path_avg_output


use_feature = False
use_reweight = True
labeled_size = trn_idx.shape[0]
batch_path_size = FLAGS.batch_path_size
batch_size = batch_path_size * num_pair_per_path 
embedding_size = FLAGS.embedding_size # Dimension of the embedding vector.
skip_window = FLAGS.skip_window # How many words to consider left and right.
num_skips = FLAGS.num_skips # How many times to reuse an input to generate a label.
num_class = FLAGS.num_class

# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.

config = tf.ConfigProto(allow_soft_placement=True)
graph = tf.Graph()

lambda1 = FLAGS.lambda_supervised

with graph.as_default(), tf.device('/gpu:0'):

    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    w_path2pair = tf.placeholder(tf.float32, shape = [batch_size, batch_path_size])
    path_dataset= tf.placeholder(tf.int32, shape = [batch_path_size, len(dataset[0])])
    path_id = tf.placeholder(tf.int32, shape = [None, ])

    # Variables.
    with tf.device('/cpu:0'):
	embeddings = tf.Variable(features_matrix, dtype=tf.float32, trainable = True)

    w = tf.Variable(tf.truncated_normal([embedding_size, num_class]))
    b = tf.Variable(tf.zeros([num_class]))

    # Supervised training, fixed embedding, update supervised w and b
    clf_idx = tf.placeholder(tf.int32, shape=[None])
    clf_y = tf.placeholder(tf.float32, shape=[None, trn_y.shape[1]])

    embed_x = tf.nn.embedding_lookup(embeddings, clf_idx)
    embed_x = tf.stop_gradient(embed_x)


   
    logit_y = tf.matmul(embed_x, w) + b
    predictions = tf.nn.sigmoid(logit_y)
    
    #tf.summary.histogram("predictions", predictions)

    clf_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits = logit_y, labels = clf_y))

    clf_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.supervised_learning_rate).minimize(clf_loss)

    #tf.summary.histogram("clf_loss", clf_loss)



    # Semi-supervised training, fix w, b in the sueprvised branch, 
    # update embedding and word2vec w and b (softmax_weights, softmax_biases)

    softmax_weights = tf.Variable(
	tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)

    weight_avg = tf.Variable(
        tf.truncated_normal([1, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    biase_avg = tf.Variable(tf.zeros([1]))
#    if (use_reweight):
#        rnn_inputs = tf.nn.embedding_lookup(embeddings, path_dataset)
#        reweight_each_path, cg_outputs, cg_last_output = Average_Paths(
#            rnn_inputs, weight_avg, biase_avg)
##         reweight_each_path = tf.reshape(reweight, [-1, 1])
#        reweight_each_pair = tf.matmul(w_path2pair, tf.transpose(reweight_each_path))
#    else:
##         reweight_each_path = tf.ones(shape=[batch_path_size, 1])
#        reweight_each_pair = tf.ones(shape=[batch_size, 1])

# #   for datasets in the deepwalk, multi-class
    embed_supervised_x = tf.nn.embedding_lookup(embeddings, clf_idx)
    w_supervised = tf.Variable(w, dtype=tf.float32, trainable = False)
    b_supervised = tf.Variable(b, dtype=tf.float32, trainable = False)
    logit_y_supervised = tf.matmul(embed_supervised_x, w_supervised) + b_supervised

    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
	    labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size)) + lambda1 * tf.reduce_mean(
		tf.nn.sigmoid_cross_entropy_with_logits(logits = logit_y_supervised, labels = clf_y))
    #tf.summary.histogram("loss", loss)

    global_step = tf.Variable(0, trainable=False)

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.semisupervised_learning_rate).minimize(loss, global_step=global_step)


#     clf_lr = 0.25
    # # for 1 class
#     feature_dataset = tf.placeholder(tf.float32, shape=[None, trn_f.shape[1]])
#     l_x_hid = tf.layers.dense(inputs = feature_dataset, units = clf_y.shape[1],
#                               activation = tf.nn.softmax, kernel_initializer = glorot_uniform_initializer())
#     if (use_feature):
#         logit_emd = tf.layers.dense(inputs = embed_x, units = clf_y.shape[1],
#                                     activation=tf.nn.softmax, kernel_initializer=glorot_uniform_initializer())
#         l_f = tf.concat([l_x_hid, logit_emd], axis = 1)
#         logit_y = tf.layers.dense(inputs = l_f, units = clf_y.shape[1],
#                                   activation=tf.nn.softmax, kernel_initializer=glorot_uniform_initializer())
#     else:
#     #   for datasets in the icml paper, single-class
#         logit_y = tf.layers.dense(inputs = embed_x, units = clf_y.shape[1],
#                                   activation=tf.nn.softmax, kernel_initializer=glorot_uniform_initializer())

#     clf_loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(logits = logit_y, labels = clf_y))



    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
#     norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
#     normalized_embeddings = embeddings / norm
#     valid_embeddings = tf.nn.embedding_lookup(
#         normalized_embeddings, valid_dataset)
#     similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

def running_test():
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()

        batch_data, batch_labels, batch_path, batch_path_id, w_p2p = generate_batch(
            batch_path_size, num_skips, skip_window)
        feed_dict = {train_dataset : batch_data,
                     train_labels : batch_labels,
                     path_dataset : batch_path,
                     path_id : batch_path_id,
                     w_path2pair : w_p2p}
        _, l, res_embed = session.run([optimizer, loss, embed], feed_dict=feed_dict)
        print('loss: %.4f' %(l))



use_feature = FLAGS.use_feature
emb_steps = FLAGS.emb_steps #10000 #50000001
clf_steps = FLAGS.clf_steps # 1000 
embedding_path = FLAGS.embedding_path
def running():
    total_step = 0
    with tf.Session(graph=graph, config=config) as session:
	#train_writer = tf.summary.FileWriter( './log/train ', session.graph)
        tf.global_variables_initializer().run()
        print('Initialized')

        while (True):
            average_emb_loss = 0
            average_clf_loss = 0
            for step in range(emb_steps):
		#merge = tf.summary.merge_all()

                batch_data, batch_labels, batch_path, batch_path_id, w_p2p = generate_batch(
                    batch_path_size, num_skips, skip_window)
                feed_dict = {train_dataset : batch_data,
                             train_labels : batch_labels,
                             path_dataset : batch_path,
			     clf_idx : trn_idx, 
			     clf_y : trn_y}
                             #path_id : batch_path_id,
                             #w_path2pair : w_p2p}
                _, l = session.run([optimizer, loss], feed_dict=feed_dict)
		#train_writer.add_summary(summary, total_step)
                average_emb_loss += l
                if step % 2000 == 0:
                    if step > 0:
                        average_emb_loss = average_emb_loss / 2000.0
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average embedding loss at step %d: %f' % (step, average_emb_loss))
                    average_emb_loss = 0

            for step in range(clf_steps):
                if (use_feature):
                    # for datasets in the icml paper
                    feed_dict = {clf_idx : trn_idx, clf_y : trn_y, feature_dataset : trn_f}
                else:
                    feed_dict = {clf_idx : trn_idx, clf_y : trn_y}

		#print(trn_idx[0])
		#print(trn_y[0])
		#sys.exit(0)

                _, l, res_w, res_b = session.run([clf_optimizer, clf_loss, w, b], feed_dict=feed_dict)
#		print('-----save parameters-----')
#		w_filename = '/hdd2/graph_embedding/customized/results/exp_blogcatalog_semi_avg_label10/w_iter%d.txt' %total_step
#		np.savetxt(w_filename, res_w)
#		b_filename = '/hdd2/graph_embedding/customized/results/exp_blogcatalog_semi_avg_label10/b_iter%d.txt' %total_step
#		np.savetxt(b_filename, res_b)


                average_clf_loss += l
                if step % 1000 == 0:
                    if step > 0:
                        average_clf_loss = average_clf_loss / 1000.0
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average classification loss at step %d: %f' % (step, average_clf_loss))
                    average_clf_loss = 0

            # note that this is expensive (~20% slowdown if computed every 500 steps)
    #         y_p = tf.argmax(logit_y, 1)
    #         feed_dict = {clf_x : tst_x, clf_y : tst_y}
    #         _, l, res_logit_y = session.run([clf_optimizer, clf_loss, logit_y], feed_dict=feed_dict)
    #         y_true = np.argmax(tst_y,1)
    #         print("micro: ", f1_score(y_true, res_pred_y, average='micro'))
    #         print("macro: ", f1_score(y_true, res_pred_y, average='macro'))

            res_y_pred = tf.round(predictions)
	    res_y_pred_val = predictions
            res_y_true = tf.round(clf_y)
	    res_feature = embed_x


            if (use_feature):
		trn_y_pred = res_y_pred.eval({clf_idx : trn_idx, clf_y: trn_y, feature_dataset : trn_f})
		trn_y_ture = res_y_true.eval({clf_idx : trn_idx, clf_y: trn_y, feature_dataset : trn_f})
		tst_y_pred = res_y_pred.eval({clf_idx : tst_idx, clf_y: tst_y, feature_dataset : tst_f})
		tst_y_ture = res_y_true.eval({clf_idx : tst_idx, clf_y: tst_y, feature_dataset : tst_f})
		#print("Epoch %d, trn acc %.6f acc %.6f:" % (total_step,
                #                                            f1_score(trn_y_ture, trn_y_pred.flatten()),
                #                                            f1_score(tst_y_ture, tst_y_pred.flatten())))
            else:
		pass
                trn_y_pred = res_y_pred.eval({clf_idx : trn_idx, clf_y: trn_y})
                trn_y_true = res_y_true.eval({clf_idx : trn_idx, clf_y: trn_y})
		trn_y_pred_val = res_y_pred_val.eval({clf_idx : trn_idx, clf_y: trn_y})
                tst_y_pred = res_y_pred.eval({clf_idx : tst_idx, clf_y: tst_y})
                tst_y_true = res_y_true.eval({clf_idx : tst_idx, clf_y: tst_y})
		tst_y_pred_val = res_y_pred_val.eval({clf_idx : tst_idx, clf_y: tst_y})

		tst_features_val = res_feature.eval({clf_idx : tst_idx, clf_y: tst_y})

		print("Epoch %d, trn: micro-f1 %.6f, macro: %.6f, tst: micro-f1 %.6f, macro: %.6f:" % (total_step,
							    f1_score(trn_y_true, trn_y_pred, average = 'micro'),
							    f1_score(trn_y_true, trn_y_pred, average = 'macro'),
							    f1_score(tst_y_true, tst_y_pred, average='micro' ),
							    f1_score(tst_y_true, tst_y_pred, average = 'macro')))
#		print('------trn_y_pred_val---')
#		print(trn_y_pred_val.tolist()[0])
#		print('------trn_y_pred-------')
#                print(trn_y_pred.tolist()[0])
#		print('------trn_y_true-------')
#                print(trn_y_true.tolist()[0])
#		print('------tst_y_pred_val---')
#		print(tst_y_pred_val.tolist()[0])
#		print('------tst_y_pred-------')
#		print(tst_y_pred.tolist()[0])
#		print('------tst_y_true-------')
#		print(tst_y_true.tolist()[0])

		#sys.exit(0)

            if total_step % 1 == 0:
                embedding_filename = embedding_path + 'blog_embeddings_iter%d.txt' %total_step
                not_normal_embeddings = embeddings.eval()
                ordered_embeddings = [not_normal_embeddings[dictionary[str(node)]] for node in range(len(dictionary))]
                np.savetxt(embedding_filename, ordered_embeddings)

            total_step += 1


        final_embeddings = normalized_embeddings.eval()
        not_normal_embeddings = embeddings.eval()


# In[ ]:

if __name__ == '__main__':
    running()




