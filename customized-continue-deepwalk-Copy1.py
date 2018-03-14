
# coding: utf-8

# In[41]:

import os
import sys
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

from collections import defaultdict as dd
import tensorflow as tf
from tensorflow.python.ops.init_ops import glorot_uniform_initializer, RandomUniform, RandomNormal

from keras import layers
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

from gensim.models import Word2Vec
from gensim.models.word2vec import Vocab
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

np.random.seed(1)
random.seed(1)


# In[24]:


# GRADED FUNCTION: softmax

def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """

    ### START CODE HERE ### (≈ 3 lines of code)
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)

    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum

    ### END CODE HERE ###

    return s


# In[25]:

# Initialize parameters
DATASET = 'blogcatalog'

embedding_size = 128
learning_rate = 0.1
gl_learning_rate = 0.1
batch_size = 200
neg_samp = 0
model_file = 'trans.model'

window_size = 10
path_size = 10

g_batch_size = 200
g_learning_rate = 0.1
g_sample_size = 100

use_feature = True
update_emb = True
layer_loss =  True


# In[26]:

# Load the dataset
#NAMES = ['x', 'y', 'tx', 'ty', 'graph']
#OBJECTS = []
#for i in range(len(NAMES)):
#    f = "/hdd2/graph_embedding/dataset/blogcatalog/trans.{}.{}".format(DATASET, NAMES[i])
#    print(f)
#    OBJECTS.append(cPickle.load(open(f)))
#x, y, tx, ty, graph = tuple(OBJECTS)



def comp_iter(iter):
    """an auxiliary function used for computing the number of iterations given the argument iter.
    iter can either be an int or a float.
    """
    if iter >= 1:
        return iter
    return 1 if random.random() < iter else 0


def count_textfiles(file_list):
    count = -1
    num_path = 0
    for file_name in file_list:
        with open(file_name, 'r') as f:
            for line in f:
                num_path += 1
                for l in line.split():
                    count = max(count, int(l))
    return count, num_path

def gen_graph_from_path_collection(file_list):
    num_line = 0
    list_path = []
    g, gy = [], []
    for file_name in file_list:
        with open(file_name, 'r') as f:
            for line in f:
                path = [int(t) for t in line.split()]
                list_path.append(path)
                for l in range(len(path)):
                    for m in range(l - window_size, l + window_size + 1):
                        if m < 0 or m >= len(path): continue
                        g.append([path[l], path[m]])
                        gy.append(1.0)
                if (num_line % 1000 == 0):
                    tmp_g, tmp_gy, tmp_path = g, gy, list_path
                    list_path = []
                    g, gy = [], []
                    yield (np.array(tmp_g, dtype = np.int32),
                           np.array(tmp_path, dtype = np.float32))
                else:
                    pass
                num_line += 1
            yield (np.array(tmp_g, dtype = np.int32),
                           np.array(tmp_path, dtype = np.float32))


file_list = ['/hdd2/graph_embedding/customized/blogcatalog.embeddings.walks.0']
deepwalk_generator = gen_graph_from_path_collection(file_list)


# In[42]:

class Skipgram(Word2Vec):
    """A subclass to allow more customization of the Word2Vec internals."""

    def __init__(self, vocabulary_counts=None, **kwargs):

        self.vocabulary_counts = None

        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["workers"] = kwargs.get("workers", cpu_count())
        kwargs["size"] = kwargs.get("size", 128)
        kwargs["sentences"] = kwargs.get("sentences", None)
        kwargs["window"] = kwargs.get("window", 10)
        kwargs["sg"] = 1
        kwargs["hs"] = 0
        kwargs['negative'] = 5
        kwargs['iter'] = 0
        kwargs['compute_loss'] = True
        kwargs['sample'] = 0
        kwargs['min_alpha'] = 0.025

        if vocabulary_counts != None:
            self.vocabulary_counts = vocabulary_counts

        super(Skipgram, self).__init__(**kwargs)

class WalksCorpus(object):
    def __init__(self, file_list):
        self.file_list = file_list
    def __iter__(self):
        for file in self.file_list:
            with open(file, 'r') as f:
                for line in f:
                    yield line.split()


# In[34]:

v, num_path = count_textfiles(file_list)
vocabulary_size = v + 1


# In[188]:

window_size = 10
representation_size = 128
deepwalk_generator = gen_graph_from_path_collection(file_list)
embedding_filename = '/hdd2/graph_embedding/customized/blog_embeddings.txt'






# In[ ]:




# In[ ]:




# In[2]:

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels


# In[3]:

def scoring(emb_filename, matfile):
    # 0. Files
    embeddings_file = emb_filename

    # 1. Load Embeddings
    embeddings = np.loadtxt(embeddings_file)

    # 2. Load labels
    mat = sio.loadmat(matfile)
    A = mat['network']
    labels_matrix = mat['group']
    labels_count = labels_matrix.shape[1]
    mlb = MultiLabelBinarizer(range(labels_count))

    # Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
    features_matrix = embeddings

    # 2. Shuffle, to create train/test groups
    shuffles = []
    for x in range(1):
        shuffles.append(skshuffle(features_matrix, labels_matrix))

    # 3. to score each train/test group
    all_results = defaultdict(list)

#     if args.all:
#         training_percents = numpy.asarray(range(1, 10)) * .1
#     else:
#         training_percents = [0.1, 0.5, 0.9]
    training_percents = [0.1]
    for train_percent in training_percents:
        for shuf in shuffles:

            X, y = shuf

            training_size = int(train_percent * X.shape[0])

            X_train = X[:training_size, :]
            y_train_ = y[:training_size]

            y_train = [[] for x in range(y_train_.shape[0])]


            cy =  y_train_.tocoo()
            for i, j in zip(cy.row, cy.col):
                y_train[i].append(j)

            assert sum(len(l) for l in y_train) == y_train_.nnz

            X_test = X[training_size:, :]
            y_test_ = y[training_size:]

            y_test = [[] for _ in range(y_test_.shape[0])]

            cy =  y_test_.tocoo()
            for i, j in zip(cy.row, cy.col):
                y_test[i].append(j)

            clf = TopKRanker(LogisticRegression())
            clf.fit(X_train, y_train_)

            # find out how many labels should be predicted
            top_k_list = [len(l) for l in y_test]
            preds = clf.predict(X_test, top_k_list)

            results = {}
            averages = ["micro", "macro"]
            for average in averages:
                results[average] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)

            all_results[train_percent].append(results)

    print ('Results, using embeddings of dimensionality', X.shape[1])
    print ('-------------------')
    for train_percent in sorted(all_results.keys()):
        print ('Train percent:', train_percent)
    for index, result in enumerate(all_results[train_percent]):
        print ('Shuffle #%d:   ' % (index + 1), result)
    avg_score = defaultdict(float)
    for score_dict in all_results[train_percent]:
        for metric, score in iteritems(score_dict):
            avg_score[metric] += score
    for metric in avg_score:
        avg_score[metric] /= len(all_results[train_percent])
    print ('Average score:', dict(avg_score))
    print ('-------------------')





if __name__ == '__main__':
    walks_corpus = WalksCorpus(file_list)
    print('training...')
    model = Skipgram(sentences=walks_corpus, vocabulary_counts=vocabulary_size, size=128,
            window=10, min_count=0, trim_rule=None, workers=8)
    model.wv.save_word2vec_format('/hdd2/graph_embedding/customized/model_ns5_iter1.output')


#matfile = '/hdd2/graph_embedding/deepwalk/example_graphs/blogcatalog.mat'
#embedding_filename = '/hdd2/graph_embedding/customized/blog_embeddings.5650.txt'
#scoring(embedding_filename, matfile)




