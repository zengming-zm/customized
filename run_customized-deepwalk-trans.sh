#!/bin/bash

python customized-deepwalk-trans.py --gpu_id=0 --batch_path_size=2 --num_pair_per_path=400 --embedding_size=128 --skip_window=10 --num_skips=20 --num_class=39 --num_neg_samples=64 --lambda_supervised=0.1 --supervised_learning_rate=0.25 --semisupervised_learning_rate=0.1 --use_feature=False --emb_steps=10000 --clf_steps=1000 --embedding_path="/hdd2/graph_embedding/customized/results/exp_blogcatalog_semi_avg_label10_3/" --rand_walk_paths="/hdd2/graph_embedding/customized/blogcatalog.embeddings.walks.0" --source_file="/hdd2/graph_embedding/deepwalk/example_graphs/blogcatalog.mat"
