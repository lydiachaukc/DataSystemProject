#!/bin/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs_local/cuda-10.2/lib64
export PATH=$PATH:/pkgs_local/cuda-10.2/bin
#source ../ditto/ditto_venv/bin/activate
python3 trainning_all.py --task=dirty_amazon_itunes --running_NumBertMatcher_crossencoder=True --data_was_preprocessed=False --running_BertMatcher=False
python3 trainning_all.py --task=abt_buy --running_NumBertMatcher_crossencoder=True --data_was_preprocessed=False --running_BertMatcher=False
python3 trainning_all.py --task=dirty_walmart_amazon --running_NumBertMatcher_crossencoder=True --data_was_preprocessed=False --running_BertMatcher=False
#python3 trainning_all.py --task=dirty_dblp_acm --running_NumBertMatcher_crossencoder=True --data_was_preprocessed=False --running_BertMatcher=False
#python3 trainning_all.py --task=dirty_dblp_scholar --running_NumBertMatcher_crossencoder=True --data_was_preprocessed=False --running_BertMatcher=False
python3 trainning_all.py --task=amazon_google --running_NumBertMatcher_crossencoder=True --data_was_preprocessed=False --running_BertMatcher=False