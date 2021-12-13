#!/bin/sh
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs_local/cuda-10.2/lib64
#export PATH=$PATH:/pkgs_local/cuda-10.2/bin
#source ../ditto/ditto_venv/bin/activate
python3 trainning_all.py --task dirty_amazon_itunes
python3 trainning_all.py --task abt_buy
python3 trainning_all.py --task dirty_walmart_amazon
python3 trainning_all.py --task dirty_dblp_acm
python3 trainning_all.py --task dirty_dblp_scholar
python3 trainning_all.py --task amazon_google