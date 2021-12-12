#!/bin/sh
srun --partition=smallgpunodes --mem=12G --gres=gpu:1 --pty bash -i
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs_local/cuda-10.2/lib64
export PATH=$PATH:/pkgs_local/cuda-10.2/bin

python3 trainning_all.py --task amazon_google
python3 trainning_all.py --task dirty_amazon_itunes
python3 trainning_all.py --task dirty_dblp_acm
python3 trainning_all.py --task dirty_dblp_scholar
python3 trainning_all.py --task dirty_walmart_amazon