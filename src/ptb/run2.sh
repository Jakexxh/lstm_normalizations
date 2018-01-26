#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

#python ptb_word_lm_base.py --lr=1.0 --g=5.0 --rnn_mode=cn_sep --num_gpus=1 --save_path=$HOME/log/ptb
#python ptb_word_lm_base.py --lr=1.0 --g=5.0 --rnn_mode=pcc_sep --num_gpus=1 --save_path=$HOME/log/ptb
#python ptb_word_lm_base.py --lr=1.0 --g=1.0 --rnn_mode=wn_sep --num_gpus=1 --save_path=$HOME/log/ptb
python ptb_word_lm_base.py --lr=1.0 --g=0.0 --rnn_mode=base --num_gpus=1 --save_path=$HOME/log/ptb_cob
python ptb_word_lm_base.py --lr=1.0 --g=1.0 --rnn_mode=bn_sep --num_gpus=1 --save_path=$HOME/log/ptb_cob
python ptb_word_lm_base.py --lr=1.0 --g=1.0 --rnn_mode=ln_sep --num_gpus=1 --save_path=$HOME/log/ptb_cob

#python ptb_word_lm_base.py --lr=1.0 --g=0.1 --rnn_mode=scale_cn --num_gpus=1 --save_path=/$HOME/log/ptb
#python ptb_word_lm_base.py --lr=1.0 --g=1.0--rnn_mode=scale_cn --num_gpus=1 --save_path=/$HOME/log/ptb
#python ptb_word_lm_base.py ---lr=1.0 --g=10.0 --rnn_mode=scale_cn --num_gpus=1 --save_path=/$HOME/log/ptb

#python ptb_word_lm_base.py --lr=1.0 --g=0.1 --rnn_mode=cn_sep --num_gpus=1 --save_path=/$HOME/log/ptb_bias
#python ptb_word_lm_base.py --lr=1.0 --g=1.0 --rnn_mode=cn_sep --num_gpus=1 --save_path=/$HOME/log/ptb_bias
#python ptb_word_lm_base.py ---lr=1.0 --g=0.01 --rnn_mode=cn_sep --num_gpus=1 --save_path=/$HOME/log/ptb_bias

#python ptb_word_lm_base.py --lr=1.0 --g=0.1 --rnn_mode=pcc_sep --num_gpus=1 --save_path=/$HOME/log/ptb_bias
#python ptb_word_lm_base.py --lr=1.0 --g=1.0 --rnn_mode=pcc_sep --num_gpus=1 --save_path=/$HOME/log/ptb_bias
#python ptb_word_lm_base.py --lr=1.0 --g=0.01 --rnn_mode=pcc_sep --num_gpus=1 --save_path=/$HOME/log/ptb_bias

#python ptb_word_lm_base.py --lr=1.0 --g=0.01 --rnn_mode=bn_sep --num_gpus=1 --save_path=/$HOME/log/ptb_bn
#python ptb_word_lm_base.py --lr=1.0 --g=0.1 --rnn_mode=bn_sep --num_gpus=1 --save_path=/$HOME/log/ptb_bn
#python ptb_word_lm_base.py --lr=1.0 --g=1.0 --rnn_mode=bn_sep --num_gpus=1 --save_path=/$HOME/log/ptb_bn

#python ptb_word_lm_base.py --lr=1.0 --g=0.01 --rnn_mode=wn_sep --num_gpus=1 --save_path=/$HOME/log/ptb_bias
#python ptb_word_lm_base.py --lr=1.0 --g=0.1 --rnn_mode=wn_sep --num_gpus=1 --save_path=/$HOME/log/ptb_bias
#python ptb_word_lm_base.py --lr=1.0 --g=1.0 --rnn_mode=wn_sep --num_gpus=1 --save_path=/$HOME/log/ptb_bias
