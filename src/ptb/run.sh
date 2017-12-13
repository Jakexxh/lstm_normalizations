#!/usr/bin/env bash

python ptb_word_lm.py --lr=1.0 --rnn_mode=basic --num_gpus=1 --save_path=/tmp/log/ptb_ref_base/basic_1.0

python ptb_word_lm.py --lr=1.0 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_base/cn_sep_g1.0_1.0

python ptb_word_lm.py --lr=1.0 --rnn_mode=wn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_base/wn_sep_g1.0_1.0

python ptb_word_lm.py --lr=1.0 --rnn_mode=ln_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_base/ln_sep_g0.1_1.0

python ptb_word_lm.py --lr=1.0 --rnn_mode=bn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_base/bn_sep_g0.1_1.0

python ptb_word_lm.py --lr=1.0 --rnn_mode=pcc_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_base/pcc_sep_g1.0_1.0


python ptb_word_lm_base.py --lr=1.0 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig/cn_sep_g1.0_1.0

python ptb_word_lm_base.py --lr=1.0 --rnn_mode=wn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig/wn_sep_g1.0_1.0

python ptb_word_lm_base.py --lr=1.0 --rnn_mode=ln_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig/ln_sep_g0.1_1.0

python ptb_word_lm_base.py --lr=1.0 --rnn_mode=pcc_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig/pcc_sep_g1.0_1.0
