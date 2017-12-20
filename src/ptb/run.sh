#!/usr/bin/env bash
#python ptb_word_lm_base.py --lr=0.1 --rnn_mode=bn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig2/bn_sep_g0.01_0.1
#python ptb_word_lm_base.py --lr=1.0 --rnn_mode=bn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig2/bn_sep_g0.01_1.0
#python ptb_word_lm_base.py --lr=10.0 --rnn_mode=bn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig2/bn_sep_g0.01_10.0
##
#python ptb_word_lm_base.py --lr=0.1 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig2/cn_sep_g0.1_0.1
#python ptb_word_lm_base.py --lr=1.0 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig2/cn_sep_g0.1_1.0
#python ptb_word_lm_base.py --lr=10.0 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig2/cn_sep_g0.1_10.0

#python ptb_word_lm_base.py --lr=0.1 --rnn_mode=wn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig/wn_sep_g1.0_0.1
#python ptb_word_lm_base.py --lr=10.0 --rnn_mode=wn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig/wn_sep_g1.0_10.0

#python ptb_word_lm_base.py --lr=0.1 --rnn_mode=ln_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig2/ln_sep_g0.01_0.1
#python ptb_word_lm_base.py --lr=1.0 --rnn_mode=ln_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig2/ln_sep_g0.01_1.0
#python ptb_word_lm_base.py --lr=10.0 --rnn_mode=ln_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig2/ln_sep_g0.01_10.0
#
#python ptb_word_lm_base.py --lr=0.1 --rnn_mode=pcc_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig2/pcc_sep_g0.1_0.1
#python ptb_word_lm_base.py --lr=1.0 --rnn_mode=pcc_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig2/pcc_sep_g0.1_1.0
#python ptb_word_lm_base.py --lr=10.0 --rnn_mode=pcc_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig2/pcc_sep_g0.1_10.0


#python ptb_word_lm_base.py --lr=1.0 --g=0.0 --rnn_mode=base --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal

python ptb_word_lm_base.py --lr=1.0 --g=0.001 --rnn_mode=bn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal
python ptb_word_lm_base.py --lr=1.0 --g=0.0001 --rnn_mode=bn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal
#python ptb_word_lm_base.py --lr=1.0 --g=1.0 --rnn_mode=bn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal

#python ptb_word_lm_base.py --lr=1.0 --g=0.01 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal
#python ptb_word_lm_base.py --lr=1.0 --g=0.1 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal
#python ptb_word_lm_base.py --lr=1.0 --g=1.0 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal
#
#python ptb_word_lm_base.py --lr=1.0 --g=0.01 --rnn_mode=wn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal
#python ptb_word_lm_base.py --lr=1.0 --g=0.1 --rnn_mode=wn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal
#python ptb_word_lm_base.py --lr=1.0 --g=1.0 --rnn_mode=wn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal

python ptb_word_lm_base.py --lr=1.0 --g=0.001 --rnn_mode=ln_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal
python ptb_word_lm_base.py --lr=1.0 --g=0.0001 --rnn_mode=ln_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal
#python ptb_word_lm_base.py --lr=1.0 --g=1.0 --rnn_mode=ln_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal

#python ptb_word_lm_base.py --lr=1.0 --g=0.01 --rnn_mode=pcc_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal
#python ptb_word_lm_base.py --lr=1.0 --g=0.1 --rnn_mode=pcc_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal
#python ptb_word_lm_base.py --lr=1.0 --g=1.0 --rnn_mode=pcc_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig_orthogonal

#python ptb_word_lm_base.py --lr=10.0 --g=0.0001 --rnn_mode=bn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4
#python ptb_word_lm_base.py --lr=0.1 --g=0.0001 --rnn_mode=bn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4
#
#python ptb_word_lm_base.py --lr=1.0 --g=0.001 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4
#python ptb_word_lm_base.py --lr=10.0 --g=0.001 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4
#python ptb_word_lm_base.py --lr=0.1 --g=0.001 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4

#python ptb_word_lm_base.py --lr=1.0 --g=10.0 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4
#
#python ptb_word_lm_base.py --lr=1.0 --g=0.001 --rnn_mode=pcc_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4
#python ptb_word_lm_base.py --lr=10.0 --g=0.001 --rnn_mode=pcc_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4
#python ptb_word_lm_base.py --lr=0.1 --g=0.001 --rnn_mode=pcc_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4


#python ptb_word_lm_base.py --lr=0.01 --g=0.01 --rnn_mode=ln_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4

#python ptb_word_lm_base.py --lr=1.0 --g=0.1 --rnn_mode=ln_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4
#
#python ptb_word_lm_base.py --lr=1.0 --g=1.0 --rnn_mode=ln_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4
#
#
#python ptb_word_lm_base.py --lr=1.0 --g=0.001 --rnn_mode=wn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4
#python ptb_word_lm_base.py --lr=10.0 --g=0.001 --rnn_mode=wn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4
#python ptb_word_lm_base.py --lr=0.1 --g=0.001 --rnn_mode=wn_sep --num_gpus=1 --save_path=/tmp/log/ptb_ref_orig4


