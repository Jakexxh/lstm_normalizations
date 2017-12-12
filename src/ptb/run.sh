#!/usr/bin/env bash
#python mnist/test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/mnist_iden_tr/cn_sep_1_0.001  --lr=0.001
#python mnist/test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/mnist_iden_tr/cn_sep_1_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/mnist_iden_tr/cn_sep_1_0.1 --lr=0.1
#python mnist/test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/mnist_iden_tr/cn_sep_1_1.0 --lr=1.0
#
#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_iden_tr/wn_sep_0.1_0.001  --lr=0.001
#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_iden_tr/wn_sep_0.1_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_iden_tr/wn_sep_0.1_0.1 --lr=0.1
#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_iden_tr/wn_sep_0.1_1.0 --lr=1.0
#
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_refactor/ln_sep_0.1_1.0 --lr=1.0
#
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_0.001  --lr=0.001
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_0.1 --lr=0.1
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_1.0 --lr=1.0

#python mnist/test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/mnist_refactor/cn_sep_1.0_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/mnist_refactor/cn_sep_1.0_0.1 --lr=0.1
#python mnist/test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/mnist_refactor/cn_sep_1.0_1.0 --lr=1.0

#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_refactor/ln_sep_0.1_0.001  --lr=0.001
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_refactor/ln_sep_0.1_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_refactor/ln_sep_0.1_0.1 --lr=0.1

#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_refactor/wn_sep_1.0_0.001  --lr=0.001
#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_refactor/wn_sep_1.0_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_refactor/wn_sep_1.0_0.1 --lr=0.1
#
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_refactor/ln_sep_0.1_0.001  --lr=0.001
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_refactor/ln_sep_0.1_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_refactor/ln_sep_0.1_0.1 --lr=0.1

#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_iden_tr/wn_sep_0.1_0.001  --lr=0.001
#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_last/wn_sep_0.1_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_iden_tr/wn_sep_0.1_0.1 --lr=0.1
#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_iden_tr/wn_sep_0.1_1.0 --lr=1.0

#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_last/ln_sep_1.0_0.01  --lr=0.01
# python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_iden_tr/ln_sep_1.0_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_iden_tr/ln_sep_0.1_0.1 --lr=0.1
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_iden_tr/ln_sep_0.1_1.0 --lr=1.0

#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_last/bn_sep_0.1_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_0.1 --lr=0.1
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_1.0 --lr=1.0

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
