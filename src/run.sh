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
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_iden_tr/ln_sep_0.1_0.001  --lr=0.001
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_iden_tr/ln_sep_0.1_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_iden_tr/ln_sep_0.1_0.1 --lr=0.1
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_iden_tr/ln_sep_0.1_1.0 --lr=1.0
#
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_0.001  --lr=0.001
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_0.1 --lr=0.1
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_1.0 --lr=1.0

#python mnist/test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/mnist_sig/cn_sep_1.0_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/mnist_sig/cn_sep_1.0_0.1 --lr=0.1
#python mnist/test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/mnist_sig/cn_sep_1.0_1.0 --lr=1.0

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

#python ptb/ptb_word_lm.py --lr=0.1 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb/cn_sep_1.0
#python ptb/ptb_word_lm.py --lr=1.0 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb/cn_sep_1.0
#python ptb/ptb_word_lm.py --lr=10.0 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb/cn_sep_10.0

python char-rnn-tensorflow/train.py --model=base --lr=0.001  --save_dir=/tmp/char/save/base_0.001 --log_dir=/tmp/char/log/base_0.001
python char-rnn-tensorflow/sample.py --save_dir=/tmp/char/save/base_0.001
python char-rnn-tensorflow/train.py --model=base --lr=0.01  --save_dir=/tmp/char/save/base_0.01 --log_dir=/tmp/char/log/base_0.01
python char-rnn-tensorflow/sample.py --save_dir=/tmp/char/save/base_0.01
python char-rnn-tensorflow/train.py --model=base --lr=0.1  --save_dir=/tmp/char/save/base_0.1 --log_dir=/tmp/char/log/base_0.1
python char-rnn-tensorflow/sample.py --save_dir=/tmp/char/save/base_0.1

python char-rnn-tensorflow/train.py --model=base --lr=0.001  --save_dir=/tmp/char/save/cn_sep_g1.0_0.001 --log_dir=/tmp/char/log/cn_sep_g1.0_0.001
python char-rnn-tensorflow/sample.py --save_dir=/tmp/char/save/cn_sep_g1.0_0.001
python char-rnn-tensorflow/train.py --model=base --lr=0.01  --save_dir=/tmp/char/save/cn_sep_g1.0_0.01 --log_dir=/tmp/char/log/cn_sep_g1.0_0.01
python char-rnn-tensorflow/sample.py --save_dir=/tmp/char/save/cn_sep_g1.0_0.01
python char-rnn-tensorflow/train.py --model=base --lr=0.1  --save_dir=/tmp/char/save/cn_sep_g1.0_0.1 --log_dir=/tmp/char/log/cn_sep_g1.0_0.1
python char-rnn-tensorflow/sample.py --save_dir=/tmp/char/save/cn_sep_g1.0_0.1