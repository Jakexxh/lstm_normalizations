#!/usr/bin/env bash
#python test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/vani_mnist_ref/cn_sep_1.0_0.001  --lr=0.001
#python test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/vani_mnist_ref/cn_sep_1.0_0.01  --lr=0.01
#python test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/vani_mnist_ref/cn_sep_1.0_0.1 --lr=0.1
#
#python test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/vani_mnist_ref/wn_sep_1.0_0.001  --lr=0.001
#python test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/vani_mnist_ref/wn_sep_1.0_0.01  --lr=0.01
#python test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/vani_mnist_ref/wn_sep_1.0_0.1 --lr=0.1


python test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/vani_mnist_ref/bn_sep_0.1_0.001  --lr=0.001
python test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/vani_mnist_ref/bn_sep_0.1_0.01  --lr=0.01
python test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/vani_mnist_ref/bn_sep_0.1_0.1 --lr=0.1

python test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/vani_mnist_ref/ln_sep_0.1_0.001  --lr=0.001
python test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/vani_mnist_ref/ln_sep_0.1_0.01  --lr=0.01
python test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/vani_mnist_ref/ln_sep_0.1_0.1 --lr=0.1


#python test_784*1.py --cell=base --log_dir=/tmp/logs/vani_mnist_ref/base_0.001  --lr=0.001
#python test_784*1.py --cell=base --log_dir=/tmp/logs/vani_mnist_ref/base_0.01  --lr=0.01
#python test_784*1.py --cell=base --log_dir=/tmp/logs/vani_mnist_ref/base_0.1 --lr=0.1
#
#python test_784*1.py --cell=pcc_sep --log_dir=/tmp/logs/vani_mnist_ref/pcc_sep_1.0_0.001  --lr=0.001
#python test_784*1.py --cell=pcc_sep --log_dir=/tmp/logs/vani_mnist_ref/pcc_sep_1.0_0.01  --lr=0.01
#python test_784*1.py --cell=pcc_sep --log_dir=/tmp/logs/vani_mnist_ref/pcc_sep_1.0_0.1 --lr=0.1


