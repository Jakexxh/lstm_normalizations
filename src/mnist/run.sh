#!/usr/bin/env bash
python test_784*1_base.py --cell=wn_sep --log_dir=/tmp/logs/mnist_final --g=0.5 --lr=0.001
python test_784*1_base.py --cell=wn_sep --log_dir=/tmp/logs/mnist_final --g=0.5 --lr=0.01
python test_784*1_base.py --cell=wn_sep --log_dir=/tmp/logs/mnist_final --g=0.5 --lr=0.1

python test_784*1_base.py --cell=wn_sep --log_dir=/tmp/logs/mnist_final --g=1.0 --lr=0.001
python test_784*1_base.py --cell=wn_sep --log_dir=/tmp/logs/mnist_final --g=1.0 --lr=0.01
python test_784*1_base.py --cell=wn_sep --log_dir=/tmp/logs/mnist_final --g=1.0 --lr=0.1

python test_784*1_base.py --cell=wn_sep --log_dir=/tmp/logs/mnist_final --g=5.0 --lr=0.001
python test_784*1_base.py --cell=wn_sep --log_dir=/tmp/logs/mnist_final --g=5.0 --lr=0.01
python test_784*1_base.py --cell=wn_sep --log_dir=/tmp/logs/mnist_final --g=5.0 --lr=0.1

python test_784*1_base.py --cell=wn_sep --log_dir=/tmp/logs/mnist_final --g=0.1 --lr=0.001
python test_784*1_base.py --cell=wn_sep --log_dir=/tmp/logs/mnist_final --g=0.1 --lr=0.01
python test_784*1_base.py --cell=wn_sep --log_dir=/tmp/logs/mnist_final --g=0.1 --lr=0.1

python test_784*1_base.py --cell=base --log_dir=/tmp/logs/mnist_final --g=0.0 --lr=0.001
python test_784*1_base.py --cell=base --log_dir=/tmp/logs/mnist_final --g=0.0 --lr=0.001
python test_784*1_base.py --cell=base --log_dir=/tmp/logs/mnist_final --g=0.0 --lr=0.001

