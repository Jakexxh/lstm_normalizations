#!/usr/bin/env bash

python train.py --model=base --lr=0.001  --save_dir=/tmp/char/save/base_0.001 --log_dir=/tmp/char/log/base_0.001
python sample.py --save_dir=/tmp/char/save/base_0.001
python train.py --model=base --lr=0.01  --save_dir=/tmp/char/save/base_0.01 --log_dir=/tmp/char/log/base_0.01
python sample.py --save_dir=/tmp/char/save/base_0.01
python train.py --model=base --lr=0.1  --save_dir=/tmp/char/save/base_0.1 --log_dir=/tmp/char/log/base_0.1
python sample.py --save_dir=/tmp/char/save/base_0.1

python train.py --model=base --lr=0.001  --save_dir=/tmp/char/save/cn_sep_g1.0_0.001 --log_dir=/tmp/char/log/cn_sep_g1.0_0.001
python sample.py --save_dir=/tmp/char/save/cn_sep_g1.0_0.001
python train.py --model=base --lr=0.01  --save_dir=/tmp/char/save/cn_sep_g1.0_0.01 --log_dir=/tmp/char/log/cn_sep_g1.0_0.01
python sample.py --save_dir=/tmp/char/save/cn_sep_g1.0_0.01
python train.py --model=base --lr=0.1  --save_dir=/tmp/char/save/cn_sep_g1.0_0.1 --log_dir=/tmp/char/log/cn_sep_g1.0_0.1
python sample.py --save_dir=/tmp/char/save/cn_sep_g1.0_0.1