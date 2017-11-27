#!/usr/bin/env bash

#python train.py --model=base --lr=0.001  --save_dir=/tmp/char_seq100/save/base_0.001 --log_dir=/tmp/char_seq100/log/base_0.001
#python sample.py --save_dir=/tmp/char_seq100/save/base_0.001
#python train.py --model=base --lr=0.01  --save_dir=/tmp/char_seq100/save/base_0.01 --log_dir=/tmp/char_seq100/log/base_0.01
#python sample.py --save_dir=/tmp/char_seq100/save/base_0.01
#python train.py --model=base --lr=0.1  --save_dir=/tmp/char_seq100/save/base_0.1 --log_dir=/tmp/char_seq100/log/base_0.1
#python sample.py --save_dir=/tmp/char_seq100/save/base_0.1

# python train.py --model=cn_sep --lr=0.001  --save_dir=/tmp/char_seq100/save/cn_sep_g1.0_0.001 --log_dir=/tmp/char_seq100/log/cn_sep_g1.0_0.001
# python sample.py --save_dir=/tmp/char_seq100/save/cn_sep_g1.0_0.001
# python train.py --model=cn_sep --lr=0.01  --save_dir=/tmp/char_seq100/save/cn_sep_g1.0_0.01 --log_dir=/tmp/char_seq100/log/cn_sep_g1.0_0.01
# python sample.py --save_dir=/tmp/char_seq100/save/cn_sep_g1.0_0.01
# python train.py --model=cn_sep --lr=0.1  --save_dir=/tmp/char_seq100/save/cn_sep_g1.0_0.1 --log_dir=/tmp/char_seq100/log/cn_sep_g1.0_0.1
# python sample.py --save_dir=/tmp/char_seq100/save/cn_sep_g1.0_0.1

python train.py --model=cn_sep_scale --lr=0.001  --save_dir=/tmp/char_seq100/save/cn_sep_scale_g1.0_0.001 --log_dir=/tmp/char_seq100/log/cn_sep_scale_g1.0_0.001
python sample.py --save_dir=/tmp/char_seq100/save/cn_sep_scale_g1.0_0.001
python train.py --model=cn_sep_scale --lr=0.01  --save_dir=/tmp/char_seq100/save/cn_sep_scale_g1.0_0.01 --log_dir=/tmp/char_seq100/log/cn_sep_scale_g1.0_0.01
python sample.py --save_dir=/tmp/char_seq100/save/cn_sep_scale_g1.0_0.01
python train.py --model=cn_sep_scale --lr=0.1  --save_dir=/tmp/char_seq100/save/cn_sep_scale_g1.0_0.1 --log_dir=/tmp/char_seq100/log/cn_sep_scale_g1.0_0.1
python sample.py --save_dir=/tmp/char_seq100/save/cn_sep_scale_g1.0_0.1

#python train.py --model=wn_sep --lr=0.001  --save_dir=/tmp/char/save/wn_sep_g5.0_0.001 --log_dir=/tmp/char/log/wn_sep_g5.0_0.001
#python sample.py --save_dir=/tmp/char/save/wn_sep_g5.0_0.001
#python train.py --model=wn_sep --lr=0.01  --save_dir=/tmp/char/save/wn_sep_g5.0_0.01 --log_dir=/tmp/char/log/wn_sep_g5.0_0.01
#python sample.py --save_dir=/tmp/char/save/wn_sep_g5.0_0.01
#python train.py --model=wn_sep --lr=0.1  --save_dir=/tmp/char/save/wn_sep_g5.0_0.1 --log_dir=/tmp/char/log/wn_sep_g5.0_0.1
#python sample.py --save_dir=/tmp/char/save/wn_sep_g5.0_0.1
#python train.py --model=wn_sep --lr=1.0  --save_dir=/tmp/char/save/wn_sep_g1.0_1.0 --log_dir=/tmp/char/log/wn_sep_g1.0_1.0
#python sample.py --save_dir=/tmp/char/save/cn_sep_g1.0_1.0
