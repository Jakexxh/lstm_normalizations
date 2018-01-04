#!/usr/bin/env bash

#python draw.py --cell=base --g=0.0

python draw.py --cell=cn_sep --g=1.0
python draw.py --cell=cn_sep --g=5.0
python draw.py --cell=cn_sep --g=10.0

python draw.py --cell=wn_sep --g=0.01
python draw.py --cell=wn_sep --g=0.1
python draw.py --cell=wn_sep --g=1.0
python draw.py --cell=wn_sep --g=5.0

python draw.py --cell=ln_sep --g=0.01
python draw.py --cell=ln_sep --g=0.1
python draw.py --cell=ln_sep --g=1.0
python draw.py --cell=ln_sep --g=5.0

python draw.py --cell=pcc_sep --g=1.0
python draw.py --cell=pcc_sep --g=5.0
python draw.py --cell=pcc_sep --g=10.0

python draw.py --cell=bn_sep --g=0.01
python draw.py --cell=bn_sep --g=0.1
python draw.py --cell=bn_sep --g=1.0
