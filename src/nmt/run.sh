#!/usr/bin/env bash

## scale_cn for ptb_base

python -m nmt.nmt \
    --unit_type=base \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --grain=0.0 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_ref \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

python -m nmt.nmt \
    --unit_type=cn_sep \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --grain=1.0 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_ref \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

python -m nmt.nmt \
    --unit_type=cn_sep \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --grain=5.0 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_ref \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

python -m nmt.nmt \
    --unit_type=cn_sep \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --grain=10.0 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_ref \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

python -m nmt.nmt \
    --unit_type=pcc_sep \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --grain=1.0 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_ref \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

python -m nmt.nmt \
    --unit_type=pcc_sep \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --grain=5.0 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_ref \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

python -m nmt.nmt \
    --unit_type=pcc_sep \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --grain=10.0 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_ref \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

python -m nmt.nmt \
    --unit_type=wn_sep \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --grain=0.5 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_ref \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

python -m nmt.nmt \
    --unit_type=wn_sep \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --grain=1.0 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_ref \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

python -m nmt.nmt \
    --unit_type=wn_sep \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --grain=5.0 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_ref \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

python -m nmt.nmt \
    --unit_type=ln_sep \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --grain=0.05 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_ref \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

python -m nmt.nmt \
    --unit_type=ln_sep \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --grain=0.1 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_ref \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

python -m nmt.nmt \
    --unit_type=ln_sep \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --grain=0.5 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_ref \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

#python -m nmt.nmt \
#    --unit_type=bn_sep \
#    --src=vi --tgt=en \
#    --learning_rate=1.0 \
#    --grain=0.05 \
#    --vocab_prefix=../../data/nmt_data/vocab  \
#    --train_prefix=../../data/nmt_data/train \
#    --dev_prefix=../../data/nmt_data/tst2012  \
#    --test_prefix=../../data/nmt_data/tst2013 \
#    --out_dir=/tmp/nmt_model/nmt_ref \
#    --num_train_steps=12000 \
#    --steps_per_stats=100 \
#    --num_layers=2 \
#    --num_units=128 \
#    --dropout=0.2 \
#    --metrics=bleu
#
#python -m nmt.nmt \
#    --unit_type=bn_sep \
#    --src=vi --tgt=en \
#    --learning_rate=1.0 \
#    --grain=0.1 \
#    --vocab_prefix=../../data/nmt_data/vocab  \
#    --train_prefix=../../data/nmt_data/train \
#    --dev_prefix=../../data/nmt_data/tst2012  \
#    --test_prefix=../../data/nmt_data/tst2013 \
#    --out_dir=/tmp/nmt_model/nmt_ref \
#    --num_train_steps=12000 \
#    --steps_per_stats=100 \
#    --num_layers=2 \
#    --num_units=128 \
#    --dropout=0.2 \
#    --metrics=bleu
#
#python -m nmt.nmt \
#    --unit_type=bn_sep \
#    --src=vi --tgt=en \
#    --learning_rate=1.0 \
#    --grain=0.5 \
#    --vocab_prefix=../../data/nmt_data/vocab  \
#    --train_prefix=../../data/nmt_data/train \
#    --dev_prefix=../../data/nmt_data/tst2012  \
#    --test_prefix=../../data/nmt_data/tst2013 \
#    --out_dir=/tmp/nmt_model/nmt_ref \
#    --num_train_steps=12000 \
#    --steps_per_stats=100 \
#    --num_layers=2 \
#    --num_units=128 \
#    --dropout=0.2 \
#    --metrics=bleu
#
