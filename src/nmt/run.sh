#!/usr/bin/env bash

#python -m nmt.nmt \
#    --unit_type=base \
#    --src=vi --tgt=en \
#    --learning_rate=1.0 \
#    --vocab_prefix=../../data/nmt_data/vocab  \
#    --train_prefix=../../data/nmt_data/train \
#    --dev_prefix=../../data/nmt_data/tst2012  \
#    --test_prefix=../../data/nmt_data/tst2013 \
#    --out_dir=/tmp/nmt_model/nmt_1/base_0.1 \
#    --num_train_steps=12000 \
#    --steps_per_stats=100 \
#    --num_layers=2 \
#    --num_units=128 \
#    --dropout=0.2 \
#    --metrics=bleu

python -m nmt.nmt \
   --unit_type=cn_sep \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_1/cn_sep_g100.0_lr1.0 \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

#python -m nmt.nmt \
#    --unit_type=wn_sep \
#    --src=vi --tgt=en \
#    --learning_rate=1.0 \
#    --vocab_prefix=../../data/nmt_data/vocab  \
#    --train_prefix=../../data/nmt_data/train \
#    --dev_prefix=../../data/nmt_data/tst2012  \
#    --test_prefix=../../data/nmt_data/tst2013 \
#    --out_dir=/tmp/nmt_model/nmt_1/wn_sep_g10.0_lr1.0 \
#    --num_train_steps=12000 \
#    --steps_per_stats=100 \
#    --num_layers=2 \
#    --num_units=128 \
#    --dropout=0.2 \
#    --metrics=bleu
#
#python -m nmt.nmt \
#    --unit_type=ln_sep \
#    --src=vi --tgt=en \
#    --learning_rate=1.0 \
#    --vocab_prefix=../../data/nmt_data/vocab  \
#    --train_prefix=../../data/nmt_data/train \
#    --dev_prefix=../../data/nmt_data/tst2012  \
#    --test_prefix=../../data/nmt_data/tst2013 \
#    --out_dir=/tmp/nmt_model/nmt_1/ln_sep_g1.0_lr1.0 \
#    --num_train_steps=12000 \
#    --steps_per_stats=100 \
#    --num_layers=2 \
#    --num_units=128 \
#    --dropout=0.2 \
#    --metrics=bleu

python -m nmt.nmt \
    --unit_type=pcc_sep \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/nmt_1/pcc_sep_g100.0_lr1.0 \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

# python -m nmt.nmt \
#     --attention=scaled_luong \
#     --src=vi --tgt=en \
#     --vocab_prefix=/tmp/nmt_data/vocab  \
#     --train_prefix=/tmp/nmt_data/train \
#     --dev_prefix=/tmp/nmt_data/tst2012  \
#     --test_prefix=/tmp/nmt_data/tst2013 \
#     --out_dir=/tmp/nmt_attention_model \
#     --num_train_steps=12000 \
#     --steps_per_stats=100 \
#     --num_layers=2 \
#     --num_units=128 \
#     --dropout=0.2 \
#     --metrics=bleu
