#!/usr/bin/env bash

## scale_cn for ptb_base
#python -m nmt.nmt \
#    --unit_type=layer_norm_lstm \
#    --src=vi --tgt=en \
#    --learning_rate=1.0 \
#    --grain=0.0 \
#    --vocab_prefix=../../data/nmt_data/vocab  \
#    --train_prefix=../../data/nmt_data/train \
#    --dev_prefix=../../data/nmt_data/tst2012  \
#    --test_prefix=../../data/nmt_data/tst2013 \
#    --out_dir=/tmp/nmt_model/default_layernorm \
#    --num_train_steps=12000 \
#    --steps_per_stats=100 \
#    --num_layers=2 \
#    --num_units=128 \
#    --dropout=0.2 \
#    --metrics=bleu

export CUDA_VISIBLE_DEVICES=1
python -m nmt.nmt \
    --attention=scaled_luong \
    --unit_type=base \
    --src=vi --tgt=en \
    --encoder_type=bi \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=$HOME/log/nmt_attention_model \
    --learning_rate=1.0 \
    --grain=5.0 \
    --start_decay_step=8000 \
    --decay_steps=1000 \
    --decay_facto=0.5 \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=512 \
    --dropout=0.2 \
    --metrics=bleu
