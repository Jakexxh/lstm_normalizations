#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=0
python -m nmt.nmt \
    --attention=scaled_luong \
    --unit_type=base \
    --src=vi --tgt=en \
    --encoder_type=bi \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=$HOME/log/nmt_attention_model2_base \
    --learning_rate=1.0 \
    --grain=0.0 \
    --start_decay_step=8000 \
    --decay_steps=1000 \
    --decay_factor=0.5 \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=512 \
    --dropout=0.2 \
    --metrics=bleu
