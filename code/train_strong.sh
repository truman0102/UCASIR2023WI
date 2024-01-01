#!/bin/bash

EXP=/data/zhouyan/EXP/IR2023
PRETRAIN=${EXP}/bert-base-mdoc-bm25
DATA=${EXP}/data

bsz=4
epoch=1
lr=1e-5
num_example=10
max_len=512

CKPT=${EXP}/ckpt/freeze-strong-num${num_example}-bsz${bsz}-lr${lr}
LOG_DIR=${EXP}/log/freeze-strong-num${num_example}-bsz${bsz}-lr${lr}
mkdir -p ${CKPT}
mkdir -p ${LOG_DIR}

CUDA_VISIBLE_DEVICES=2 python train.py \
    --device cuda \
    --freeze \
    --strong-negative \
    --data-dir ${DATA} \
    --pretrained-path ${PRETRAIN} \
    --save-dir ${CKPT} \
    --log-dir ${LOG_DIR} \
    --num-epoch ${epoch} \
    --max-len ${max_len} \
    --num-example ${num_example} \
    --batch-size ${bsz} \
    --lr ${lr} \
    |& tee -a ${LOG_DIR}/train.log
