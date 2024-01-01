<!-- ---
output:
  html_document:
    toc: yes
--- -->

# UCASIR2023WI

This is the group work for Information Retrieval course in UCAS.

## Introduction

The task of this project is to rerank retrieved passages of [TREC 2022](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2022).

Know more aoubt TREC 2022 from the [Overview paper](https://trec.nist.gov/pubs/trec31/papers/Overview_deep.pdf) and [other reports](https://trec.nist.gov/pubs/trec31/xref.html#deep).

## Model

Our model is based on the [Reranker]([#reranker-required](https://github.com/luyug/Reranker)) model. The input of the model is the embedding (max length 512) of a pair of query and passage/sentence, and the output is a score indicating the relevance of the pair.

Download the [pretrained model](https://huggingface.co/Luyu/bert-base-mdoc-bm25) from hugingface.

> Gao, L., Dai, Z., & Callan, J. (2021). Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline (arXiv:2101.08751). arXiv. https://doi.org/10.48550/arXiv.2101.08751

## Train

1. 准备好bert-base-mdoc-bm25预训练模型

2. 在linux下`./train.sh`训练基础模型，`./train_strong`训练增强负例模型。注意修改\$CKPT和\$LOG_DIR两个目录，以及CUDA_VISIBLE_DEVICES

## Test

1. 复制special_tokens_map.json，tokenizer_config.json与vocab.txt到\$CKPT目录下
2. `CUDA_VISIBLE_DEVICES=0 python test.py`进行测试，注意修改15行的模型名为\$CKPT目录的名字。输出结果在res/\$CKPT目录下
3. 使用trec_eval工具计算ndcg_cut值，[详见](#evaluation)

## Evaluation

### Results

本实验提供的重排结果包括
1. bert_base_top100.txt: bert-base-mdoc-bm25模型的top100结果
2. model_5_top100.txt: freeze-num5-bsz4-lr3e-5-epoch1模型的top100结果(自训练)
3. model_10_top100.txt: freeze-strong-num10-bsz4-lr1e-4-epoch2模型的top100结果(自训练)

### Metrics

- NDGC@10
- Ground Truth: files that contain `qrels` in their names

1. cd to `trec_eval` folder
2. run './trec_eval -m ndcg_cut `path/to/qrels` `path/to/top100`

### Run

```bash
./trec_eval -m ndcg_cut ../data/test_2022.qrels.pass.withDupes.txt ../data/test_2022_passage_top100.txt
ndcg_cut_5              all     0.2888
ndcg_cut_10             all     0.2692
ndcg_cut_15             all     0.2561
ndcg_cut_20             all     0.2524
ndcg_cut_30             all     0.2432
ndcg_cut_100            all     0.2133
ndcg_cut_200            all     0.1711
ndcg_cut_500            all     0.1575
ndcg_cut_1000           all     0.1555
```

```bash
./trec_eval -m ndcg_cut ../data/test_2022.qrels.pass.withDupes.txt ../res/bert-base-mdoc-bm25.trec
ndcg_cut_5              all     0.4917
ndcg_cut_10             all     0.4365
ndcg_cut_15             all     0.4111
ndcg_cut_20             all     0.3912
ndcg_cut_30             all     0.3630
ndcg_cut_100            all     0.2622
ndcg_cut_200            all     0.2152
ndcg_cut_500            all     0.2007
ndcg_cut_1000           all     0.1987
```

```bash
./trec_eval -m ndcg_cut ../data/test_2022.qrels.pass.withDupes.txt ../res/freeze-num5-bsz4-lr3e-5-epoch1.trec
ndcg_cut_5              all     0.4938
ndcg_cut_10             all     0.4415
ndcg_cut_15             all     0.4169
ndcg_cut_20             all     0.3942
ndcg_cut_30             all     0.3646
ndcg_cut_100            all     0.2630
ndcg_cut_200            all     0.2156
ndcg_cut_500            all     0.2009
ndcg_cut_1000           all     0.1989
```

```bash
./trec_eval -m ndcg_cut ../data/test_2022.qrels.pass.withDupes.txt ../res/freeze-strong-num10-bsz4-lr1e-4-epoch2.trec
ndcg_cut_5              all     0.4714
ndcg_cut_10             all     0.4289
ndcg_cut_15             all     0.4055
ndcg_cut_20             all     0.3826
ndcg_cut_30             all     0.3580
ndcg_cut_100            all     0.2589
ndcg_cut_200            all     0.2118
ndcg_cut_500            all     0.1973
ndcg_cut_1000           all     0.1953
```

## Methodology

### Data Augmentation

Data augmentation consists of two parts: query augmentation and passage augmentation.

Query augmentation is to generate new queries based on the given passage. The method we use is to via DocT5query model, with the prefix `text2query` and the passage as the input, joined by ": ". The output is $k$ new queries, where $k$ is a hyperparameter.

### Training

#### Contrastive Loss

$$
\begin{aligned}
L&=-\log\frac{\exp(score(q,d^+))}{\sum_{d\in G_q}\exp(score(q,d))}\\
&=-score(q,d^+)+\log\sum_{d\in G_q}\exp(score(q,d))
\end{aligned}
$$

where $G_q$ is the set of passages containing the positive passage $d^+$ for query $q$ and sampled negative passages $d^-$, and $score(q,d)$ is the score of the pair of query $q$ and passage $d$, which is the output of the model.

Given a set of queries $Q$ and passages $D$, the loss function is

$$
L=\frac{1}{|Q|}\sum_{q\in Q}\sum_{d\in G_q}L(q,d)
$$

It is worth noting that the CL aims to maximize the score of the positive passage and minimize the scores of the sum of all passages, strictly requiring the quality of retrieved passages $G_q$.

## Structure

```bash
├── code
│   ├── ...
│   │   ├── ...
│   ├── data.ipynb
│   ├── reranker.ipynb
│   ├── test.ipynb
│   ├── utils.py
│   └── ...
├── data
│   ├── ...
│   │   ├── ...
│   ├── collection.sampled.tsv
│   ├── train_sample_passv2_qrels.tsv
│   ├── train_sample_queries.tsv
│   └── ...
├── res
│   ├── ...
│   │   ├── ...
│   ├── bert-base-mdoc-bm25.trec
│   ├── freeze-num5-bsz4-lr3e-5-epoch1.trec
│   ├── freeze-strong-num10-bsz4-lr1e-4-epoch2.trec
│   └── ...
├── model
│   ├── Reranker
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   └── ...
├── figure
├── trec_eval
│   ├── test
│   │   ├── ...
│   └── ...
├── papers.bib
├── requirements.txt
├── README.md
├── .gitignore
└── ...
```

## Installation

[Requirements](https://github.com/luyug/Reranker#installation-and-dependencies) from the reference model should be fllowed.

## Dataset

A [data processor](/code/utils.py) is provided to load the dataset. The dataset is stored in the `data` folder. Call `read_in_memory()` to load the dataset into memory.

```python
from utils import Data
data_processor = Data()
data_processor.read_in_memory()
```
```bash
# output
正在处理文件collection.sampled.tsv 读取文件的格式为('pid', 'passage')
正在处理文件train_sample_queries.tsv 读取文件的格式为('qid', 'query')
正在处理文件train_sample_passv2_qrels.tsv 读取文件的格式为('qid', 'mark', 'pid', 'rating')
正在处理文件val_2021_53_queries.tsv 读取文件的格式为('qid', 'query')
正在处理文件val_2021_passage_top100.txt 读取文件的格式为('qid', 'mark', 'pid', 'rank', 'score', 'sys_id')
正在处理文件val_2021.qrels.pass.final.txt 读取文件的格式为('qid', 'mark', 'pid', 'rating')
正在处理文件test_2022_76_queries.tsv 读取文件的格式为('qid', 'query')
正在处理文件test_2022_passage_top100.txt 读取文件的格式为('qid', 'mark', 'pid', 'rank', 'score', 'sys_id')
正在处理文件test_2022.qrels.pass.withDupes.txt 读取文件的格式为('qid', 'mark', 'pid', 'rating')
```

A tutorial of how to load and process the dataset can be found in [data.ipynb](/code/data.ipynb)