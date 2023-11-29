# UCASIR2023WI

This is the group work for Information Retrieval course in UCAS.

## Introduction

The task of this project is to rerank retrieved passages of [TREC 2022](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2022).

Know more aoubt TREC 2022 from the [Overview paper](https://trec.nist.gov/pubs/trec31/papers/Overview_deep.pdf) and [other reports](https://trec.nist.gov/pubs/trec31/xref.html#deep).

## Dataset

A data processor is provided to load the dataset.

```python
from /code/utils import Data
data_processor = Data()
```
Call `read_in_memory()` to load the dataset into memory.

```python
data_processor.read_in_memory()
```

A tutorial of how to load and process the dataset can be found in [data.ipynb](/code/data.ipynb)

## Models

### Reference Model

#### [Reranker](https://github.com/luyug/Reranker) (required)

Download the model from [here](https://huggingface.co/Luyu/bert-base-mdoc-bm25)

#### [~~ColBERT~~](https://github.com/stanford-futuredata/ColBERT) (optional)

Download the model from [here](https://huggingface.co/colbert-ir/colbertv2.0)

#### [~~reranking BERT~~](https://github.com/nyu-dl/dl4marco-bert) (optional)

Download the model from [here](https://huggingface.co/amberoad/bert-multilingual-passage-reranking-msmarco)

### Reports in TREC 2022

#### [CIP](https://trec.nist.gov/pubs/trec31/papers/CIP.D.pdf)

#### [HLATR](https://trec.nist.gov/pubs/trec31/papers/Ali.D.pdf)

### Ours

Our model is based on the [reranker](#reranker-required) model.

See [reranker.ipynb](/code/reranker.ipynb) and [test.ipynb](/code/test.ipynb) for details.

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
├── model
│   ├── Reranker
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   ├── ColBERT
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   ├── rBERT
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
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

## Evaluation

### Metrics

- NDGC@10
- Ground Truth: files that contain `qrels` in their names

### Run
