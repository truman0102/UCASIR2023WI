# UCASIR2023WI

This is the group work for Information Retrieval course in UCAS.

## Introduction

The task of this project is to rerank retrieved passages of [TREC 2022](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2022).

Know more aoubt TREC 2022 from the [Overview paper](https://trec.nist.gov/pubs/trec31/papers/Overview_deep.pdf) and [other reports](https://trec.nist.gov/pubs/trec31/xref.html#deep).

## Dataset

A tutorial of how to load and process the dataset can be found in [data.ipynb](/code/data.ipynb)

## Structure

```bash
├── code
│   ├── ...
│   │   ├── ...
│   ├── data.ipynb
│   ├── utils.py
│   └── ...
├── data
│   ├── ...
│   │   ├── ...
│   ├── collection.sampled.tsv
│   ├── train_sample_passv2_qrels.tsv
│   ├── train_sample_queries.tsv
│   ├── ...
├── trec_eval
│   ├── test
│   │   ├── ...
│   └── ...
├── .gitignore
├── requirements.txt
├── README.md
└── ...
```

## Evaluation

### Metrics

- NDGC@10
- Ground Truth: files that contain `qrels` in their names

### Run