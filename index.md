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

## Dataset

A [data processor](https://github.com/truman0102/UCASIR2023WI/blob/main/code/utils.py) is provided to load the dataset. The dataset is stored in the `data` folder. Call `read_in_memory()` to load the dataset into memory.

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

A tutorial of how to load and process the dataset can be found in [data.ipynb](https://github.com/truman0102/UCASIR2023WI/blob/main/code/data.ipynb)

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

Our model is based on the [Reranker](#reranker-required) model. The input of the model is the embedding (max length 512) of a pair of query and passage/sentence, and the output is a score indicating the relevance of the pair.

See [reranker.ipynb](https://github.com/truman0102/UCASIR2023WI/blob/main/code/reranker.ipynb) and [test.ipynb](https://github.com/truman0102/UCASIR2023WI/blob/main/code/test.ipynb) for details.

## Training

### Data Generation

Reranker has the advantage of being able to give a higher score to the correct answer, but it does not do a good job of judging texts that are semantically similar but are not actually the correct answer.

Given the correct answer to a question, we try to replace certain words in the text to generate a semantically similar but incorrect answer. Random substitutions can be detrimental to the legitimacy of the utterance and the quality of the generated data, such as replacing a noun with a verb or changing an irrelevant preposition. In contrast, replacing words by referring to dependency analysis trees in natural language processing results in more stable generated data.

For each passage, we split it into sentences and conduct tokenization and dependency analysis, generating a dictionary to store the index, lemma, POS, and dependency of each token. Each passage is constructed into a dictionary with the following format:

```python
{
  passage_id:{
    1: [
      (token_index, token_text, token_lemma, token_pos, token_tag, token_dep, token_parent, token_parent_index),
      ...
    ]
    2: [
      (token_index, token_text, token_lemma, token_pos, token_tag, token_dep, token_parent, token_parent_index),
      ...
    ]
    ...
  }
}
```

The data is stored in [passage_tokens.pkl]() and is detailed in [gen.ipynb](https://github.com/truman0102/UCASIR2023WI/blob/main/code/gen.ipynb).

### Loss Function

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
│   ├── gen.ipynb
│   ├── model.ipynb
│   ├── test.ipynb
│   ├── utils.py
│   ├── model.py
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
│   └── ...
├── figure
├── trec_eval
│   ├── test
│   │   ├── ...
│   └── ...
├── papers.bib
├── requirements.txt
├── README.md
├── index.html
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