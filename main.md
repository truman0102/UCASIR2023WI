# Introduction

# Background

- 介绍重排任务
- 现有方法

# Methodologies

## Data Augmentation

- 抽取hard negative sample
- 使用docT5query生成new queries

## Contrastive Learning

$$
\begin{aligned}
L&=-\log\frac{\exp(score(q,d^+))}{\sum_{d\in G_q}\exp(score(q,d))}\\
&=-score(q,d^+)+\log\sum_{d\in G_q}\exp(score(q,d))
\end{aligned}
$$

where $G_q$ is the set of passages containing the positive passage $d^+$ for query $q$ and sampled negative passages $d^-$, and $score(q,d)$ is the score of the pair of query $q$ and passage $d$, which is the output of the model. Given a set of queries $Q$ and passages $D$, the loss function is

$$
L=\frac{1}{|Q|}\sum_{q\in Q}\sum_{d\in G_q}L(q,d)
$$

# Experiments

## Datasets

## Model

# Results

# Conclusion
