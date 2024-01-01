from typing import List

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataset import RerankDatasetValItem
from utils import logger


class NCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits, hidden_states=None):
        # logits: [B, N]
        # B: batch size, N: num example (0=positive, 1...=negative)
        # logits中每一条为query和passage的相似度分数
        # InfoNCE loss
        probs = F.softmax(logits / self.temperature, dim=1)
        pos_score = probs[:, 0]
        neg_score = probs[:, 1:]
        loss = - torch.log(pos_score / (pos_score + neg_score.sum(dim=1)))
        loss = loss.mean()
        return loss


class NCELossStrong(nn.Module):
    def __init__(self, num_negative, temperature=0.05):
        super().__init__()
        self.num_negative = num_negative
        self.temperature = temperature

    def forward(self, logits, hidden_states):
        # InfoNCE loss
        # 从负样本中选取与正样本相似度最高的num_negative个样本
        pos_score, neg_score = self.get_closest(logits, hidden_states)
        loss = - torch.log(pos_score / (pos_score + neg_score.sum(dim=1)))
        loss = loss.mean()
        return loss

    def get_closest(self, logits, hidden_states):
        # logits: [B, N]
        # B: batch size, N: num example (0=positive, 1...=negative)
        # logits中每一条为query和passage的相似度分数
        # 从负样本中选取与正样本相似度最高的num_negative个样本
        # hidden_states: [B, N, L, H]
        # 计算句子隐状态
        hidden_pos = hidden_states[:, 0] # [B, L, H]
        hidden_pos_mean = hidden_pos.mean(dim=1) # [B, H]
        hidden_neg = hidden_states[:, 1:] # [B, N-1, L, H]
        hidden_neg_mean = hidden_neg.mean(dim=2) # [B, N-1, H]
        # 计算每个负样本与正样本的相似度
        sims = torch.cosine_similarity(hidden_pos_mean.unsqueeze(1), hidden_neg_mean, dim=2) # [B, N-1]
        topk, indices = torch.topk(sims, self.num_negative, dim=1) # [B, num_negative]
        # 选取与正样本相似度最高的num_negative个样本
        probs = F.softmax(logits / self.temperature, dim=1)
        pos_score = probs[:, 0]
        neg_score = probs[:, 1:].gather(1, indices)
        return pos_score, neg_score


def log(base, x):
    """
    log function with base
    """
    if x == 0:
        return 0
    else:
        return np.log(x) / np.log(base)


class NDCG:
    def __init__(self, direct_gain=None):
        self.update(direct_gain)

    @property
    def length(self):
        return self.direct_gain.shape[0]

    def __call__(self, direct_gain=None, method="NDCG"):
        if direct_gain is not None:
            self.direct_gain = direct_gain
        if method == "NDCG":
            return self.NDCG()
        elif method == "DCG":
            return self.DCG()
        elif method == "NCG":
            return self.NCG()
        elif method == "CG":
            return self.CG()
        else:
            raise ValueError("method must be NDCG, DCG, NCG or CG")

    def update(self, rating):
        if isinstance(rating, (list, tuple)):
            self.direct_gain = np.array(rating)
        elif isinstance(rating, np.ndarray):
            self.direct_gain = rating
        elif rating is None:
            pass
        else:
            raise ValueError("rating must be list, tuple or ndarray")

    def CG(self, vector=None):
        """
        Cumulative Gain
        """
        if vector is None:
            vector = self.direct_gain
        return np.cumsum(vector)

    def DCG(self, b=2, vector=None):
        """
        Discounted Cumulative Gain
        """
        if vector is None:
            vector = self.direct_gain
        cg = self.CG(vector)
        res = np.zeros_like(vector, dtype=float)
        for i in range(b - 1):
            res[i] = cg[i]
        for i in range(b - 1, self.length):
            res[i] = res[i - 1] + vector[i] / log(b, i + 1)

        return res

    def BV(self):
        """
        Best Vector
        """
        return np.sort(self.direct_gain)[::-1]

    def NCG(self, vector=None):
        """
        Normalized Cumulative Gain
        """
        return self.CG() / self.CG(self.BV())

    def NDCG(self, b=2):
        """
        Normalized Discounted Cumulative Gain
        """
        return self.DCG(b) / self.DCG(b, self.BV())

    def NDCG_at_k(self, k=10, b=2):
        """
        Normalized Discounted Cumulative Gain at k
        """
        return self.NDCG(b)[k - 1]


def calcu_ndcg(item: RerankDatasetValItem, logits: torch.Tensor):
    res = NDCG()
    ratings = item.ratings
    res.update(ratings)
    res.NDCG_at_k()
    # logits: [1, N]
    # rerank 100 passages based on logits
    scores, indices = torch.sort(logits, descending=True)
    # [1, N] -> [N]
    indices = indices.squeeze(0).tolist()
    # rerank
    reranked_ratings = [ratings[i] for i in indices]
    res.update(reranked_ratings)
    ndcg = res.NDCG_at_k()
    return ndcg


def calcu_ndcg_dry(items: List[RerankDatasetValItem]):
    res = NDCG()
    ndcg = 0
    for item in items:
        scores = item.scores
        ratings = item.ratings
        res.update(ratings)
        res.NDCG_at_k()
        # rerank 100 passages based on scores
        indices = np.argsort(scores)[::-1].tolist()
        reranked_ratings = [ratings[i] for i in indices]
        res.update(reranked_ratings)
        sample_res = res.NDCG_at_k()
        # nan or inf detected
        if np.isnan(sample_res) or np.isinf(sample_res):
            logger(f"NaN or Inf detected, qid: {item.qid}, sample_res: {sample_res}")
        else:
            ndcg += sample_res
    ndcg /= len(items)
    return ndcg


def get_reranked(item: RerankDatasetValItem, logits: torch.Tensor):
    ratings = item.ratings
    # logits: [1, N]
    # rerank 100 passages based on logits
    scores, indices = torch.sort(logits, descending=True)
    # [1, N] -> [N]
    indices = indices.squeeze(0).tolist()
    # rerank
    reranked_ratings = [ratings[i] for i in indices]
    reranked_scores = scores.squeeze(0).tolist()
    reranked_pids = [item.pids[i] for i in indices]
    reranked_passages = [item.passages[i] for i in indices]
    reranked_item = RerankDatasetValItem(
        qid=item.qid,
        query=item.query,
        pids=reranked_pids,
        passages=reranked_passages,
        scores=reranked_scores,
        ratings=reranked_ratings
    )
    return reranked_item