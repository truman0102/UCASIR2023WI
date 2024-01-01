import os
import pandas as pd

from utils import logger

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import torch


@dataclass
class RerankDatasetItem(object):
    qid: str
    query: str
    passages : List[str]


@dataclass
class RerankDatasetValItem(object):
    qid: str
    query: str
    pids: List[str]
    passages: List[str]
    scores : List[float]
    ratings: List[int]


CORPUS_COLUMNS = ["pid", "passage"]
QUERY_COLUMNS = ["qid", "query"]
QRELS_COLUMNS = ["qid", "zero", "pid", "rating"]
TOP_FILE_COLUMNS = ["qid", "qzero", "pid", "rank", "score", "sys_id"]


class RerankDataset(object):
    def __init__(
        self,
        data_dir: str,
        pretrained_path: str,
        max_len: int = 512,
        num_example: int = 5,
        batch_size: int = 4,
        is_train: bool = True,
    ):
        self.data_dir = data_dir
        self.max_len = max_len
        self.num_example = num_example
        self.batch_size = batch_size
        self.tokenizer = self.load_tokenizer(pretrained_path)
        self.corpus = self.load_corpus()
        if is_train:
            self.train_sets = self.load_train()
            self.dev_sets, self.dev_items = self.load_dev()
        else:
            self.test_sets, self.test_items = self.load_test()

    def load_tokenizer(self, pretrained_path: str):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        return tokenizer

    def load_corpus(self):
        logger("Loading corpus ...")
        corpus_df = pd.read_csv(
            os.path.join(self.data_dir, "collection.sampled.tsv"),
            sep="\t",
            names=CORPUS_COLUMNS,
        )
        return corpus_df

    def load_train(self):
        logger("Loading train data ...")
        logger("Reading train queries ...")
        train_query_df = pd.read_csv(
            os.path.join(self.data_dir, "train_sample_queries.tsv"),
            sep="\t",
            names=QUERY_COLUMNS,
        )
        logger("Reading train qrels ...")
        train_answer_df = pd.read_csv(
            os.path.join(self.data_dir, "train_sample_passv2_qrels.tsv"),
            sep="\t",
            names=QRELS_COLUMNS,
        )
        train_sets = []
        logger("tokenizing train data ...")
        for qid, query in tqdm(zip(train_query_df["qid"], train_query_df["query"]), position=0):
            pid = train_answer_df[train_answer_df["qid"] == qid]["pid"]
            pid = pid.values[0]
            pos_passage = self.corpus[self.corpus["pid"] == pid]["passage"]
            pos_passage = pos_passage.values[0]
            # 从corpus中随机抽取num_example-1个负样本
            neg_passages = self.corpus[self.corpus["pid"] != pid]["passage"].sample(n=self.num_example-1).tolist()
            # 将正样本和负样本拼接起来
            passages = [pos_passage] + neg_passages
            item = RerankDatasetItem(qid, query, passages)
            tokenized = self.tokenize(item, num_example=self.num_example) # dict
            train_sets.append(tokenized)
        return train_sets

    def load_dev(self):
        logger("Loading dev data ...")
        logger("Reading dev queries ...")
        dev_query_df = pd.read_csv(
            os.path.join(self.data_dir, "val_2021_53_queries.tsv"),
            sep="\t",
            names=QUERY_COLUMNS,
        )
        logger("Reading dev top 100...")
        dev_top100 = pd.read_csv(
            os.path.join(self.data_dir, "val_2021_passage_top100.txt"),
            sep=" ",
            names=TOP_FILE_COLUMNS,
        )
        logger("Reading dev qrels ...")
        dev_qrels = pd.read_csv(
            os.path.join(self.data_dir, "val_2021.qrels.pass.final.txt"),
            sep=" ",
            names=QRELS_COLUMNS,
        )
        dev_items = []
        dev_sets = []
        logger("tokenizing dev data ...")
        for qid, query in tqdm(zip(dev_query_df["qid"], dev_query_df["query"]), position=0):
            top100 = dev_top100[dev_top100["qid"] == qid]
            top100 = top100["pid"].tolist()
            scores = dev_top100[dev_top100["qid"] == qid]["score"].tolist()
            scores = [float(score) for score in scores]
            passages = []
            for pid in top100:
                passage = self.corpus[self.corpus["pid"] == pid]["passage"]
                passages.append(passage.values[0])
            # 从qrels里获取qid与pid都相等的条目的rating，不存在的话就是0
            ratings = []
            for pid in top100:
                rating = dev_qrels[(dev_qrels["qid"] == qid) & (dev_qrels["pid"] == pid)]["rating"]
                if rating.empty:
                    ratings.append(0)
                else:
                    ratings.append(rating.values[0])
            item = RerankDatasetValItem(qid, query, top100, passages, scores, ratings)
            dev_items.append(item)
            tokenized = self.tokenize(item, num_example=len(passages))
            # 检测是否有nan或者inf
            for k, v in tokenized.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    logger(f"Tokenize: NaN or Inf dectectd, qid: {qid}, k: {k}, v: {v}")
            dev_sets.append(tokenized)
        return dev_sets, dev_items

    def load_test(self):
        logger("Loading test data ...")
        logger("Reading test queries ...")
        test_query_df = pd.read_csv(
            os.path.join(self.data_dir, "test_2022_76_queries.tsv"),
            sep="\t",
            names=QUERY_COLUMNS,
        )
        logger("Reading test top 100...")
        test_top100 = pd.read_csv(
            os.path.join(self.data_dir, "test_2022_passage_top100.txt"),
            sep=" ",
            names=TOP_FILE_COLUMNS,
        )
        logger("Reading test qrels ...")
        test_qrels = pd.read_csv(
            os.path.join(self.data_dir, "test_2022.qrels.pass.withDupes.txt"),
            sep=" ",
            names=QRELS_COLUMNS,
        )
        test_items = []
        test_sets = []
        logger("tokenizing test data ...")
        for qid, query in tqdm(zip(test_query_df["qid"], test_query_df["query"]), position=0):
            top100 = test_top100[test_top100["qid"] == qid]
            top100 = top100["pid"].tolist()
            scores = test_top100[test_top100["qid"] == qid]["score"].tolist()
            scores = [float(score) for score in scores]
            passages = []
            for pid in top100:
                passage = self.corpus[self.corpus["pid"] == pid]["passage"]
                passages.append(passage.values[0])
            # 从qrels里获取qid与pid都相等的条目的rating，不存在的话就是0
            ratings = []
            for pid in top100:
                rating = test_qrels[(test_qrels["qid"] == qid) & (test_qrels["pid"] == pid)]["rating"]
                if rating.empty:
                    ratings.append(0)
                else:
                    ratings.append(rating.values[0])
            item = RerankDatasetValItem(qid, query, top100, passages, scores, ratings)
            test_items.append(item)
            tokenized = self.tokenize(item, num_example=len(passages))
            # 检测是否有nan或者inf
            for k, v in tokenized.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    logger(f"Tokenize: NaN or Inf dectectd, qid: {qid}, k: {k}, v: {v}")
            test_sets.append(tokenized)
        return test_sets, test_items

    def tokenize(self, item: RerankDatasetItem or RerankDatasetValItem, num_example: int):
        query = item.query
        passages = item.passages # List[str]
        input_ids = torch.empty((num_example, self.max_len), dtype=torch.long)
        attention_mask = torch.empty((num_example, self.max_len), dtype=torch.long)
        for i, passage in enumerate(passages):
            inputs = self.tokenizer(
                query,
                passage,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids[i] = inputs["input_ids"]
            attention_mask[i] = inputs["attention_mask"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def dataloader(self, shuffle=True):
        logger("Creating dataloader ...")
        train_loader = DataLoader(
            self.train_sets,
            batch_size=self.batch_size,
            shuffle=shuffle
        )
        dev_loader = DataLoader(
            self.dev_sets,
            batch_size=1,
            shuffle=False
        )
        return train_loader, dev_loader

    def testloader(self, shuffle=False):
        logger("Creating testloader ...")
        test_loader = DataLoader(
            self.test_sets,
            batch_size=1,
            shuffle=shuffle
        )
        return test_loader