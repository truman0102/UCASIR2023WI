import time

import os
import csv
import math
import numpy as np
from collections import namedtuple, defaultdict

csv.field_size_limit(500 * 1024 * 1024)


def logger(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] {msg}")


def log(base, x):
    """
    log function with base
    """
    if x == 0:
        return 0
    else:
        return np.log(x) / np.log(base)


def rename(filename, conjunction="_"):
    return conjunction.join(filename.split("."))


class Data:
    def __init__(self, root_path=os.getcwd()):
        self.root_path = root_path
        self.data_dict = defaultdict(dict)

        self.data_dict["collection.sampled.tsv"] = {
            "name": "collection.sampled",
            "postfix": "tsv",
            "desc": "",
            "format": ("pid", "passage"),
        }
        self.data_dict["train_sample_queries.tsv"] = {
            "name": "train_sample_queries",
            "postfix": "tsv",
            "desc": "",
            "format": ("qid", "query"),
        }
        self.data_dict["train_sample_passv2_qrels.tsv"] = {
            "name": "train_sample_passv2_qrels",
            "postfix": "tsv",
            "desc": "",
            "format": ("qid", "mark", "pid", "rating"),
        }
        self.data_dict["val_2021_53_queries.tsv"] = {
            "name": "val_2021_53_queries",
            "postfix": "tsv",
            "desc": "",
            "format": ("qid", "query"),
        }
        self.data_dict["val_2021_passage_top100.txt"] = {
            "name": "val_2021_passage_top100",
            "postfix": "txt",
            "desc": "",
            "format": ("qid", "mark", "pid", "rank", "score", "sys_id"),
        }
        self.data_dict["val_2021.qrels.pass.final.txt"] = {
            "name": "val_2021.qrels.pass.final",
            "postfix": "txt",
            "desc": "",
            "format": ("qid", "mark", "pid", "rating"),
        }
        self.data_dict["test_2022_76_queries.tsv"] = {
            "name": "test_2022_76_queries",
            "postfix": "tsv",
            "desc": "",
            "format": ("qid", "query"),
        }
        self.data_dict["test_2022_passage_top100.txt"] = {
            "name": "test_2022_passage_top100",
            "postfix": "txt",
            "desc": "",
            "format": ("qid", "mark", "pid", "rank", "score", "sys_id"),
        }
        self.data_dict["test_2022.qrels.pass.withDupes.txt"] = {
            "name": "test_2022.qrels.pass.withDupes",
            "postfix": "txt",
            "desc": "",
            "format": ("qid", "mark", "pid", "rating"),
        }

        for file in self.data_dict.keys():
            self.data_dict[file]["path"] = os.path.join(root_path, "data", file)
            self.data_dict[file]["tuple"] = namedtuple(
                rename(self.data_dict[file]["name"]), self.data_dict[file]["format"]
            )

    def yield_file(self, filename):
        assert filename in self.data_dict, "文件名错误"  # 判断文件名是否在字典中
        data = self.data_dict[filename]  # 获取文件名对应的字典
        print(f"读取文件的格式为{data['format']}")
        if data["postfix"] == "tsv":  # 判断文件后缀是否为tsv
            with open(
                data["path"], "r", encoding="utf-8", errors="ignore"
            ) as f:  # 打开文件
                reader = csv.reader(f, delimiter="\t")  # 读取文件
                for row in reader:
                    yield data["tuple"](*row)  # 以制表符分割
        elif data["postfix"] == "txt":  # 判断文件后缀是否为txt
            with open(
                data["path"], "r", encoding="utf-8", errors="ignore"
            ) as f:  # 打开文件
                for line in f:
                    yield data["tuple"](*line.split())  # 以空格分割
        else:
            raise NotImplementedError

    def read_in_memory(self):
        """
        读取文件到内存中
        """
        self.dataset = {}
        for file, d in self.data_dict.items():
            print(f"正在处理文件{file}", end=" ")
            self.dataset[d["name"]] = {}
            if file.__contains__("qrels"):
                for row in self.yield_file(file):
                    qid = getattr(row, "qid")
                    pid = getattr(row, "pid")
                    rate = int(getattr(row, "rating"))
                    if self.dataset[d["name"]].get(qid) is None:
                        self.dataset[d["name"]][qid] = {}
                    self.dataset[d["name"]][qid][pid] = rate
            elif file.__contains__("top100"):
                for row in self.yield_file(file):
                    qid = getattr(row, "qid")
                    pid = getattr(row, "pid")
                    rank = int(getattr(row, "rank"))
                    score = float(getattr(row, "score"))
                    if self.dataset[d["name"]].get(qid) is None:
                        self.dataset[d["name"]][qid] = {}
                    self.dataset[d["name"]][qid][rank] = (pid, score)
            else:
                key = "pid" if "pid" in d["format"] else "qid"
                for row in self.yield_file(file):
                    self.dataset[d["name"]][getattr(row, key)] = {
                        k: v
                        for k, v in zip(
                            d["format"],
                            row,
                        )
                        if k != key
                    }


class NDCG:
    def __init__(self, direct_gain=None):
        self.update(direct_gain)

    @property
    def length(self):
        return self.direct_gain.shape[0]

    def __call__(self, direct_gain=None, method="NDCG"):
        if direct_gain is not None:
            self.direct_gain = (
                direct_gain
                if isinstance(direct_gain, np.ndarray)
                else np.array(direct_gain)
            )
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
            res[i] = res[i - 1] + vector[i] / math.log(b, i + 1)
            # res[i] = res[i - 1] + vector[i] / log(b, i + 1)

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

    def NDCG_at_k(self, rating=None, k=10, b=2):
        """
        Normalized Discounted Cumulative Gain at k
        """
        if rating is not None:
            self.update(rating)
        return self.NDCG(b)[k - 1]
