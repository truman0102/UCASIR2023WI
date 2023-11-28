import os
import csv
from collections import namedtuple, defaultdict

csv.field_size_limit(500 * 1024 * 1024)


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
            self.dataset[d["name"]] = {}
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

    def store_in_sqlite(self):
        """
        读取文件到sqlitedict中 (外存)
        """
        import sqlitedict

        for file, d in self.data_dict.items():
            print(f"正在处理文件{file}")
            db = sqlitedict.SqliteDict(
                os.path.join(self.root_path, "data", f"{d['name']}.db"),
                tablename=d["name"],
                autocommit=True,
            )
            key = "pid" if "pid" in d["format"] else "qid"
            for row in self.yield_file(file):
                db[getattr(row, key)] = {
                    k: v
                    for k, v in zip(
                        d["format"],
                        row,
                    )
                    if k != key
                }
            print(f"文件{file}处理完成")
            db.close()


if __name__ == "__main__":
    data_processor = Data(root_path="../")
    # data.store_in_sqlite()
