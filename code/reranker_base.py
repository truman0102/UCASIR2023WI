import csv
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if __name__ == "__main__":
    # 读取数据
    from utils import Data

    data_processor = Data(root_path="../")
    data_processor.read_in_memory()

    # 加载模型
    path = "../model/Reranker"
    tokenizer = AutoTokenizer.from_pretrained(path)
    reranker = AutoModelForSequenceClassification.from_pretrained(path)

    # 计算验证集上的重排结果
    val_reranker_top100 = []
    for qid, top_100_passages in tqdm(
        data_processor.dataset["val_2021_passage_top100"].items()
    ):  # 遍历验证集中的每个query
        query = [data_processor.dataset["val_2021_53_queries"][qid]["query"]] * 100
        top_100_pid = [top_100_passages[i][0] for i in range(1, 101)]
        top_100_passages = [
            data_processor.dataset["collection.sampled"][pid]["passage"]
            for pid in top_100_pid
        ]  # 获取对应的passage
        inputs = tokenizer(
            query,
            top_100_passages,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )  # 将query和passage转换为模型的输入格式
        with torch.no_grad():
            scores = reranker(**inputs).logits.numpy().reshape(-1)  # 计算模型的输出
        rank = scores.argsort()[::-1]  # 对输出进行排序
        val_reranker_top100.extend(
            [
                [qid, "Q0", top_100_pid[r], i + 1, scores[r], "reranker"]
                for i, r in enumerate(rank)
            ]
        )

    with open("val_reranker_top100.txt", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(val_reranker_top100)

    # 计算测试集上的重排结果
    test_reranker_top100 = []
    for qid, top_100_passages in tqdm(
        data_processor.dataset["test_2022_passage_top100"].items()
    ):
        query = [data_processor.dataset["test_2022_76_queries"][qid]["query"]] * 100
        top_100_pid = [top_100_passages[i][0] for i in range(1, 101)]
        top_100_passages = [
            data_processor.dataset["collection.sampled"][pid]["passage"]
            for pid in top_100_pid
        ]
        inputs = tokenizer(
            query,
            top_100_passages,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            scores = reranker(**inputs).logits.numpy().reshape(-1)
        rank = scores.argsort()[::-1]
        test_reranker_top100.extend(
            [
                [qid, "Q0", top_100_pid[r], i + 1, scores[r], "reranker"]
                for i, r in enumerate(rank)
            ]
        )

    with open("test_reranker_top100.txt", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(test_reranker_top100)
