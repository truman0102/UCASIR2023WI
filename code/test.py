import os
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if __name__ == "__main__":
    # 读取数据
    from utils import Data

    data_processor = Data(root_path="../")
    data_processor.read_in_memory()

    # 加载模型
    model_name = "freeze-num5-bsz4-lr1e-5"
    epoch = 3
    path = f"../ckpt/{model_name}/epoch{epoch}"
    tokenizer = AutoTokenizer.from_pretrained(path)
    reranker = AutoModelForSequenceClassification.from_pretrained(path)
    # classifier可能没load上来，需要手动load
    reranker.classifier = nn.Sequential(
        nn.Linear(reranker.config.hidden_size, 256),
        nn.GELU(),
        nn.Linear(256, 1),
    )
    reranker.load_state_dict(torch.load(f"{path}/pytorch_model.bin"))
    reranker.eval()

    # check cuda available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reranker.to(device)

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
        )  # 将query和passage转换为模型的输入格式
        # device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = reranker(**inputs)
            logits = out.logits
        # cpu
        inputs = {k: v.cpu() for k, v in inputs.items()}
        logits = logits.cpu()
        scores = logits.numpy().reshape(-1)  # 计算模型的输出
        rank = scores.argsort()[::-1]  # 对输出进行排序
        test_reranker_top100.extend(
            [
                [qid, "Q0", top_100_pid[r], i + 1, scores[r], "reranker"]
                for i, r in enumerate(rank)
            ]
        )

    out_dir = "../res"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    with open(f"{out_dir}/{model_name}-epoch{epoch}.trec", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(test_reranker_top100)
