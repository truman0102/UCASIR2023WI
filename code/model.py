import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Rerank(nn.Module):
    def __init__(self, pretrained_path, max_length=512, replace_classifer=False):
        super(Rerank, self).__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_path)
        if replace_classifer:
            self.model.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.GELU(),
                nn.Linear(256, 1),
            )
        self.freeze()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs

    def encode(self, query, passage):
        if isinstance(passage, str):
            inputs = self.tokenizer(
                query, passage, return_tensors="pt", padding=True, truncation=True
            )
            return inputs["input_ids"], inputs["attention_mask"]
        elif isinstance(passage, (tuple, list, set)):
            return self.encode_batch(query, passage)

    def encode_batch(self, query, passages):
        inputs_ids_batch = torch.empty(
            (len(passages), self.max_length), dtype=torch.long
        )
        attention_mask_batch = torch.empty(
            (len(passages), self.max_length), dtype=torch.long
        )
        for i, passage in enumerate(passages):
            inputs = self.tokenizer(
                query,
                passage,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            inputs_ids_batch[i] = inputs["input_ids"][0]
            attention_mask_batch[i] = inputs["attention_mask"][0]
        return inputs_ids_batch, attention_mask_batch

    def score(self, query, passage):
        input_ids, attention_mask = self.encode(query, passage)
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask=attention_mask).logits
        return logits

    def freeze(self):
        for _, param in self.model.bert.named_parameters():
            param.requires_grad = False


if __name__ == "__main__":
    path = "model/Reranker"
    model = Rerank(path, replace_classifer=True)
    query = "What is the capital of China?"
    passage = "Beijing is the capital of China."
    passages = [
        "Beijing is the capital of China.",
        "Shanghai is the largest city in China.",
    ]
    score = model.score(query, passage)
    print(score)
    score = model.score(query, passages)
    print(score)
