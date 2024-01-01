from utils import logger

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification
from typing import List

class BERTReranker(nn.Module):
    def __init__(
        self,
        args,
        pretrained_path: str,
        num_example: int = 5,
        max_len: int = 512,
        freeze: bool = False,
        replace_classifier: bool = True,
    ):
        super().__init__()
        self.args = args
        self.num_example = num_example
        self.max_len = max_len
        self.bert = AutoModelForSequenceClassification.from_pretrained(pretrained_path)
        if freeze:
            self.freeze_bert()
        if replace_classifier:
            self.replace_classifier()

    def replace_classifier(self):
        classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        # init
        classifier = self.init_params(classifier)
        self.bert.classifier = classifier
    
    def init_params(self, classifier: nn.Module):
        for param in classifier.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)
        return classifier

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, net_inputs):
        # net_inputs: [B, N, L] -> [B*N, L]
        sample_num = net_inputs["input_ids"].shape[1]
        net_inputs["input_ids"] = net_inputs["input_ids"].view(-1, self.max_len)
        net_inputs["attention_mask"] = net_inputs["attention_mask"].view(-1, self.max_len)
        out = self.bert(**net_inputs, output_hidden_states=True)
        logits, hidden_states = out["logits"], out["hidden_states"][-1]
        # logits: [B*N, 1] -> [B, N]
        logits = logits.view(-1, sample_num)
        # hidden_states: [B*N, L, H] -> [B, N, L, H]
        hidden_states = hidden_states.view(-1, sample_num, self.max_len, hidden_states.shape[-1])
        return logits, hidden_states

    def save(self, path: str):
        logger(f"Saving model to {path}")
        self.bert.save_pretrained(path)
