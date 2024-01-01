import argparse
import os
from tqdm import tqdm

from utils import logger

import torch
from torch import nn
from tensorboardX import SummaryWriter
from model import BERTReranker
from dataset import RerankDataset
from criterion import NCELoss, NCELossStrong
from criterion import calcu_ndcg, calcu_ndcg_dry, get_reranked


class Trainer(object):
    def __init__(self, args, writer=None):
        self.args = args
        self.writer = writer
        self.log_interval = args.log_interval
        self.device = self.set_devices(args.device)
        self.dataset = self.build_dataloader(args)
        self.train_loader, self.dev_loader = self.dataset.dataloader()
        self.model = self.load_model(self.device, args)
        self.criterion = self.build_criterion(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.num_epoch = args.num_epoch
        self.save_dir = args.save_dir

    def set_devices(self, device):
        if device == "cpu":
            return torch.device("cpu")
        else:
            return torch.device("cuda")

    def load_model(self, device, args):
        logger(f"Loading model from {args.pretrained_path}")
        model = BERTReranker(
            args,
            pretrained_path=args.pretrained_path,
            num_example=args.num_example,
            max_len=args.max_len,
            freeze=args.freeze
        )
        model.to(device)
        return model

    def build_dataloader(self, args):
        dataset = RerankDataset(
            args.data_dir,
            pretrained_path=args.pretrained_path,
            max_len=args.max_len,
            num_example=args.num_example,
            batch_size=args.batch_size
        )
        return dataset

    def load_test_data(self, args):
        logger("Loading test data ...")
        test_dataset = RerankDataset(
            args.data_dir,
            pretrained_path=args.pretrained_path,
            max_len=args.max_len,
            batch_size=args.batch_size,
            is_train=False
        )
        test_sets, test_items = test_dataset.test_sets, test_dataset.test_items
        return test_dataset, test_sets, test_items

    def build_criterion(self, args):
        logger("Building criterion ...")
        if args.strong_negative:
            criterion = NCELossStrong(num_negative=4)
        else:
            criterion = NCELoss()
        return criterion

    def train(self):
        for epoch in range(self.num_epoch):
            logger(f"Epoch {epoch + 1} / {self.num_epoch} started.")
            self.train_one_epoch(epoch)
            logger(f"Epoch {epoch + 1} / {self.num_epoch} finished. Validating ...")
            self.valid(epoch)
            model_path = os.path.join(self.save_dir, f"epoch{epoch + 1}")
            logger(f"Saving model to {model_path}")
            self.model.save(model_path)

    def train_one_epoch(self, epoch):
        self.model.train()
        for batch_idx, sample in enumerate(self.train_loader):
            # device
            for k, v in sample.items():
                sample[k] = v.to(self.device)
            logits, hidden_states = self.model(sample)
            logits = logits.cpu()
            hidden_states = hidden_states.cpu()
            loss = self.criterion(logits, hidden_states)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                logger(f"Epoch {epoch + 1} / {self.num_epoch}, batch {batch_idx}, loss {loss}")
                if self.writer:
                    self.writer.add_scalar("loss", loss, epoch * len(self.train_loader) + batch_idx)

    def valid(self, epoch):
        self.model.eval()
        if epoch < 0:
            logger("Begin dry run validation ...")
            val_items = self.dataset.dev_items
            ndcg = calcu_ndcg_dry(val_items)
            logger(f"Dry run validation finished, ndcg: {ndcg}")
        else:
            logger(f"Begin validation epoch {epoch + 1} ...")
            with torch.no_grad():
                ndcg = 0
                for batch_idx, sample in tqdm(enumerate(self.dev_loader), position=0):
                    item = self.dataset.dev_items[batch_idx]
                    val_bsz = 10
                    batch_num = sample["input_ids"].shape[1] // val_bsz
                    input_ids = torch.split(sample["input_ids"], val_bsz, dim=1)
                    attention_masks = torch.split(sample["attention_mask"], val_bsz, dim=1)
                    samples = [
                        {
                            "input_ids": input_ids[i],
                            "attention_mask": attention_masks[i]
                        }
                        for i in range(batch_num)
                    ]
                    logits = []
                    for sample in samples:
                        # device
                        for k, v in sample.items():
                            sample[k] = v.to(self.device)
                        _logits = self.model(sample)[0].cpu()
                        logits.append(_logits)
                        # check nan or inf
                        if torch.isnan(logits[-1]).any() or torch.isinf(logits[-1]).any():
                            logger(f"NaN or Inf detected, qid: {item.qid}, logits: {logits[-1]}")
                    logits = torch.cat(logits, dim=1).squeeze(0)
                    ndcg += calcu_ndcg(item, logits)
                logger("ndcg calculated")
                ndcg /= len(self.dev_loader)
                logger(f"Epoch {epoch + 1} ndcg: {ndcg}")
                if self.writer:
                    self.writer.add_scalar("ndcg", ndcg, epoch)

    def test(self, args):
        test_dataset, test_sets, test_items = self.load_test_data(self.args)
        test_loader = test_dataset.testloader()
        reranked_items = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, sample in tqdm(enumerate(test_loader), position=0):
                item = test_items[batch_idx]
                # device
                for k, v in sample.items():
                    sample[k] = v.to(self.device)
                logits = self.model(sample)[0].cpu()
                # check nan or inf
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logger(f"NaN or Inf detected, qid: {item.qid}, logits: {logits}")
                # to cpu
                sample["input_ids"] = sample["input_ids"].cpu()
                sample["attention_mask"] = sample["attention_mask"].cpu()
                reranked = get_reranked(item, logits)
                reranked_items.append(reranked)
        logger("reranked finished")
        # write reranked results to file
        out_path = os.path.join(args.log_dir, "reranked.trec")
        logger("writing reranked results to {} ...".format(out_path))
        with open(out_path, "w") as f:
            # qid Q0 pid rank score run_id
            run_id = f"reranker-freeze-num{args.num_example}-bsz{args.batch_size}-lr{args.lr}"
            for item in reranked_items:
                qid = item.qid
                pids = item.pids
                scores = item.scores
                for rank, (pid, score) in enumerate(zip(pids, scores)):
                    f.write(f"{qid} Q0 {pid} {rank + 1} {score} {run_id}\n")
        logger("writing reranked results to file finished")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--pretrained-path", type=str, required=True, default="bert-reranker")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-epoch", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--num-example", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze", action="store_true", default=False)
    parser.add_argument("--strong-negative", action="store_true", default=False)
    args = parser.parse_args()
    writer = SummaryWriter(args.log_dir)
    # set seed
    torch.manual_seed(args.seed)
    trainer = Trainer(args, writer)
    # trainer.valid(-1)
    logger("Begin training ...")
    trainer.train()
    logger("Training finished.")
    # trainer.test(args)


if __name__ == "__main__":
    main()