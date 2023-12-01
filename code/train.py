import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import warnings

warnings.filterwarnings("ignore")


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001, cuda=None):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.cuda = torch.cuda.is_available() if cuda is None else cuda
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                f"Validation score improved ({self.val_score:.4f} --> {epoch_score:.4f}). Saving model!"
            )
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score
