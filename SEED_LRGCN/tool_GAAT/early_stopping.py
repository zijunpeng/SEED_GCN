import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, subject_name, val_acc, model, epoch):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(subject_name, val_acc, model, epoch)
        elif score <= self.best_score + self.delta:
            pass

        else:
            self.best_score = score
            self.save_checkpoint(subject_name, val_acc, model, epoch)
            self.counter = 0

    def save_checkpoint(self, subject_name, val_acc, model, epoch):
        '''Saves model when validation acc increase.'''
        if self.verbose:
            self.trace_func(f'Validation acc increased ({self.val_acc_max:.6f} --> {val_acc:.6f}) in epoch ({epoch}).  Saving model ...')
        model_save_path = self.path + subject_name + ".pt"
        torch.save(model.state_dict(), model_save_path)
        self.val_acc_max = val_acc