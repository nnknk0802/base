import os, sys
import random
import itertools
from functools import namedtuple

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from src.network import load_net
from src.dataset import load_dataloaders
from src.utils import get_logger, iterate, record_score, dict_to_tuple

class Trainer():
    def __init__(self, c):
        self.c = c
        torch.backends.cudnn.benchmark = True

        self.logger, self.dir_path = get_logger(self.c['model_name'])
        self.score_path = os.path.join(self.dir_path, 'score.csv')
        self.logger.info(self.c)

    def fit(self):
        for c, param in iterate(self.c):
            self.logger.info(param)
            self.run(dict_to_tuple(c))

    def run(self, c):
        random.seed(c.seed)
        torch.manual_seed(c.seed)

        net = load_net().cuda()
        self.net = torch.nn.DataParallel(net)
        self.datasets, self.dataloaders = load_dataloaders(batch_size=c.bs)

        self.optimizer = optim.Adam(net.parameters(), lr=c.lr)
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

        for epoch, phase in itertools.product(range(1, c.n_epoch+1), ['train', 'val']):
            loss, acc = self.epoch(phase)
            self.logger.debug(f'{epoch}, {phase}, loss: {loss:.3f}, acc: {acc:.3f}')

            d = {'epoch': epoch, 'phase': phase, 'loss': loss, 'acc': acc}
            record_score(d, self.score_path)

    def epoch(self, phase):
        preds, labels, total_loss = [], [], 0
        self.net.train() if phase == 'train' else self.net.eval()

        for inputs_, labels_ in tqdm(self.dataloaders[phase]):
            inputs_ = inputs_.cuda()
            labels_ = labels_.cuda()
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs_ = self.net(inputs_)
                onehot = torch.eye(10)[labels_].cuda()
                loss = self.criterion(outputs_, onehot)

                if phase == 'train':
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

            preds += [torch.argmax(outputs_, dim=1).detach().cpu().numpy()]
            labels += [labels_.detach().cpu().numpy()]
            total_loss += float(loss.detach().cpu().numpy()) * len(inputs_)

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        total_loss /= len(preds)
        acc = sum(preds == labels) / len(preds)

        return total_loss, acc