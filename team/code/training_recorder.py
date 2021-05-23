import numpy as np
import torch
from torch._six import container_abcs

from attrdict import AttrDict
from functools import partial

# 이건 첫번째 대회때 만든거 다시 재활용한거입니다!
# 코드 오류가 없기를 
class RunningLossRecorder:
    def __init__(self, number_to_record):
        self.n = number_to_record
        self.records = None
        self.cnt = 0
        self.index = 0

    def _setup(self, loss):
        if isinstance(loss, container_abcs.Mapping) or isinstance(loss, container_abcs.Sequence):
            self.records = np.empty(self.n, dtype=object)
        else:
            self.records = np.zeros(self.n)

    def setup_records(self, loss):
        self._setup(loss)

    def add(self, loss):
        if self.records is None:
            self.setup_records(loss)

        if self.cnt < self.n:
            self.cnt += 1

        self.records[self.index] = loss
        self.index += 1
        if self.index == self.n:
            self.index = 0

    def loss(self):      
        if self.cnt == 0:
            return 0  

        valid_records = self.records[:self.cnt]
        elem = valid_records[0]
        if self.records.dtype != object:
            return 'single', valid_records.sum() / self.cnt
        if isinstance(elem, container_abcs.Mapping):
            return 'dict', AttrDict({k:np.array([d[k] for d in valid_records]).sum() / self.cnt for k in elem})
        if isinstance(elem, container_abcs.Sequence):
            return 'list', list(np.array(list(zip(*valid_records))).sum(axis=-1) / self.cnt)
        raise NotImplementedError()