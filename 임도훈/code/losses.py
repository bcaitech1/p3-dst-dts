import torch.nn as nn
from model import masked_cross_entropy_for_value

from attrdict import AttrDict

class Trade_Loss:
    def __init__(self, pad_token_id, n_gate):
        self.loss_fnc_1 = masked_cross_entropy_for_value
        self.loss_fnc_2 = nn.CrossEntropyLoss()
        self.pad_token_id = pad_token_id
        self.n_gate = n_gate

    def __call__(self, all_point_outputs, all_gate_outputs, target_ids, gating_ids):
        loss_1 = self.loss_fnc_1(
            all_point_outputs.contiguous(),
            target_ids.contiguous().view(-1),
            self.pad_token_id
        )

        loss_2 = self.loss_fnc_2(
            all_gate_outputs.contiguous().view(-1, self.n_gate),
            gating_ids.contiguous().view(-1),
        )

        loss = loss_1 + loss_2

        return AttrDict(
            loss = loss,
            gen_loss = loss_1,
            gating_loss = loss_2,
        )

class SUBMT_Loss:
    def __init__(self):
        self.nll = nn.CrossEntropyLoss(ignore_index=-1)

    def __call__(self, outputs, labels):
        ds, ts, js = labels.shape
        loss_slot = []
        loss = 0
        for i in range(js):
            _dist = outputs[i]
            label = labels[:, :, i]
            _loss = self.nll(_dist.view(ds * ts, -1), label.view(-1))
            loss_slot.append(_loss)
            loss += _loss

        ret = AttrDict(loss=loss)
        for i, v in enumerate(loss_slot):
            ret[f'loss_slot_{i}'] = v
        return ret