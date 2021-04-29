import torch.nn as nn

from functools import partial
from attrdict import AttrDict
from model import masked_cross_entropy_for_value

class TRADE_Loss:
    def __init__(self, pad_token_id):
        self.loss_fnc_1 = partial(masked_cross_entropy_for_value, pad_idx=pad_token_id) # generation
        self.loss_fnc_2 = nn.CrossEntropyLoss()  # gating

    def __call__(self, batch, outputs):
        all_point_outputs = outputs.all_point_outputs
        all_gate_outputs = outputs.all_gate_outputs
        target_ids = batch.target_ids
        gating_ids = batch.gating_ids

        loss_1 = self.loss_fnc_1(
            all_point_outputs.contiguous(),
            target_ids.contiguous().view(-1),
        )

        loss_2 = self.loss_fnc_2(
            all_gate_outputs.contiguous().view(-1, all_gate_outputs.size(-1)),
            gating_ids.contiguous().view(-1),
        )
        loss = loss_1 + loss_2
        return AttrDict(
            loss=loss,
            loss_generation=loss_1,
            loss_gating=loss_2,
        )