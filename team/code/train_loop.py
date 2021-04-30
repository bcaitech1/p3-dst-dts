import torch
import torch.nn as nn
import random
from torch.cuda.amp import autocast
from model import masked_cross_entropy_for_value

def trade_train_loop(args, model, batch, pad_token_id=0,
        loss_fnc_1=masked_cross_entropy_for_value, loss_fnc_2=nn.CrossEntropyLoss()):
    input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
        b.to(args.device) if not isinstance(b, list) else b for b in batch
    ]

    # teacher forcing
    if (
        args.teacher_forcing_ratio > 0.0
        and random.random() < args.teacher_forcing_ratio
    ):
        tf = target_ids
    else:
        tf = None

    with autocast(enabled=args.use_amp):
        all_point_outputs, all_gate_outputs = model(
            input_ids, segment_ids, input_masks, target_ids.size(-1), tf
        )

        # generation loss
        loss_1 = loss_fnc_1(
            all_point_outputs.contiguous(),
            target_ids.contiguous().view(-1),
            pad_token_id,
        )

        # gating loss
        loss_2 = loss_fnc_2(
            all_gate_outputs.contiguous().view(-1, args.n_gate),
            gating_ids.contiguous().view(-1),
        )
        loss = loss_1 + loss_2

    return loss

def submt_train_loop(args, model, batch):
    input_ids, segment_ids, input_masks, target_ids, num_turns, guids  = \
        [b.to(args.device) if not isinstance(b, list) else b for b in batch]

    # Forward
    with autocast(enabled=args.use_amp):
        loss, loss_slot, acc, acc_slot, _ = model(input_ids, segment_ids, input_masks, target_ids, 1)

    return loss
    