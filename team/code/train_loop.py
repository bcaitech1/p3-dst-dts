import torch.nn as nn
import random
from torch.cuda.amp import autocast
from model import masked_cross_entropy_for_value
from attrdict import AttrDict

def trade_train_loop(args, model, batch, loss_fnc):
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

        loss_dict = loss_fnc(all_point_outputs, all_gate_outputs, 
                target_ids, gating_ids)

    return loss_dict

def submt_train_loop(args, model, batch, loss_fnc):
    input_ids, segment_ids, input_masks, target_ids, num_turns, guids  = \
        [b.to(args.device) if not isinstance(b, list) else b for b in batch]

    # Forward
    with autocast(enabled=args.use_amp):
        outputs, _ = model(input_ids, segment_ids, input_masks)
        loss_dict = loss_fnc(outputs, target_ids)
    
    return loss_dict