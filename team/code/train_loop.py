import torch
import torch.nn as nn
import random
from torch.cuda.amp import autocast
from model import masked_cross_entropy_for_value
from attrdict import AttrDict

import wandb
from typing import Union
from collections.abc import Callable

def trade_train_loop(
    args: Union[AttrDict, wandb.Config],
    model: nn.Module,
    batch: tuple,
    loss_fnc: Callable[..., AttrDict]
    ) -> AttrDict:
    """ TRADE에 학습 시킬때 사용하는 train loop

    Args:
        args (Union[AttrDict, wandb.Config]): 각종 설정 다 있는 args
        model (nn.Module): train에 사용될 모델
        batch (tuple): 현재 batch, 실제 내용은 코드보고 확인(여기에 다 넣는게 맞는지 모르겠음)
            batch 내용
            input_ids, segment_ids, input_masks, gating_ids, target_ids, guids
        loss_fnc (Callable[..., AttrDict]): loss 생성에 사용될 loss function
            (필요한 파라미터 여기에 표시하는게 맞을까?)
            파라미터들
            gen_outputs, gate_outputs, target_ids, gating_ids

    Returns:
        AttrDict: loss들
    """
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

def submt_train_loop(
    args: Union[AttrDict, wandb.Config],
    model: nn.Module,
    batch: tuple,
    loss_fnc: Callable[..., AttrDict]
    ) -> AttrDict:
    """ SUBMT에 학습 시킬때 사용하는 train loop

    Args:
        args (Union[AttrDict, wandb.Config]): 각종 설정 다 있는 args
        model (nn.Module): train에 사용될 모델
        batch (tuple): 현재 batch, 실제 내용은 코드보고 확인(여기에 다 넣는게 맞는지 모르겠음)
            batch 내용
            input_ids, segment_ids, input_masks, target_ids, num_turns, guids
        loss_fnc (Callable[..., AttrDict]): loss 생성에 사용될 loss function
            (필요한 파라미터 여기에 표시하는게 맞을까?)
            파라미터들
            outputs, target_ids

    Returns:
        AttrDict: loss들
    """
    input_ids, segment_ids, input_masks, target_ids, num_turns, guids  = \
        [b.to(args.device) if not isinstance(b, list) else b for b in batch]

    # Forward
    with autocast(enabled=args.use_amp):
        outputs, _ = model(input_ids, segment_ids, input_masks)
        loss_dict = loss_fnc(outputs, target_ids)
    
    return loss_dict

def som_dst_train_loop(
    args: Union[AttrDict, wandb.Config],
    model: nn.Module,
    batch: tuple,
    loss_fnc: Callable[..., AttrDict]
    ) -> AttrDict:
    """ SOM-DST에 학습 시킬때 사용하는 train loop

    Args:
        args (Union[AttrDict, wandb.Config]): 각종 설정 다 있는 args
        model (nn.Module): train에 사용될 모델
        batch (tuple): 현재 batch, 실제 내용은 코드보고 확인(여기에 다 넣는게 맞는지 모르겠음)
            batch 내용
            input_ids, input_masks, segment_ids, state_position_ids, op_ids,
            domain_ids, gen_ids, max_value, max_update, guids
        loss_fnc (Callable[..., AttrDict]): loss 생성에 사용될 loss function
            (필요한 파라미터 여기에 표시하는게 맞을까?)
            파라미터들
            domain_scores, state_scores, gen_scores, domain_ids,
            op_ids, gen_ids

    Returns:
        AttrDict: loss들
    """
    input_ids, input_masks, segment_ids, state_position_ids, op_ids,\
            domain_ids, gen_ids, max_value, max_update, guids = \
        [b.to(args.device) if torch.is_tensor(b) else b for b in batch]

    if (
        args.teacher_forcing_ratio > 0.0
        and random.random() < args.teacher_forcing_ratio
    ):
        tf = gen_ids
    else:
        tf = None

    # Forward
    with autocast(enabled=args.use_amp):
        domain_scores, state_scores, gen_scores = model(input_ids, segment_ids,
             state_position_ids, input_masks, max_value, op_ids, max_update, tf)
        loss_dict = loss_fnc(domain_scores, state_scores, gen_scores,
            domain_ids, op_ids, gen_ids)

    return loss_dict