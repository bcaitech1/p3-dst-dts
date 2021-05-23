import wandb
from attrdict import AttrDict

import argparse
import torch.nn as nn
from typing import Union

def setup(conf: dict, args: argparse.Namespace) -> Union[AttrDict, wandb.Config]:
    """ 주어진 args를 wandb 사용할 경우 연동하고 
        사용 안할경우 wandb.Config랑 비슷한 AttrDict으로 변환해서 리턴

    Args:
        conf (dict): yaml에서 읽은 전체 config, wandb 사용 여부 확인을 위해 사용
        args (argparse.Namespace): yaml에서 SharedPrams, 특정 ModelParams 정보
                argparse.Namespace인 이유는 모르겠음, 옛날 코드 그냥 사용해서 같기도

    Returns:
        Union[AttrDict, wandb.Config]: wandb.Config와 비슷한 형태의 args 
    """
    wandb_conf = AttrDict(conf['wandb'])
    if conf['wandb']['using']:
        
        wandb.init(project=wandb_conf.project, entity=wandb_conf.entity,
                 config=args, tags=wandb_conf.tags)
        args = wandb.config
    else:
        args = AttrDict(vars(args))
    args.wandb = wandb_conf

    return args

def watch_model(args: dict, model: nn.Module):
    """ wandb watch model wrapper 함수(wandb 사용 안할 경우 처리하게)

    Args:
        args (dict): wandb 사용여부 있는 args
        model (nn.Module): watch할 모델
    """
    if not args.wandb['using']:
        return
    
    wandb.watch(model)

def train_log(args: dict, loss_dict:dict, step:int, epoch:int):
    """ wandb에 training loss logging하기

    Args:
        args (dict): wandb 사용여부 있는 args
        loss_dict (dict): training loss가 있는 dict
        step (int): wandb에 로깅에 사용할 step(batch단위로 사용)
        epoch (int): 현재 epoch 정보
    """
    if not args.wandb['using']:
        return 

    if len(loss_dict) > 5: # wandb에서 그래프 너무 많이 생성되는거 방지
        loss_dict = {f'train_loss': loss_dict['loss']}
    else:
        loss_dict = {f'train_{k}':v for k, v in loss_dict.items()}
    loss_dict['epoch'] = epoch
    wandb.log(loss_dict, step=step)

def val_log(args:dict, eval_result:dict, loss_dict:dict, step:int, epoch:int):
    """ wandb에 validation 관련 logging하기

    Args:
        args (dict): wandb 사용여부 있는 args
        eval_result (dict): evaluation 결과(loss말고 다른 metrics)
        loss_dict (dict): training loss가 있는 dict
        step (int): wandb에 로깅에 사용할 step(batch단위로 사용)
        epoch (int): 현재 epoch 정보
    """
    if not args.wandb['using']:
        return 

    eval_result = {f'val_{k}':v for k, v in eval_result.items()}
    if len(loss_dict) > 5: # wandb에서 그래프 너무 많이 생성되는거 방지
        loss_dict = {f'val_loss': loss_dict['loss']}
    else:
        loss_dict = {f'val_{k}':v for k, v in loss_dict.items()}
    loss_dict['epoch'] = epoch
    wandb.log(eval_result, step=step)
    wandb.log(loss_dict, step=step)

def best_jga_log(args:dict, best_score: float):
    """ wandb에 현재 best jga logging하기

    Args:
        args (dict):  wandb 사용여부 있는 args
        best_score (float): 최고 Joint Goal Accuracy
    """
    if not args.wandb['using']:
        return 

    wandb.run.summary['best_val_jga'] = best_score
    