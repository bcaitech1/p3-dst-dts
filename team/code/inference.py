import argparse
import os
import json
import sys
import yaml
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
import copy
from data_utils import WOSDataset
from prepare import get_data, get_stuff, get_model
import parser_maker

from training_recorder import RunningLossRecorder

from typing import Union
from collections.abc import Callable

""" 특정 모델로 predictions 하는 스크립트
-t, --task_dir 로 원하는 모델이 있는 폴더 설정
결과는 해당 폴더에 pred.csv로 저장
"""

def postprocess_state(state: list) -> list:
    """ 꿔=바로우, 12:23 같은 값들 처리

    Args:
        state (list): 기존 states

    Returns:
        list: 수정된 states
    """
    for i, s in enumerate(state):
        s = s.replace(" : ", ":")ㄷ
        s = s.replace(" & ", "&")
        s = s.replace(" = ", "=")
        s = s.replace("( ", "(")
        s = s.replace(" )", ")")
        state[i] = s.replace(" , ", ", ")
    return state


def trade_inference(model, eval_loader, processor, device, use_amp=False, 
        loss_fnc=None) -> Union[dict, tuple[dict, dict]]:
    """ 주어진 TRADE 모델을 주어진 validation loader로 inference함
    loss fnc 제공되면 loss dict(다양한 loss가 있는 dict)도 리턴함

    Args:
        model: inference용 모델
        eval_loader: validation용 data loader
        processor: 현재 모델이 사용하는 preprocessor
        device: torch device
        use_amp (bool, optional): amp 사용 여부. Defaults to False.
        loss_fnc (Callable, optional): 사용할 loss 함수. Defaults to None.

    Returns:
        Union[dict, tuple[dict, dict]]: predicted states | predicted stateds, loss dict
    """
    model.eval()
    predictions = {}
    pbar = tqdm(eval_loader, total=len(eval_loader), file=sys.stdout)
    loss_recorder = RunningLossRecorder(len(eval_loader))
    for batch in pbar:
        input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
            b.to(device) if not isinstance(b, list) else b for b in batch
        ]

        with torch.no_grad():
            with autocast(enabled=use_amp):
                o, g = model(input_ids, segment_ids, input_masks, 9)

            if loss_fnc is not None:
                with autocast(enabled=use_amp):
                    loss_dict = loss_fnc(o, g, target_ids, gating_ids)

                cpu_loss_dict = {k:v.item() for k, v in loss_dict.items()}
                loss_recorder.add(cpu_loss_dict)

            _, generated_ids = o.max(-1)
            _, gated_ids = g.max(-1)


        for guid, gate, gen in zip(guids, gated_ids.tolist(), generated_ids.tolist()):
            prediction = processor.recover_state(gate, gen)
            prediction = postprocess_state(prediction)
            predictions[guid] = prediction
    pbar.close()

    if loss_fnc is not None:
        return predictions, loss_recorder.loss()[1]
    else:
        return predictions

def sumbt_inference(model, eval_loader, processor, device, use_amp=False,
        loss_fnc=None) -> Union[dict, tuple[dict, dict]]:
    """ 주어진 SUMBT 모델을 주어진 validation loader로 inference함
    loss fnc 제공되면 loss dict(다양한 loss가 있는 dict)도 리턴함

    Args:
        model: inference용 모델
        eval_loader: validation용 data loader
        processor: 현재 모델이 사용하는 preprocessor
        device: torch device
        use_amp (bool, optional): amp 사용 여부. Defaults to False.
        loss_fnc (Callable, optional): 사용할 loss 함수. Defaults to None.

    Returns:
        Union[dict, tuple[dict, dict]]: predicted states | predicted stateds, loss dict
    """
    model.eval()
    predictions = {}
    
    pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), file=sys.stdout)
    loss_recorder = RunningLossRecorder(len(eval_loader))
    for step, batch in pbar:
        input_ids, segment_ids, input_masks, target_ids, num_turns, guids  = \
            [b.to(device) if not isinstance(b, list) else b for b in batch]
        
        with torch.no_grad():
            with autocast(enabled=use_amp):
                outputs, pred_slots = model(input_ids, segment_ids, input_masks, None)

            if loss_fnc is not None:
                with autocast(enabled=use_amp):
                    loss_dict = loss_fnc(outputs, target_ids)

                cpu_loss_dict = {k:v.item() for k, v in loss_dict.items()}
                loss_recorder.add(cpu_loss_dict)
            
        pred_slots = pred_slots.detach().cpu()
        for guid, num_turn, p_slot in zip(guids, num_turns, pred_slots):
            pred_states = processor.recover_state(p_slot, num_turn)
            for t_idx, pred_state in enumerate(pred_states):
                predictions[f'{guid}-{t_idx}'] = pred_state
    pbar.close()

    if loss_fnc is not None:
        return predictions, loss_recorder.loss()[1]
    else:
        return predictions

def som_dst_inference(model, eval_loader, processor, device, use_amp=False,
        loss_fnc=None) -> Union[dict, tuple[dict, dict]]:
    """ 주어진 SOM-DST 모델을 주어진 validation loader로 inference함
    loss fnc 제공되면 loss dict(다양한 loss가 있는 dict)도 리턴함

    Args:
        model: inference용 모델
        eval_loader: validation용 data loader
        processor: 현재 모델이 사용하는 preprocessor
        device: torch device
        use_amp (bool, optional): amp 사용 여부. Defaults to False.
        loss_fnc (Callable, optional): 사용할 loss 함수. Defaults to None.

    Returns:
        Union[dict, tuple[dict, dict]]: predicted states | predicted stateds, loss dict
    """

    model.eval()
    predictions = {}
    
    pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), file=sys.stdout)
    loss_recorder = RunningLossRecorder(len(eval_loader))
    for step, batch in pbar:
        input_ids, input_masks, segment_ids, state_position_ids, op_ids,\
            domain_ids, gen_ids, max_value, max_update, guids = \
        [b.to(args.device) if not isinstance(b, list) else b for b in batch]
        
        with torch.no_grad():
            with autocast(enabled=use_amp):
                domain_scores, state_scores, gen_scores = model(input_ids, segment_ids,
                    state_position_ids, input_masks, max_value, op_ids)
                

            if loss_fnc is not None:
                with autocast(enabled=use_amp):
                    loss_dict = loss_fnc(domain_scores, state_scores, gen_scores,
                        domain_ids, op_ids, gen_ids)

                cpu_loss_dict = {k:v.item() for k, v in loss_dict.items()}
                loss_recorder.add(cpu_loss_dict)
            
        pred_slots = pred_slots.detach().cpu()
        for guid, num_turn, p_slot in zip(guids, num_turns, pred_slots):
            pred_states = processor.recover_state(p_slot, num_turn)
            for t_idx, pred_state in enumerate(pred_states):
                predictions[f'{guid}-{t_idx}'] = pred_state
    pbar.close()

    if loss_fnc is not None:
        return predictions, loss_recorder.loss()[1]
    else:
        return predictions


def inference(task_dir:str=None):
    """ 특정 모델이 있는 폴더를 주면 해당 모델로 prediction하는 함수

    Args:
        task_dir (str, optional): 모델이 있는 폴더. Defaults to None.

    Raises:
        NotImplementedError: 지원하지 않는 모델 설정시
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # conf=dict()
    # with open(config_root) as f:
    #     conf = yaml.load(f, Loader=yaml.FullLoader)

    # 필요한 데이터, slot meta, onotology 가져옴
    eval_data = json.load(open(f"/opt/ml/input/data/eval_dataset/eval_dials.json", "r"))

    config = json.load(open(f"{task_dir}/exp_config.json", "r"))
    
    print(config)
    config = argparse.Namespace(**config)
    _, slot_meta, ontology = get_data(config)

    config.device = device

    # tokenizer, preprocessor + data -> examples -> features로 변경
    tokenizer, processor, eval_features, _ = get_stuff(config,
                 eval_data, None, slot_meta, ontology)

    # 필요한 dataset, data loader 생성
    eval_data = WOSDataset(eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_loader = DataLoader(
        eval_data,
        batch_size=8,
        sampler=eval_sampler,
        collate_fn=processor.collate_fn,
    )
    print("# eval:", len(eval_data))

    print(slot_meta)

    # 모델 가져오는 부분
    model =  get_model(config, tokenizer, ontology, slot_meta)


    ckpt = torch.load(f"{task_dir}/model-best.bin", map_location="cpu")
    # ckpt = torch.load("/opt/ml/gyujins_file/model-best.bin", map_location="cpu")

    model.load_state_dict(ckpt)
    model.to(device)
    print("Model is loaded")

    # prediction하는 부분
    if config.ModelName == 'TRADE':
        inference_func = trade_inference
    elif config.ModelName == 'SUMBT':
        inference_func = sumbt_inference
    else:
        raise NotImplementedError()

    predictions = inference_func(model, eval_loader, processor, device, config.use_amp)
    
    # 결과 저장 부분
    json.dump(
        predictions,
        open(f"{task_dir}/pred.csv", "w"),
        indent=2,
        ensure_ascii=False,
    )

    print(f"Inference finished!\n output file is : {task_dir}/pred.csv")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', 
                        type=str,
                        help="Get config file following root",
                        default='./conf.yml')
    parser.add_argument('-t', '--task_dir', 
                        type=str,
                        help="Get task_dir",
                        default=None)                    
    parser = parser_maker.update_parser(parser)

    config_args = parser.parse_args()
    # config_root = config_args.config
    # print(f'Using config: {config_root}')
    inference(config_args.task_dir)