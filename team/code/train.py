import argparse
import json
import os
import random
import yaml
import pprint
import sys
import copy
from copy import deepcopy
import pickle
from attrdict import AttrDict
from collections import Counter,defaultdict

import torch
import torch.nn as nn

from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm

from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

from data_utils import (WOSDataset, load_dataset,
                        seed_everything)
from evaluation import _evaluation
from eda import *
from train_loop import trade_train_loop, submt_train_loop, som_dst_train_loop
from inference import trade_inference, sumbt_inference, som_dst_inference

from prepare import get_data, get_stuff, get_model, set_directory
from losses import Trade_Loss, SUBMT_Loss, SOM_DST_Loss

import wandb_stuff
import parser_maker
from training_recorder import RunningLossRecorder
from aug_utils import *

""" 이 파일은 단일로 사용시 모델 학습을 할 수 있고 
    다른 파일에서 사용 가능한 train함수가 있음

모델 학습은 config 파일을 사용해서 학습함
그냥 사용하면 정해진 default config path를 사용하고
-c, --config로 특정 config path에 있는 파일 사용 가능
"""

def train(config_root: str) -> str:
    """ config를 받아서 해당 내용에 맞게 학습시키는 함수

    Args:
        config_root (str): config 파일 이름, 형식은 yaml이어야됨

    Raises:
        NotImplementedError: 지원되지 않는 모델로 학습시킬려고 할때 뜸

    Returns:
        str: 학습된 모델이 저장된 폴더 이름 리턴, inference랑 연동을 위해서인듯
    """

    # Config 가져오고 기본 설정하는 부분
    print(f'Using config: {config_root}')

    # conf는 SharedParams, 모든 모델 ModleParmarms,
    #  Wandb 사용여부 wandb로 이루어짐
    with open(config_root) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    print(f"Currnet Using Model : {conf['ModelName']}")

    model_name = conf['ModelName']

    # args_dict는 yaml에서 정의된 SharedParams랑 특정 ModelParams만 사용
    args_dict = deepcopy(conf['SharedPrams'])
    args_dict.update(conf[model_name])

    args = argparse.Namespace(**args_dict)

    args.ModelName = conf['ModelName']
    basic_args = args
    args = wandb_stuff.setup(conf, args)
    print(f'Get Args {model_name}')
    pprint.pprint({k:v for k, v in args.items()})
    print()

    args.device = torch.device(args.device_pref if torch.cuda.is_available() else "cpu")

    # random seed 고정
    seed_everything(args.random_seed)

    # 호환성 위해서 추가
    if 'train_from_trained' not in args:
        args.train_from_trained = None
    if 'use_zero_segment_id' not in args:
        args.use_zero_segment_id = False
    if 'filter_old_data' not in args:
        args.filter_old_data = False
    if 'use_val_idxs' not in args:
        args.use_val_idxs = None

    # 이미 학습된 모델에서 가져올 경우
    if args.train_from_trained is not None:
        trained_config = json.load(open(f'{args.train_from_trained}/exp_config.json'))
        trained_config = AttrDict(trained_config)
        print(f'Changing ModelName, preprocessor, model_class to {args.train_from_trained} config')
        args.ModelName = trained_config.ModelName
        args.preprocessor = trained_config.preprocessor
        args.model_class = trained_config.model_class
        args.model_name_or_path = trained_config.model_name_or_path

    # Data Loading
    data, slot_meta, ontology = get_data(args)

    # 특정 다이얼로그를 validation으로 사용하고 싶을 때
    if args.train_from_trained is not None and args.use_trained_val_idxs:
        given_val_idxs = json.load(open(f'{args.train_from_trained}/dev_idxs.json', 'r'))
    elif args.use_val_idxs is not None:
        print(f'loading only val-idxs from {args.use_val_idxs}')
        given_val_idxs = json.load(open(f'{args.use_val_idxs}/dev_idxs.json', 'r'))
    else:
        given_val_idxs = None

    # data train, val로 분리
    if args.ModelName == 'SOM_DST':
        train_data, dev_data, dev_labels, dev_idxs = load_dataset(data, given_dev_idx=given_val_idxs,
            filter_old_data=args.filter_old_data, dev_has_label=True)
    else:
        train_data, dev_data, dev_labels, dev_idxs = load_dataset(data, given_dev_idx=given_val_idxs,
            filter_old_data=args.filter_old_data)
    
    # data -> examples -> features로 변환해서 얻기
    # 연관된 tokenizer, processor, model 얻기
    if args.train_from_trained is not None:
        tokenizer, processor, train_features, dev_features = get_stuff(trained_config,
            train_data, dev_data, slot_meta, ontology)
        model =  get_model(trained_config, tokenizer, ontology, slot_meta)
    else:
        tokenizer, processor, train_features, dev_features = get_stuff(args,
            train_data, dev_data, slot_meta, ontology)
        model =  get_model(args, tokenizer, ontology, slot_meta)

    if args.train_from_trained is not None:
        ckpt = torch.load(f"{args.train_from_trained}/model-best.bin", map_location="cpu")
        model.load_state_dict(ckpt)
    
    pbar = tqdm(desc=f'Moving model to {args.device} -- waiting...', bar_format='{desc} -> {elapsed}')
    model.to(args.device)
    pbar.set_description(f'Moving model to {args.device} -- DONE')  
    pbar.close()

    wandb_stuff.watch_model(args, model)

    train_data = WOSDataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=processor.collate_fn,
    )

    dev_data = WOSDataset(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.eval_batch_size,
        sampler=dev_sampler,
        collate_fn=processor.collate_fn,
    )

    print()
    print("# train:", len(train_data))
    print("# dev:", len(dev_data))
    print()
    print('##### START TRAINING #####')

    # Optimizer 및 Scheduler 선언
    ### TODO: args.no_decay로 하면 어떨까?(이건 결국 안함)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # 실제 학습하는 부분
    n_epochs = args.num_train_epochs
    t_total = len(train_loader) * n_epochs
    warmup_steps = int(t_total * args.warmup_ratio)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    if not os.path.exists(args.train_result_dir):
        os.mkdir(args.train_result_dir)
    

    # train의 result들을 task 이름 폴더에 저장
    task_dir = f'{args.train_result_dir}/{args.task_name}'

    if os.path.exists(f'{args.train_result_dir}/{args.task_name}'):
        i = 1
        while os.path.exists(f'{task_dir}_{i}'):
            i += 1
        
        task_dir = f'{task_dir}_{i}'

    os.mkdir(f'{task_dir}')

    set_directory(f'{task_dir}/graph')

    print('\n')
    print(f'Current result dir : {task_dir}')
    print('\n')

    # 현재 세팅 저장
    args_save = {k:v for k, v in args.items() if k in basic_args}
    json.dump(
        args_save,
        open(f"{task_dir}/exp_config.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    json.dump(
        slot_meta,
        open(f"{task_dir}/slot_meta.json", "w"),
        indent=2,
        ensure_ascii=False,
    )

    json.dump(
        ontology,
        open(f"{task_dir}/edit_ontology_metro.json", "w"),
        indent=2,
        ensure_ascii=False,
    )

    json.dump(
        dev_idxs,
        open(f'{task_dir}/dev_idxs.json', 'w'),
        indent=2,
        ensure_ascii=False,
    )

    # 각각 모델마다 train할 때 다른 형식(batch에서 나온거나, teacher_forcing)인거 같아서
    # 각각 모델마다 각각 train, inference loop 만듬
    if args.ModelName == 'TRADE':
        train_loop = trade_train_loop
        loss_fnc = Trade_Loss(tokenizer.pad_token_id, args.n_gate)
        train_loop_kwargs = AttrDict(loss_fnc=loss_fnc)
        inference_func = trade_inference
    elif args.ModelName == 'SUMBT':
        train_loop = submt_train_loop
        loss_fnc = SUBMT_Loss()
        train_loop_kwargs = AttrDict(loss_fnc=loss_fnc)
        inference_func = sumbt_inference
    elif args.ModelName == 'SOM_DST':
        train_loop = som_dst_train_loop
        loss_fnc = SOM_DST_Loss(tokenizer.pad_token_id)
        train_loop_kwargs = AttrDict(loss_fnc=loss_fnc)
        inference_func = som_dst_inference
    else:
        raise NotImplementedError()

    scaler = GradScaler(enabled=args.use_amp)
    best_score, best_checkpoint = 0, 0
    total_step = 0

    loss_recorder = RunningLossRecorder(args.train_running_loss_len)

    wrong_list=[] #에폭별로 wrong_answer를 담아둘 배열
    correct_list=[] #wrong_list의 에폭별 오답률을 위해 correct_answer를 담아둘 배열
    guid_compare_dict_list=[]
    for epoch in range(n_epochs):
        model.train()
        pbar2 = tqdm(enumerate(train_loader), total=len(train_loader), file=sys.stdout)
        loss_showing = 'none'
        step = 0
        pbar2.set_description(f'[{epoch}/{n_epochs}] {loss_showing}')
        for step, batch in pbar2:
            optimizer.zero_grad()

            loss_dict = train_loop(args, model, batch, **train_loop_kwargs)
            loss = loss_dict.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            
            scale = scaler.get_scale()
            scaler.update()
            step_scheduler = scaler.get_scale() == scale
            
            if step_scheduler:
                scheduler.step()

            cpu_loss_dict = {k:v.item() for k, v in loss_dict.items()}
            loss_recorder.add(cpu_loss_dict)

            # log step
            if step != 0 and (step % args.train_log_step == 0 or step == len(train_loader) - 1):
                log_loss_dict = loss_recorder.loss()[1]
                if len(log_loss_dict) >= 5:
                    loss_showing = f'loss: {log_loss_dict.loss:.4f}'
                else:
                    loss_showing = ' '.join([f'{k}: {v:.4f}' for k, v in log_loss_dict.items()])
                pbar2.set_description(f'[{epoch}/{n_epochs}] {loss_showing}')
                # print(
                    # f"\n[{epoch}/{n_epochs}] [{step}/{len(train_loader)}] {loss_showing}",
                    # end=''
                # )   
                wandb_stuff.train_log(args, log_loss_dict, total_step, epoch + step / len(train_loader))

            total_step += 1
        pbar2.close()
        # validation 부분
        val_predictions, val_loss_dict = inference_func(model, dev_loader, processor, args.device, args.use_amp, 
                loss_fnc=loss_fnc)
        # 현재 에폭에서 eval_result 외에도 틀린 예측값, ground truth값을 뽑아낸다
        eval_result,now_wrong_list,now_correct_list,guid_compare_dict = _evaluation(val_predictions, dev_labels, slot_meta)
        #eda
        domain_counter,slot_counter,value_counter=get_Domain_Slot_Value_distribution_counter(Counter(now_wrong_list))
        o_domain_counter,o_slot_counter,o_value_counter=get_Domain_Slot_Value_distribution_counter(Counter(now_correct_list))
        draw_EDA('domain',domain_counter,o_domain_counter, epoch)
        draw_EDA('slot',slot_counter,o_slot_counter, epoch)
        draw_EDA('value',value_counter,o_value_counter, epoch)
        draw_WrongDomslot(guid_compare_dict, epoch)
        wrong_list.append(now_wrong_list)
        correct_list.append(now_correct_list)
        guid_compare_dict_list.append(guid_compare_dict)

        # 화면 출력용
        print('---------Validation-----------')
        for k, v in eval_result.items():
            print(f"{k}: {v:.4f}")
        if len(loss_dict) >= 5:
            loss_showing = f'loss: {loss_dict.loss:.4f}'
        else:
            loss_showing = ' '.join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
        print(loss_showing)
        print('------------------------------')

        # wandb logging
        wandb_stuff.val_log(args, eval_result, loss_dict, total_step, epoch+1)

        if best_score < eval_result['joint_goal_accuracy']:
            print("Update Best checkpoint!",)
            best_score = eval_result['joint_goal_accuracy']
            wandb_stuff.best_jga_log(args, best_score)
            best_checkpoint = epoch
            
            if args.save_model:
                torch.save(model.state_dict(), f"{task_dir}/model-best.bin")

        # if epoch % 5 == 4:
        #     print(f'saving to {args.train_result_dir}/model-{epoch}.bin"')
        #     torch.save(model.state_dict(), f"{args.train_result_dir}/model-{epoch}.bin")
    print(f"Best checkpoint: {best_checkpoint}",)
    # draw_WrongTrend(wrong_list)

    # save
    with open(f"{task_dir}/guid_compare_dict_list.pickle", 'wb') as f:
        pickle.dump(guid_compare_dict_list, f, pickle.HIGHEST_PROTOCOL)
    with open(f"{task_dir}/dev_idxs.pickle", 'wb') as f:
        pickle.dump(dev_idxs, f, pickle.HIGHEST_PROTOCOL)

    # load
    # with open('dev_idx.pickle', 'rb') as f:
    #     di_load = pickle.load(f)
    # with open('guid_compare_dict.pickle', 'rb') as f:
    #     gcd_load = pickle.load(f)
    return task_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('-c', '--config', 
                        type=str,
                        help="Get config file following root",
                        default='/opt/ml/project/team/code/conf2.yml')

    # wandb sweep 연동을 위해서 추가함
    # (코드 합치다가 어떻게 되서 wandb sweep 연동 됬는지는 모름
    # 어차피 학습 시간 길어서 잘 사용 안하기도 하고
    parser = parser_maker.update_parser(parser)
    config_args = parser.parse_args()

    config_root = config_args.config
    print(f'Using config: {config_root}')
    
    train(config_root)
    

