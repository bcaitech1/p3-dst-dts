import argparse
import json
import os
import random
import yaml
import pprint
import sys
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
from eda import get_Domain_Slot_Value_distribution_counter, draw_EDA,draw_WrongTrend
from train_loop import trade_train_loop, submt_train_loop
from inference import trade_inference, sumbt_inference 

from prepare import get_stuff, get_model
from losses import Trade_Loss, SUBMT_Loss
import wandb_stuff
import parser_maker
from training_recorder import RunningLossRecorder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('-c', '--config', 
                        type=str,
                        help="Get config file following root",
                        default='/opt/ml/p3-dst-dts/dohoon/code/conf copy.yml')
                        
    parser = parser_maker.update_parser(parser)

    config_args = parser.parse_args()
    config_name = config_args.config
    config_args.config = None
    print(f'Using config: {config_name}')

    with open(config_name) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    print(f"Currnet Using Model : {conf['ModelName']}")

    model_name = conf['ModelName']
    args = argparse.Namespace(**conf[model_name])
    args = parser_maker.update_config(args, config_args)
    args.ModelName = conf['ModelName']
    basic_args = args
    args = wandb_stuff.setup(conf, args)
    print(f'Get Args {model_name}')
    pprint.pprint({k:v for k, v in args.items()})
    print()
    
    args.device = torch.device(args.device_pref if torch.cuda.is_available() else "cpu")

    # random seed 고정
    seed_everything(args.random_seed)

    # Data Loading
    train_data_file = f"{args.data_dir}/train_dials.json"
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
    
    ontology = json.load(open(f"{args.data_dir}/edit_ontology_metro.json"))
    
    train_data, dev_data, dev_labels = load_dataset(train_data_file,
                 use_small=args.use_small_data)

    tokenizer, processor, train_features, dev_features = get_stuff(args,
                 train_data, dev_data, slot_meta, ontology)
    
    # Slot Meta tokenizing for the decoder initial inputs
    tokenized_slot_meta = []
    for slot in slot_meta:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )
    
    # Model 선언
    model =  get_model(args, tokenizer, ontology, slot_meta)
    # print(f'Moving model to {args.device}')
    pbar = tqdm(desc=f'Moving model to {args.device} -- waiting...', bar_format='{desc} -> {elapsed}')
    model.to(args.device)
    pbar.set_description(f'Moving model to {args.device} -- DONE')  
    pbar.close()
    # print(f'Finished moving model to {args.device}')

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
    ### 이 부분은 SUMBT에만 있던거
    ### TODO: args.no_decay로 하면 어떨까?
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

    n_epochs = args.num_train_epochs
    t_total = len(train_loader) * n_epochs
    warmup_steps = int(t_total * args.warmup_ratio)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    args_save = {k:v for k, v in args.items() if k in basic_args}
    json.dump(
        args_save,
        open(f"{args.model_dir}/exp_config.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    json.dump(
        slot_meta,
        open(f"{args.model_dir}/slot_meta.json", "w"),
        indent=2,
        ensure_ascii=False,
    )

    json.dump(
        ontology,
        open(f"{args.model_dir}/edit_ontology_metro.json", "w"),
        indent=2,
        ensure_ascii=False,
    )

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
    else:
        raise NotImplementedError()
    
    scaler = GradScaler(enabled=args.use_amp)
    best_score, best_checkpoint = 0, 0
    total_step = 0
    
    loss_recorder = RunningLossRecorder(args.train_running_loss_len)

    wrong_list=[] #에폭별로 wrong_answer를 담아둘 배열
    correct_list=[] #wrong_list의 에폭별 오답률을 위해 correct_answer를 담아둘 배열

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
        val_predictions, val_loss_dict = inference_func(model, dev_loader, processor, args.device, args.use_amp, 
                loss_fnc=loss_fnc)
        # 현재 에폭에서 eval_result 외에도 틀린 예측값, ground truth값을 뽑아낸다
        eval_result,now_wrong_list,now_correct_list = _evaluation(val_predictions, dev_labels, slot_meta)
        #eda
        domain_counter,slot_counter,value_counter=get_Domain_Slot_Value_distribution_counter(Counter(now_wrong_list))
        o_domain_counter,o_slot_counter,o_value_counter=get_Domain_Slot_Value_distribution_counter(Counter(now_correct_list))
        draw_EDA('domain',domain_counter,o_domain_counter, epoch)
        draw_EDA('slot',slot_counter,o_slot_counter, epoch)
        draw_EDA('value',value_counter,o_value_counter, epoch)
        wrong_list.append(now_wrong_list)
        correct_list.append(now_correct_list)
        print('---------Validation-----------')
        for k, v in eval_result.items():
            print(f"{k}: {v:.4f}")
        if len(loss_dict) >= 5:
            loss_showing = f'loss: {loss_dict.loss:.4f}'
        else:
            loss_showing = ' '.join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
        print(loss_showing)
        print('------------------------------')
        wandb_stuff.val_log(args, eval_result, loss_dict, total_step, epoch+1)

        if best_score < eval_result['joint_goal_accuracy']:
            print("Update Best checkpoint!",)
            best_score = eval_result['joint_goal_accuracy']
            wandb_stuff.best_jga_log(args, best_score)
            best_checkpoint = epoch
            
            if args.save_model:
                torch.save(model.state_dict(), f"{args.model_dir}/model-best.bin")

        print()
        # torch.save(model.state_dict(), f"{args.model_dir}/model-{epoch}.bin")
    print(f"Best checkpoint: {best_checkpoint}",)
    draw_WrongTrend(wrong_list)