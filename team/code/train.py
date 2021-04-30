import argparse
import json
import os
import random
import yaml
from attrdict import AttrDict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from data_utils import (WOSDataset, load_dataset,
                        seed_everything)
from evaluation import _evaluation

from train_loop import trade_train_loop, submt_train_loop
from inference import trade_inference, sumbt_inference 

from prepare import get_stuff, get_model




if __name__ == "__main__":
    with open('conf2.yml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    print(f"Currnet Using Model : {conf['ModelName'][0]}")

    if conf['ModelName'] == 'TRADE':
        print("get_args_TRADE")
        args = argparse.Namespace(**conf['TRADE'])

    if conf['ModelName'] == 'SUMBT':
        print("get_args_SUMBT")
        args = argparse.Namespace(**conf['SUMBT'])
    print(args)

    ###############

    args.device = torch.device(args.device_pref if torch.cuda.is_available() else "cpu")

    # random seed 고정
    seed_everything(args.random_seed)

    # Data Loading
    train_data_file = f"{args.data_dir}/train_dials.json"
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
    ontology = json.load(open(f"{args.data_dir}/ontology.json"))
    train_data, dev_data, dev_labels = load_dataset(train_data_file)

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
    model.to(args.device)

    train_data = WOSDataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=processor.collate_fn,
    )
    print("# train:", len(train_data))

    dev_data = WOSDataset(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.eval_batch_size,
        sampler=dev_sampler,
        collate_fn=processor.collate_fn,
    )
    print("# dev:", len(dev_data))
    
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

    args_save = {k:v for k, v in vars(args).items() if k != 'device'}
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
        open(f"{args.model_dir}/ontology.json", "w"),
        indent=2,
        ensure_ascii=False,
    )

    if args.model_class == 'TRADE':
        train_loop = trade_train_loop
        train_loop_kwargs = AttrDict(pad_token_id=tokenizer.pad_token_id)
        inference_func = trade_inference
    elif args.model_class == 'SUMBT':
        train_loop = submt_train_loop
        train_loop_kwargs = AttrDict()
        inference_func = sumbt_inference
    else:
        raise NotImplementedError()
    
    scaler = GradScaler(enabled=args.use_amp)
    best_score, best_checkpoint = 0, 0
    for epoch in tqdm(range(n_epochs)):
        model.train()
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            loss = train_loop(args, model, batch, **train_loop_kwargs)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            
            scale = scaler.get_scale()
            scaler.update()
            step_scheduler = scaler.get_scale() == scale
            
            if step_scheduler:
                scheduler.step()

            if step % 100 == 0:
                print(
                    f"[{epoch}/{n_epochs}] [{step}/{len(train_loader)}] loss: {loss.item()}"
            )   

        predictions = inference_func(model, dev_loader, processor, args.device, args.use_amp)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)
        for k, v in eval_result.items():
            print(f"{k}: {v}")

        if best_score < eval_result['joint_goal_accuracy']:
            print("Update Best checkpoint!")
            best_score = eval_result['joint_goal_accuracy']
            best_checkpoint = epoch

        torch.save(model.state_dict(), f"{args.model_dir}/model-{epoch}.bin")
    print(f"Best checkpoint: {args.model_dir}/model-{best_checkpoint}.bin")
