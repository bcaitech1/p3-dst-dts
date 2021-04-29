import argparse
import json
import os
import random
from attrdict import AttrDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

from data_utils import (WOSDataset, load_dataset,
                        seed_everything)
from eval_utils import DSTEvaluator
from evaluation import _evaluation
# from inference import inference
from model import TRADE, masked_cross_entropy_for_value
# from preprocessor import TRADEPreprocessor

from train_loop import trade_train_loop, submt_train_loop
from inference import trade_inference, sumbt_inference 

from prepare_preprocessor import get_stuff, get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/train_dataset")
    parser.add_argument("--model_dir", type=str, default="/opt/ml/results")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--warmup_ratio", type=int, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Subword Vocab만을 위한 huggingface model",
        default="monologg/koelectra-base-v3-discriminator",
    )

    # Model Specific Argument
    parser.add_argument("--hidden_size", type=int, help="GRU의 hidden size", default=768)
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="vocab size, subword vocab tokenizer에 의해 특정된다",
        default=None,
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--proj_dim", type=int,
                        help="만약 지정되면 기존의 hidden_size는 embedding dimension으로 취급되고, proj_dim이 GRU의 hidden_size로 사용됨. hidden_size보다 작아야 함.", default=None)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    args = parser.parse_args()

    #### 추가 #####

    args.preprocessor = 'TRADEPreprocessor'
    args.model_class = 'TRADE'    
    args.use_amp = True

    ###############

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
    model =  get_model(args, tokenizer, ontology)
    model.to(device)

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

    json.dump(
        vars(args),
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

        predictions = inference_func(model, dev_loader, processor, device, args.use_amp)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)
        for k, v in eval_result.items():
            print(f"{k}: {v}")

        if best_score < eval_result['joint_goal_accuracy']:
            print("Update Best checkpoint!")
            best_score = eval_result['joint_goal_accuracy']
            best_checkpoint = epoch

        torch.save(model.state_dict(), f"{args.model_dir}/model-{epoch}.bin")
    print(f"Best checkpoint: {args.model_dir}/model-{best_checkpoint}.bin")
