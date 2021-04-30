import argparse
import os
import json

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from transformers import BertTokenizer

from data_utils import (WOSDataset, get_examples_from_dialogues)
from model import TRADE
from preprocessor import TRADEPreprocessor
from prepare_preprocessor import get_model, get_stuff


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def postprocess_state(state):
    for i, s in enumerate(state):
        s = s.replace(" : ", ":")
        state[i] = s.replace(" , ", ", ")
    return state


def trade_inference(model, eval_loader, processor, device, use_amp=False):
    model.eval()
    predictions = {}
    for batch in tqdm(eval_loader, total=len(eval_loader)):
        input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
            b.to(device) if not isinstance(b, list) else b for b in batch
        ]

        with torch.no_grad():
            with autocast(enabled=use_amp):
                o, g = model(input_ids, segment_ids, input_masks, 9)

            _, generated_ids = o.max(-1)
            _, gated_ids = g.max(-1)

        for guid, gate, gen in zip(guids, gated_ids.tolist(), generated_ids.tolist()):
            prediction = processor.recover_state(gate, gen)
            prediction = postprocess_state(prediction)
            predictions[guid] = prediction
    return predictions

def sumbt_inference(model, eval_loader, processor, device, use_amp=False):
    model.eval()
    predictions = {}
    
    for step, batch in tqdm(enumerate(eval_loader), total=len(eval_loader)):
        input_ids, segment_ids, input_masks, target_ids, num_turns, guids  = \
            [b.to(device) if not isinstance(b, list) else b for b in batch]
        
        with torch.no_grad():
            with autocast(enabled=use_amp):
                output, pred_slot = model(input_ids, segment_ids, input_masks, None, 1)
            
        pred_slot = pred_slot.detach().cpu()
        
        for guid, num_turn, p_slot in zip(guids, num_turns, pred_slot):
            pred_states = processor.recover_state(p_slot, num_turn)
            for t_idx, pred_state in enumerate(pred_states):
                predictions[f'{guid}-{t_idx}'] = pred_state
    
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    args = parser.parse_args()
    
#     args.preprocessor = 'TRADEPreprocessor'
#     args.model_class = 'TRADE'    
    args.use_amp = True
    
    model_dir_path = os.path.dirname(args.model_dir)
    eval_data = json.load(open(f"{args.data_dir}/eval_dials.json", "r"))
    config = json.load(open(f"{model_dir_path}/exp_config.json", "r"))
    config = argparse.Namespace(**config)
    slot_meta = json.load(open(f"{model_dir_path}/slot_meta.json", "r"))
    ontology = json.load(open(f"{model_dir_path}/ontology.json", "r"))

    tokenizer, processor, eval_features, _ = get_stuff(config,
                 eval_data, None, slot_meta, ontology)

    eval_data = WOSDataset(eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_loader = DataLoader(
        eval_data,
        batch_size=args.eval_batch_size,
        sampler=eval_sampler,
        collate_fn=processor.collate_fn,
    )
    print("# eval:", len(eval_data))

    model =  get_model(config, tokenizer, ontology, slot_meta)

    ckpt = torch.load(args.model_dir, map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device)
    print("Model is loaded")

    if config.model_class == 'TRADE':
        inference_func = trade_inference
    elif config.model_class == 'SUMBT':
        inference_func = sumbt_inference
    else:
        raise NotImplementedError()

    predictions = inference_func(model, eval_loader, processor, device, args.use_amp)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    json.dump(
        predictions,
        open(f"{args.output_dir}/predictions.csv", "w"),
        indent=2,
        ensure_ascii=False,
    )
