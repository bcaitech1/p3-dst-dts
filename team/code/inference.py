import argparse
import os
import json
import sys

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.cuda.amp import autocast
from tqdm.auto import tqdm

from data_utils import WOSDataset
from prepare import get_model, get_stuff, filter_inference

from training_recorder import RunningLossRecorder

def postprocess_state(state):
    for i, s in enumerate(state):
        s = s.replace(" : ", ":")
        state[i] = s.replace(" , ", ", ")
    return state


def trade_inference(model, eval_loader, processor, device, use_amp=False, 
        loss_fnc=None):
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

inference = trade_inference

def sumbt_inference(model, eval_loader, processor, device, use_amp=False,
        loss_fnc=None):
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    args = parser.parse_args()
    
    model_dir_path = os.path.dirname(args.model_dir)
    eval_data = json.load(open(f"{args.data_dir}/eval_dials.json", "r"))
    config = json.load(open(f"{model_dir_path}/exp_config.json", "r"))
    slot_meta = json.load(open(f"{model_dir_path}/slot_meta.json", "r"))
    ontology = json.load(open(f"{model_dir_path}/edit_ontology_metro.json", "r"))

    config = argparse.Namespace(**config)
    config.device = torch.device(config.device_pref if torch.cuda.is_available() else "cpu")
    config.devcie = torch.device('cpu')

    eval_data, slot_meta, ontology = filter_inference(config, eval_data, slot_meta, ontology)

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

    if config.ModelName == 'TRADE':
        inference_func = trade_inference
    elif config.ModelName == 'SUMBT':
        inference_func = sumbt_inference
    else:
        raise NotImplementedError()

    predictions = inference_func(model, eval_loader, processor, device, config.use_amp)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    json.dump(
        predictions,
        open(f"{args.output_dir}/predictions3.csv", "w"),
        indent=2,
        ensure_ascii=False,
    )
