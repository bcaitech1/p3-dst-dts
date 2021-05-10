#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import BertModel, BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
from data_utils import (
    load_dataset, 
    get_examples_from_dialogues, 
    convert_state_dict, 
    DSTInputExample, 
    OpenVocabDSTFeature, 
    DSTPreprocessor, 
    WOSDataset)
    
from inference import inference
from evaluation import _evaluation


# ## Data loading

# In[2]:


train_data_file = "/opt/ml/input/data/train_dataset/train_dials.json"
slot_meta = json.load(open("/opt/ml/input/data/train_dataset/slot_meta.json"))
ontology = json.load(open("/opt/ml/input/data/train_dataset/ontology.json"))
train_data, dev_data, dev_labels = load_dataset(train_data_file)

train_examples = get_examples_from_dialogues(train_data,
                                             user_first=False,
                                             dialogue_level=False)
dev_examples = get_examples_from_dialogues(dev_data,
                                           user_first=False,
                                           dialogue_level=False)


# In[3]:


print(len(train_examples))
print(len(dev_examples))


# ## TRADE Preprocessor 

# 기존의 GRU 기반의 인코더를 BERT-based Encoder로 바꿀 준비를 합시다.
# 
# 1. 현재 `_convert_example_to_feature`에서는 `max_seq_length`를 핸들하고 있지 않습니다. `input_id`와 `segment_id`가 `max_seq_length`를 넘어가면 좌측부터 truncate시키는 코드를 삽입하세요.
# 
# 2. hybrid approach에서 얻은 교훈을 바탕으로 gate class를 3개에서 5개로 늘려봅시다.
#     - `gating2id`를 수정하세요
#     - 이에 따른 `recover_state`를 수정하세요.
#     
# 3. word dropout을 구현하세요.

# In[4]:


class TRADEPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        ontology=None,
        max_seq_length=512,
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.gating2id = {"none": 0, "dontcare": 1, "yes": 2, "no": 3, "ptr": 4}
        self.id2gating = {v: k for k, v in self.gating2id.items()}
        self.max_seq_length = max_seq_length

    def _convert_example_to_feature(self, example):
        dialogue_context = " [SEP] ".join(example.context_turns + example.current_turn)

        input_id = self.src_tokenizer.encode(dialogue_context, add_special_tokens=False)
        max_length = self.max_seq_length - 2
        if len(input_id) > max_length:
            gap = len(input_id) - max_length
            input_id = input_id[gap:]

        input_id = (
            [self.src_tokenizer.cls_token_id]
            + input_id
            + [self.src_tokenizer.sep_token_id]
        )
        segment_id = [0] * len(input_id)

        target_ids = []
        gating_id = []
        if not example.label:
            example.label = []

        state = convert_state_dict(example.label)
        for slot in self.slot_meta:
            value = state.get(slot, "none")
            target_id = self.trg_tokenizer.encode(value, add_special_tokens=False) + [
                self.trg_tokenizer.sep_token_id
            ]
            target_ids.append(target_id)
            gating_id.append(self.gating2id.get(value, self.gating2id["ptr"]))
        target_ids = self.pad_ids(target_ids, self.trg_tokenizer.pad_token_id)
        return OpenVocabDSTFeature(
            example.guid, input_id, segment_id, gating_id, target_ids
        )

    def convert_examples_to_features(self, examples):
        return list(map(self._convert_example_to_feature, examples))

    def recover_state(self, gate_list, gen_list):
        assert len(gate_list) == len(self.slot_meta)
        assert len(gen_list) == len(self.slot_meta)

        recovered = []
        for slot, gate, value in zip(self.slot_meta, gate_list, gen_list):
            if self.id2gating[gate] == "none":
                continue

            if self.id2gating[gate] in ["dontcare", "yes", "no"]:
                recovered.append("%s-%s" % (slot, self.id2gating[gate]))
                continue

            token_id_list = []
            for id_ in value:
                if id_ in self.trg_tokenizer.all_special_ids:
                    break

                token_id_list.append(id_)
            value = self.trg_tokenizer.decode(token_id_list, skip_special_tokens=True)

            if value == "none":
                continue

            recovered.append("%s-%s" % (slot, value))
        return recovered

    def collate_fn(self, batch):
        guids = [b.guid for b in batch]
        input_ids = torch.LongTensor(
            self.pad_ids([b.input_id for b in batch], self.src_tokenizer.pad_token_id)
        )
        segment_ids = torch.LongTensor(
            self.pad_ids([b.segment_id for b in batch], self.src_tokenizer.pad_token_id)
        )
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id)

        gating_ids = torch.LongTensor([b.gating_id for b in batch])
        target_ids = self.pad_id_of_matrix(
            [torch.LongTensor(b.target_ids) for b in batch],
            self.trg_tokenizer.pad_token_id,
        )
        return input_ids, segment_ids, input_masks, gating_ids, target_ids, guids


# ## Convert_Examples_to_Features 

# In[ ]:


tokenizer = BertTokenizer.from_pretrained('dsksd/bert-ko-small-minimal')
processor = TRADEPreprocessor(slot_meta, tokenizer, max_seq_length=512)

train_features = processor.convert_examples_to_features(train_examples)
dev_features = processor.convert_examples_to_features(dev_examples)


# In[ ]:


print(len(train_features))
print(len(dev_features))


# # Model 

# 1. `GRUEncoder`를 `BertModel`로 교체하세요. 이에 따라 `tie_weight` 함수가 수정되어야 합니다.

# In[ ]:


class TRADE(nn.Module):
    def __init__(self, config, slot_vocab, slot_meta, pad_idx=0):
        super(TRADE, self).__init__()
        self.slot_meta = slot_meta
        if config.model_name_or_path:
            self.encoder = BertModel.from_pretrained(config.model_name_or_path)
        else:
            self.encoder = BertModel(config)
            
        self.decoder = SlotGenerator(
            config.vocab_size,
            config.hidden_size,
            config.hidden_dropout_prob,
            config.n_gate,
            None,
            pad_idx,
        )
        
        # init for only subword embedding
        self.decoder.set_slot_idx(slot_vocab)
        self.tie_weight()

    def tie_weight(self):
        self.decoder.embed.weight = self.encoder.embeddings.word_embeddings.weight

    def forward(self, input_ids, token_type_ids, attention_mask=None, max_len=10, teacher=None):

        encoder_outputs, pooled_output = self.encoder(input_ids=input_ids)
        all_point_outputs, all_gate_outputs = self.decoder(
            input_ids, encoder_outputs, pooled_output.unsqueeze(0), attention_mask, max_len, teacher
        )

        return all_point_outputs, all_gate_outputs
    
class SlotGenerator(nn.Module):
    def __init__(
        self, vocab_size, hidden_size, dropout, n_gate, proj_dim=None, pad_idx=0
    ):
        super(SlotGenerator, self).__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_idx
        )  # shared with encoder

        if proj_dim:
            self.proj_layer = nn.Linear(hidden_size, proj_dim, bias=False)
        else:
            self.proj_layer = None
        self.hidden_size = proj_dim if proj_dim else hidden_size

        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size, 1, dropout=dropout, batch_first=True
        )
        self.n_gate = n_gate
        self.dropout = nn.Dropout(dropout)
        self.w_gen = nn.Linear(self.hidden_size * 3, 1)
        self.sigmoid = nn.Sigmoid()
        self.w_gate = nn.Linear(self.hidden_size, n_gate)

    def set_slot_idx(self, slot_vocab_idx):
        whole = []
        max_length = max(map(len, slot_vocab_idx))
        for idx in slot_vocab_idx:
            if len(idx) < max_length:
                gap = max_length - len(idx)
                idx.extend([self.pad_idx] * gap)
            whole.append(idx)
        self.slot_embed_idx = whole  # torch.LongTensor(whole)

    def embedding(self, x):
        x = self.embed(x)
        if self.proj_layer:
            x = self.proj_layer(x)
        return x

    def forward(
        self, input_ids, encoder_output, hidden, input_masks, max_len, teacher=None
    ):
        input_masks = input_masks.ne(1)
        # J, slot_meta : key : [domain, slot] ex> LongTensor([1,2])
        # J,2
        batch_size = encoder_output.size(0)
        slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device)  ##
        slot_e = torch.sum(self.embedding(slot), 1)  # J,d
        J = slot_e.size(0)

        all_point_outputs = torch.zeros(batch_size, J, max_len, self.vocab_size).to(
            input_ids.device
        )
        
        # Parallel Decoding
        w = slot_e.repeat(batch_size, 1).unsqueeze(1)
        hidden = hidden.repeat_interleave(J, dim=1)
        encoder_output = encoder_output.repeat_interleave(J, dim=0)
        input_ids = input_ids.repeat_interleave(J, dim=0)
        input_masks = input_masks.repeat_interleave(J, dim=0)
        for k in range(max_len):
            w = self.dropout(w)
            _, hidden = self.gru(w, hidden)  # 1,B,D

            # B,T,D * B,D,1 => B,T
            attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
            attn_e = attn_e.squeeze(-1).masked_fill(input_masks, -1e9)
            attn_history = F.softmax(attn_e, -1)  # B,T

            if self.proj_layer:
                hidden_proj = torch.matmul(hidden, self.proj_layer.weight)
            else:
                hidden_proj = hidden

            # B,D * D,V => B,V
            attn_v = torch.matmul(
                hidden_proj.squeeze(0), self.embed.weight.transpose(0, 1)
            )  # B,V
            attn_vocab = F.softmax(attn_v, -1)

            # B,1,T * B,T,D => B,1,D
            context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D
            p_gen = self.sigmoid(
                self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1))
            )  # B,1
            p_gen = p_gen.squeeze(-1)

            p_context_ptr = torch.zeros_like(attn_vocab).to(input_ids.device)
            p_context_ptr.scatter_add_(1, input_ids, attn_history)  # copy B,V
            p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
            _, w_idx = p_final.max(-1)

            if teacher is not None:
                w = self.embedding(teacher[:, :, k]).transpose(0, 1).reshape(batch_size * J, 1, -1)
            else:
                w = self.embedding(w_idx).unsqueeze(1)  # B,1,D
            if k == 0:
                gated_logit = self.w_gate(context.squeeze(1))  # B,3
                all_gate_outputs = gated_logit.view(batch_size, J, self.n_gate)
            all_point_outputs[:, :, k, :] = p_final.view(batch_size, J, self.vocab_size)

        return all_point_outputs, all_gate_outputs


# # 모델 및 데이터 로더 정의

# In[ ]:


slot_vocab = []
for slot in slot_meta:
    slot_vocab.append(
        tokenizer.encode(slot.replace('-', ' '),
                         add_special_tokens=False)
    )
    
config = BertConfig.from_pretrained('dsksd/bert-ko-small-minimal')
config.model_name_or_path = 'dsksd/bert-ko-small-minimal'
config.n_gate = len(processor.gating2id)
config.proj_dim = None
model = TRADE(config, slot_vocab, slot_meta)


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = WOSDataset(train_features)
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, batch_size=16, sampler=train_sampler, collate_fn=processor.collate_fn)

dev_data = WOSDataset(dev_features)
dev_sampler = SequentialSampler(dev_data)
dev_loader = DataLoader(dev_data, batch_size=8, sampler=dev_sampler, collate_fn=processor.collate_fn)


# # Optimizer & Scheduler 선언

# In[ ]:


n_epochs = 30
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

t_total = len(train_loader) * n_epochs
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0.1, num_training_steps=t_total
)
teacher_forcing = 0.5
model.to(device)

def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss

loss_fnc_1 = masked_cross_entropy_for_value  # generation
loss_fnc_2 = nn.CrossEntropyLoss()  # gating


# ## Train

# In[ ]:


for epoch in range(n_epochs):
   batch_loss = []
   model.train()
   for step, batch in enumerate(train_loader):
       input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [b.to(device) if not isinstance(b, list) else b for b in batch]
       if teacher_forcing > 0.0 and random.random() < teacher_forcing:
           tf = target_ids
       else:
           tf = None

       all_point_outputs, all_gate_outputs = model(input_ids, segment_ids, input_masks, target_ids.size(-1))  # gt - length (generation)
       loss_1 = loss_fnc_1(all_point_outputs.contiguous(), target_ids.contiguous().view(-1))
       loss_2 = loss_fnc_2(all_gate_outputs.contiguous().view(-1, 5), gating_ids.contiguous().view(-1))
       loss = loss_1 + loss_2
       batch_loss.append(loss.item())

       loss.backward()
       nn.utils.clip_grad_norm_(model.parameters(), 1.0)
       optimizer.step()
       scheduler.step()
       optimizer.zero_grad()
       if step % 100 == 0:
           print('[%d/%d] [%d/%d] %f' % (epoch, n_epochs, step, len(train_loader), loss.item()))
               
   predictions = inference(model, dev_loader, processor, device)
   eval_result = _evaluation(predictions, dev_labels, slot_meta)
   for k, v in eval_result.items():
       print(f"{k}: {v}")


# ## Inference 

# In[ ]:


eval_data = json.load(open(f"/opt/ml/input/data/eval_dataset/eval_dials.json", "r"))

eval_examples = get_examples_from_dialogues(
    eval_data, user_first=False, dialogue_level=False
)

# Extracting Featrues
eval_features = processor.convert_examples_to_features(eval_examples)
eval_data = WOSDataset(eval_features)
eval_sampler = SequentialSampler(eval_data)
eval_loader = DataLoader(
    eval_data,
    batch_size=8,
    sampler=eval_sampler,
    collate_fn=processor.collate_fn,
)


# In[ ]:


predictions = inference(model, eval_loader, processor, device)


# In[ ]:


json.dump(predictions, open('predictions.csv', 'w'), indent=2, ensure_ascii=False) 


# In[ ]:




