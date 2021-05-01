"""
Most of code is from https://github.com/SKTBrain/SUMBT
"""
import math
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from .sumbt import AutoForUtteranceEncoding, MultiHeadAttention
from torch_position_embedding import PositionEmbedding

class Transformer(nn.Module):
    def __init__(self, attn_head, output_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(attn_head, output_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffwd = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x, mask):
        tmp = self.layer_norm1(x)
        sub_layer = self.attn(tmp, tmp, tmp, mask)
        x = x + self.dropout(sub_layer)

        tmp = self.layer_norm2(x)
        sub_layer = self.ffwd(tmp)
        x = x + self.dropout(sub_layer)

        return x

class SUMBT_Custom(nn.Module):
    def __init__(self, args, num_labels, device):
        super().__init__()

        self.hidden_dim = args.hidden_dim
        self.rnn_num_layers = args.num_rnn_layers
        self.zero_init_rnn = args.zero_init_rnn
        self.max_seq_length = args.max_seq_length
        self.max_label_length = args.max_label_length
        self.num_labels = num_labels
        self.num_slots = len(num_labels)
        self.attn_head = args.attn_head
        self.device = device

        ### Utterance Encoder
        self.utterance_encoder = AutoForUtteranceEncoding.from_pretrained(
            args.model_name_or_path
        )
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        try:
            self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        except AttributeError:
            self.hidden_dropout_prob = 0.1

        if args.fix_utterance_encoder:
            for p in self.utterance_encoder.base.pooler.parameters():
                p.requires_grad = False

        ### slot, slot-value Encoder (not trainable)
        self.sv_encoder = AutoForUtteranceEncoding.from_pretrained(
            args.model_name_or_path
        )
        # os.path.join(args.bert_dir, 'bert-base-uncased.model'))
        for p in self.sv_encoder.base.parameters():
            p.requires_grad = False

        self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim)
        self.value_lookup = nn.ModuleList(
            [nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels]
        )

        ### Attention layer
        self.attn = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0)
        
        self.transformers = nn.ModuleList(
            [Transformer(self.attn_head, self.bert_output_dim, dropout=0) for _ in range(args.n_transformers)]
        )

        self.position_emb = PositionEmbedding(num_embeddings=50,
            embedding_dim=self.bert_output_dim,
            mode=PositionEmbedding.MODE_ADD)

        ### RNN Belief Tracker
        # self.nbt = nn.GRU(
        #     input_size=self.bert_output_dim,
        #     hidden_size=self.hidden_dim,
        #     num_layers=self.rnn_num_layers,
        #     dropout=self.hidden_dropout_prob,
        #     batch_first=True,
        # )
        # self.init_parameter(self.nbt)

        # if not self.zero_init_rnn:
        #     self.rnn_init_linear = nn.Sequential(
        #         nn.Linear(self.bert_output_dim, self.hidden_dim),
        #         nn.ReLU(),
        #         nn.Dropout(self.hidden_dropout_prob),
        #     )

        self.linear = nn.Linear(self.hidden_dim, self.bert_output_dim)
        self.layer_norm = nn.LayerNorm(self.bert_output_dim)

        ### Measure
        self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        ### Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1)

        ### Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def initialize_slot_value_lookup(self, label_ids, slot_ids):

        self.sv_encoder.eval()

        # Slot encoding
        slot_type_ids = torch.zeros(slot_ids.size(), dtype=torch.long).to(
            slot_ids.device
        )
        slot_mask = slot_ids > 0
        hid_slot = self.sv_encoder(
            slot_ids.view(-1, self.max_label_length),
            slot_type_ids.view(-1, self.max_label_length),
            slot_mask.view(-1, self.max_label_length),
        ).hidden_states[-1]
        
        # hid_slot = hid_slot[0].masked_fill(slot_mask[0].unsqueeze(-1) == False, 0)
        hid_slot = hid_slot[:, 0, :]
        hid_slot = hid_slot.detach()
    
        self.slot_lookup = nn.Embedding.from_pretrained(hid_slot, freeze=True)
        assert self.slot_lookup.weight.shape == (self.num_slots, self.bert_output_dim), f'{self.slot_lookup.weight.shape} {(self.num_slots, self.bert_output_dim)}'

        for s, label_id in enumerate(label_ids):
            label_type_ids = torch.zeros(label_id.size(), dtype=torch.long).to(
                label_id.device
            )
            label_mask = label_id > 0
            hid_label = self.sv_encoder(
                label_id.view(-1, self.max_label_length),
                label_type_ids.view(-1, self.max_label_length),
                label_mask.view(-1, self.max_label_length),
            ).hidden_states[-1]
            hid_label = hid_label[:, 0, :]
            hid_label = hid_label.detach()
            self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
            self.value_lookup[s].padding_idx = -1
            assert self.value_lookup[s].weight.shape == (label_id.size(0), self.bert_output_dim), f'{self.value_lookup[s].weight.shape} {(label_id.size(0), self.bert_output_dim)}'

        print("Complete initialization of slot and value lookup")
        self.sv_encoder = None

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        labels=None,
        n_gpu=1,
        target_slot=None,
    ):
        # B = Batch Size
        # M = Max Turn Size
        # N = Seq Len
        # J = Target_slot Len
        # H_GRU = RNN Hidden Dim
        
        # input_ids: [B, M, N]
        # token_type_ids: [B, M, N]
        # attention_mask: [B, M, N]
        # labels: [B, M, J]

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0)  # Batch size (B)
        ts = input_ids.size(1)  # Max turn size (M)
        bs = ds * ts # B * M
        slot_dim = len(target_slot)  # J

        # Utterance encoding
        hidden = self.utterance_encoder(
            input_ids.view(-1, self.max_seq_length),
            token_type_ids.view(-1, self.max_seq_length),
            attention_mask.view(-1, self.max_seq_length),
        ).hidden_states[-1] # [B*M, N, H]

        hidden = torch.mul(
            hidden,
            attention_mask.view(-1, self.max_seq_length, 1)
            .expand(hidden.size())
            .float(),
        )
        hidden = hidden.repeat(slot_dim, 1, 1)  # [J*M*B, N, H]

        hid_slot = self.slot_lookup.weight[
            target_slot, :
        ]  # Select target slot embedding # [J, H]
        hid_slot = hid_slot.repeat(1, bs).view(bs * slot_dim, -1)  # [J*M*B, H]

        # Attended utterance vector
        hidden = self.attn(
            hid_slot,  # q^s  [J*M*B, H] => [J*M*B, 1, H]
            hidden,  # U [J*M*B, N, H]
            hidden,  # U [J*M*B, N, H]
            mask=attention_mask.view(-1, 1, self.max_seq_length).repeat(slot_dim, 1, 1),
        ) # [J*M*B, 1, H] -> 1 = hid_slot_seq_len
        hidden = hidden.squeeze()  # h [J*M*B, H] Aggregated Slot Context
        hidden = hidden.view(slot_dim, ds, ts, -1).view(
            -1, ts, self.bert_output_dim
        )  # [J*B, M, H]

        hidden = self.position_emb(hidden)

        transformer_mask = torch.any(attention_mask, -1).repeat(slot_dim, 1).unsqueeze(1)
        for transformer in self.transformers:
            hidden = transformer(
                hidden,
                mask=transformer_mask # [J*B, 1, M]
            ) # [J*B, M, H]

        hidden = hidden.view(slot_dim, ds, ts, -1)

        # Label (slot-value) encoding
        loss = 0
        loss_slot = []
        pred_slot = []
        output = []
        for s, slot_id in enumerate(target_slot):  ## note: target_slots are successive
            # loss calculation
            hid_label = self.value_lookup[slot_id].weight
            num_slot_labels = hid_label.size(0)

            _hid_label = (
                hid_label.unsqueeze(0)
                .unsqueeze(0)
               .repeat(ds, ts, 1, 1)
                .view(ds * ts * num_slot_labels, -1)
            )
            _hidden = (
                hidden[s, :, :, :]
                .unsqueeze(2)
                .repeat(1, 1, num_slot_labels, 1)
                .view(ds * ts * num_slot_labels, -1)
            )
            _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)
            _dist = -_dist
            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.view(ds, ts, 1))
            output.append(_dist)

            if labels is not None:
                _loss = self.nll(_dist.view(ds * ts, -1), labels[:, :, s].view(-1))
                loss_slot.append(_loss.item())
                loss += _loss

        pred_slot = torch.cat(pred_slot, 2)
        
        if labels is None:
            return output, pred_slot

        # calculate joint accuracy
        accuracy = (pred_slot == labels).view(-1, slot_dim)
        acc_slot = (
            torch.sum(accuracy, 0).float()
            / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
        )
        acc = (
            sum(torch.sum(accuracy, 1) / slot_dim).float()
            / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float()
        )  # joint accuracy

        return loss, loss_slot, acc, acc_slot, pred_slot
        

    @staticmethod
    def init_parameter(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)

class SUMBT_Custom2(nn.Module):
    def __init__(self, args, num_labels, device):
        super().__init__()

        self.hidden_dim = args.hidden_dim
        self.rnn_num_layers = args.num_rnn_layers
        self.zero_init_rnn = args.zero_init_rnn
        self.max_seq_length = args.max_seq_length
        self.max_label_length = args.max_label_length
        self.num_labels = num_labels
        self.num_slots = len(num_labels)
        self.attn_head = args.attn_head
        self.device = device

        ### Utterance Encoder
        self.utterance_encoder = AutoForUtteranceEncoding.from_pretrained(
            args.model_name_or_path
        )
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        try:
            self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        except AttributeError:
            self.hidden_dropout_prob = 0.1

        if args.fix_utterance_encoder:
            for p in self.utterance_encoder.base.pooler.parameters():
                p.requires_grad = False

        ### slot, slot-value Encoder (not trainable)
        self.sv_encoder = AutoForUtteranceEncoding.from_pretrained(
            args.model_name_or_path
        )
        # os.path.join(args.bert_dir, 'bert-base-uncased.model'))
        for p in self.sv_encoder.base.parameters():
            p.requires_grad = False

        self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim*self.max_label_length)
        self.value_lookup = nn.ModuleList(
            [nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels]
        )

        ### Attention layer
        self.attn = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0)
        
        self.transformers = nn.ModuleList(
            [Transformer(self.attn_head, self.bert_output_dim, dropout=0) for _ in range(args.n_transformers)]
        )

        self.position_emb = PositionEmbedding(num_embeddings=50,
            embedding_dim=self.bert_output_dim,
            mode=PositionEmbedding.MODE_ADD)

        self.slot_pooler = nn.Linear(self.max_label_length * self.bert_output_dim, self.bert_output_dim)

        ### RNN Belief Tracker
        # self.nbt = nn.GRU(
        #     input_size=self.bert_output_dim,
        #     hidden_size=self.hidden_dim,
        #     num_layers=self.rnn_num_layers,
        #     dropout=self.hidden_dropout_prob,
        #     batch_first=True,
        # )
        # self.init_parameter(self.nbt)

        # if not self.zero_init_rnn:
        #     self.rnn_init_linear = nn.Sequential(
        #         nn.Linear(self.bert_output_dim, self.hidden_dim),
        #         nn.ReLU(),
        #         nn.Dropout(self.hidden_dropout_prob),
        #     )

        self.linear = nn.Linear(self.hidden_dim, self.bert_output_dim)
        self.layer_norm = nn.LayerNorm(self.bert_output_dim)

        ### Measure
        self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        ### Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1)

        ### Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def initialize_slot_value_lookup(self, label_ids, slot_ids):

        self.sv_encoder.eval()

        # Slot encoding
        slot_type_ids = torch.zeros(slot_ids.size(), dtype=torch.long).to(
            slot_ids.device
        )
        slot_mask = slot_ids > 0
        hid_slot = self.sv_encoder(
            slot_ids.view(-1, self.max_label_length),
            slot_type_ids.view(-1, self.max_label_length),
            slot_mask.view(-1, self.max_label_length),
        ).hidden_states[-1]
        
        hid_slot = hid_slot.masked_fill(slot_mask.unsqueeze(-1) == False, 0)
        hid_slot = hid_slot.view(self.num_slots, -1)
        hid_slot = hid_slot.detach()
    
        old = self.slot_lookup
        self.slot_lookup = nn.Embedding.from_pretrained(hid_slot, freeze=True)
        assert self.slot_lookup.weight.shape == old.weight.shape, f'{self.slot_lookup.weight.shape} {old.weight.shape}'

        for s, label_id in enumerate(label_ids):
            label_type_ids = torch.zeros(label_id.size(), dtype=torch.long).to(
                label_id.device
            )
            label_mask = label_id > 0
            hid_label = self.sv_encoder(
                label_id.view(-1, self.max_label_length),
                label_type_ids.view(-1, self.max_label_length),
                label_mask.view(-1, self.max_label_length),
            ).hidden_states[-1]
            hid_label = hid_label[:, 0, :]
            hid_label = hid_label.detach()
            self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
            self.value_lookup[s].padding_idx = -1
            assert self.value_lookup[s].weight.shape == (label_id.size(0), self.bert_output_dim), f'{self.value_lookup[s].weight.shape} {(label_id.size(0), self.bert_output_dim)}'

        print("Complete initialization of slot and value lookup")
        self.sv_encoder = None

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        labels=None,
        n_gpu=1,
        target_slot=None,
    ):
        # B = Batch Size
        # M = Max Turn Size
        # N = Seq Len
        # J = Target_slot Len
        # H_GRU = RNN Hidden Dim
        # L = Label Len
        
        # input_ids: [B, M, N]
        # token_type_ids: [B, M, N]
        # attention_mask: [B, M, N]
        # labels: [B, M, J]

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0)  # Batch size (B)
        ts = input_ids.size(1)  # Max turn size (M)
        bs = ds * ts # B * M
        slot_dim = len(target_slot)  # J

        # Utterance encoding
        hidden = self.utterance_encoder(
            input_ids.view(-1, self.max_seq_length),
            token_type_ids.view(-1, self.max_seq_length),
            attention_mask.view(-1, self.max_seq_length),
        ).hidden_states[-1] # [B*M, N, H]

        hidden = torch.mul(
            hidden,
            attention_mask.view(-1, self.max_seq_length, 1)
            .expand(hidden.size())
            .float(),
        )
        hidden = hidden.repeat(slot_dim, 1, 1)  # [J*M*B, N, H]

        hid_slot = self.slot_lookup.weight[
            target_slot, :
        ]  # Select target slot embedding # [J, H * L]
        hid_slot = hid_slot.view(slot_dim, self.max_label_length, self.bert_output_dim) # [J, L, H]
        hid_slot = hid_slot.repeat(1, bs, 1).view(bs * slot_dim, self.max_label_length, -1) # [J*M*B, L, H]

        # Attended utterance vector
        hidden = self.attn(
            hid_slot,  # q^s  [J*M*B, L, H]
            hidden,  # U [J*M*B, N, H]
            hidden,  # U [J*M*B, N, H]
            mask=attention_mask.view(-1, 1, self.max_seq_length).repeat(slot_dim, 1, 1),
        ) # [J*M*B, L, H]
        hidden = self.slot_pooler(hidden.view(-1, self.max_seq_length * self.bert_output_dim))
        # [J*M*B, H]
        hidden = hidden.view(slot_dim, ds, ts, -1).view(
            -1, ts, self.bert_output_dim
        )  # [J*B, M, H]

        hidden = self.position_emb(hidden)

        transformer_mask = torch.any(attention_mask, -1).repeat(slot_dim, 1).unsqueeze(1)
        for transformer in self.transformers:
            hidden = transformer(
                hidden,
                mask=transformer_mask # [J*B, 1, M]
            ) # [J*B, M, H]

        hidden = hidden.view(slot_dim, ds, ts, -1)

        # Label (slot-value) encoding
        loss = 0
        loss_slot = []
        pred_slot = []
        output = []
        for s, slot_id in enumerate(target_slot):  ## note: target_slots are successive
            # loss calculation
            hid_label = self.value_lookup[slot_id].weight
            num_slot_labels = hid_label.size(0)

            _hid_label = (
                hid_label.unsqueeze(0)
                .unsqueeze(0)
               .repeat(ds, ts, 1, 1)
                .view(ds * ts * num_slot_labels, -1)
            )
            _hidden = (
                hidden[s, :, :, :]
                .unsqueeze(2)
                .repeat(1, 1, num_slot_labels, 1)
                .view(ds * ts * num_slot_labels, -1)
            )
            _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)
            _dist = -_dist
            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.view(ds, ts, 1))
            output.append(_dist)

            if labels is not None:
                _loss = self.nll(_dist.view(ds * ts, -1), labels[:, :, s].view(-1))
                loss_slot.append(_loss.item())
                loss += _loss

        pred_slot = torch.cat(pred_slot, 2)
        
        if labels is None:
            return output, pred_slot

        # calculate joint accuracy
        accuracy = (pred_slot == labels).view(-1, slot_dim)
        acc_slot = (
            torch.sum(accuracy, 0).float()
            / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
        )
        acc = (
            sum(torch.sum(accuracy, 1) / slot_dim).float()
            / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float()
        )  # joint accuracy

        return loss, loss_slot, acc, acc_slot, pred_slot
        

    @staticmethod
    def init_parameter(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)