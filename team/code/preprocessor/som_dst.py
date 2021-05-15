from data_utils import DSTPreprocessor, _truncate_seq_pair, convert_state_dict
from tqdm.auto import tqdm
import torch
import random
import numpy as np

from dataclasses import dataclass
from typing import List, Optional, Union

flatten = lambda x: [i for s in x for i in s]

OP_SET = {
    '2': {'update': 0, 'carryover': 1},
    '3-1': {'update': 0, 'carryover': 1, 'dontcare': 2},
    '3-2': {'update': 0, 'carryover': 1, 'delete': 2},
    '4': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3},
    '6': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3, 'yes': 4, 'no': 5}
}

@dataclass
class SOMDSTFeature:
    guid: str
    input_ids: List[int]
    segment_ids: List[int]
    state_position_ids: List[int]

    # 불확실
    op_ids: List[int]
    domain_id: int
    gen_ids: List[List[int]]


def get_op_id(before_value, value, opcode):
    cur_op_set = OP_SET[opcode]

    if before_value == value:
        return cur_op_set['carryover']
    
    if value == 'none' and 'delete' in cur_op_set:
        return cur_op_set['delete']
    if value == 'yes' and 'yes' in cur_op_set:
        return cur_op_set['yes']
    if value == 'no' and 'no' in cur_op_set:
        return cur_op_set['no']
    if value == 'dontcare' and 'dontcare' in cur_op_set:
        return cur_op_set['dontcare']
    
    return cur_op_set['update']

# Batch Stuff: max_value, max_update

# TODO: special token add to tokenizer, [SLOT], [A-U], [S-V] [NULL]
# RESIZE MODEL EMB
# Assume example is SYS - USR order
# SLOT DATA MUST be SHUFFLED -> currently not
class SOMDSTPreprocessor(DSTPreprocessor):
    extra_special_tokens = ['[SLOT]', '[A-U]', '[S-V]', '[NULL]']
    eos_token = '[EOS]'
    def __init__(self, slot_meta, src_tokenizer, trg_tokenizer=None, ontology=None,
        max_seq=512, opcode='4'):
        src_tokenizer.add_special_tokens(
            {'additional_special_tokens': self.extra_special_tokens}
            )
        src_tokenizer.eos_token = self.eos_token

        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.max_seq = max_seq
        self.opcode = opcode
        self.op_code_type = OP_SET[self.opcode]

        self.domain_meta = sorted(list(set(x.split('-')[0] for x in self.slot_meta)))
        self.slot_domain_info = {x:self.domain_meta.index(x.split('-')[0]) for x in self.slot_meta}

        self.slot_token_id = self.src_tokenizer.encode('[SLOT]', add_special_tokens=False)[0]
        self.s_v_token_id = self.src_tokenizer.encode('[S-V]', add_special_tokens=False)[0]

    def _convert_example_to_feature(self, example):
        guid = example.guid

        states = convert_state_dict(example.label)
        before_states = convert_state_dict(example.before_label)
        tok_slots = []
        op_ids = []
        domain_score = [0] * len(self.domain_meta)
        gen_ids = []
        for i in range(len(domain_score)):
            domain_score[i] += random.random() / 4
        for slot in self.slot_meta:
            value = states.get(slot, "none")
            before_value = before_states.get(slot, 'none')
            op_id = get_op_id(before_value, value, self.opcode)
            op_ids.append(op_id)
            domain_score[self.slot_domain_info[slot]] += OP_SET[self.opcode]['carryover'] != op_id
            if value == 'none':
                value = '[NULL]'
        
            tok_slot_data = self.src_tokenizer.encode(slot, add_special_tokens=False)
            tok_value = self.src_tokenizer.encode(value, add_special_tokens=False)
            tok_slot = [self.slot_token_id] + tok_slot_data + [self.s_v_token_id] + tok_value

            tok_slots.extend(tok_slot)

            if OP_SET[self.opcode]['update'] == op_id:
                gen_ids.append(tok_value + [self.src_tokenizer.eos_token_id])

        domain_id = np.argmax(domain_score)
        history = example.context_turns[-2:]
        if len(history) == 0:
            history = ['', '']
        concat_history = history[0] + '[A-U]' + history[1]
        tok_history = self.src_tokenizer.encode(concat_history, add_special_tokens=False)

        current = example.current_turn[0] + '[A-U]' + example.current_turn[1] + self.src_tokenizer.sep_token
        tok_current = self.src_tokenizer.encode(current, add_special_tokens=False)

        max_seq_len = self.max_seq - 3
        slots_len = min(len(tok_slots), max_seq_len)
        current_len = min(max_seq_len - slots_len, len(tok_current))
        history_len = min(max_seq_len - slots_len - current_len, len(tok_history))
        
        history_start = len(tok_history) - history_len
        tok_history = [self.src_tokenizer.cls_token_id] + tok_history[history_start:] + \
                [self.src_tokenizer.sep_token_id]

        current_start = len(tok_current) - current_len
        tok_current = tok_current[current_start:] + [self.src_tokenizer.sep_token_id]

        assert len(tok_slots) == slots_len

        input_ids = tok_history + tok_current + tok_slots
        segment_ids = [0] * len(tok_history) + [1] * len(tok_current) + [1] * len(tok_slots)
        
        state_position_ids = []
        for i, v in enumerate(input_ids):
            if v == self.slot_token_id:
                state_position_ids.append(i)
        
        if len(gen_ids) == 0:
            gen_ids = [[0]]
        return SOMDSTFeature(
            guid=guid,
            input_ids=input_ids,
            segment_ids=segment_ids,
            state_position_ids=state_position_ids,
            op_ids=op_ids,
            domain_id=domain_id,
            gen_ids=gen_ids,
        )

    def convert_examples_to_features(self, examples, which=''):
        tdata = tqdm(examples, desc=f'Converting {which} examples to features')
        return list(map(self._convert_example_to_feature, tdata))

    # TODO:
    def recover_state(self, pred_slots, num_turn):
        raise NotImplementedError()
        states = []
        
        for pred_slot in pred_slots[:num_turn]:
            state = []
            for s, p in zip(self.slot_meta, pred_slot):
                v = self.ontology[s][p]
                if self.use_convert_ont:
                    if self.convert_time_dict is not None and \
                        s in self.convert_time_dict['applied']:
                        v = self.convert_time_dict['revert'](v)
                if v != 'none':
                    state.append(f'{s}-{v}')
                    
            states.append(state)
            
        return states

    def collate_fn(self, batch):
        input_ids = torch.LongTensor(self.pad_ids([b.input_ids for b in batch],
                 self.src_tokenizer.pad_token_id))
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id).to(torch.long)
        segment_ids = torch.LongTensor(self.pad_ids([b.segment_ids for b in batch], 0))

        state_position_ids = torch.LongTensor([b.state_position_ids for b in batch])

        op_ids = torch.LongTensor([b.op_ids for b in batch])
        domain_ids = torch.LongTensor([b.domain_id for b in batch])
        max_update = max([len(b.gen_ids) for b in batch])
        max_value = max([len(x) for b in batch for x in b.gen_ids])
        gen_tmp = [self.pad_ids(b.gen_ids, 0, max_length=max_value) for b in batch]
        for x in gen_tmp:
            if len(x) < max_update:
                diff = max_update - len(x)
                for i in range(diff):
                    x.append([0] * max_value)
        gen_ids = self.pad_id_of_matrix(
            [torch.LongTensor(x) for x in gen_tmp],
            self.trg_tokenizer.pad_token_id,
        )
        guids = [b.guid for b in batch]
        return input_ids, input_masks, segment_ids, state_position_ids, op_ids,\
            domain_ids, gen_ids, max_value, max_update, guids

