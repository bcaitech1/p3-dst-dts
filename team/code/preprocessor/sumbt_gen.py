from data_utils import convert_state_dict
from data_utils import OpenTurnVocabDSTFeature, DSTPreprocessor, _truncate_seq_pair
from tqdm.auto import tqdm
import torch

class SUMBTGenPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        max_seq_length=64,
        max_turn_length=14,
        model_name_or_path=None
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.max_seq_length = max_seq_length
        self.max_turn_length = max_turn_length
        self.model_name_or_path = model_name_or_path

        self.gating2id = {"none": 0, "dontcare": 1, "ptr": 2}
        self.id2gating = {v: k for k, v in self.gating2id.items()}

    def _convert_example_to_feature(self, example):
        guid = example[0].guid.rsplit("-", 1)[0]  # dialogue_idx
        turns = []
        token_types = []
        labels = []
        gating_ids = []
        num_turn = None
        for turn in example[: self.max_turn_length]:
            assert len(turn.current_turn) == 2
            uttrs = []
            for segment_idx, uttr in enumerate(turn.current_turn):
                token = self.src_tokenizer.encode(uttr, add_special_tokens=False)
                uttrs.append(token)

            _truncate_seq_pair(uttrs[0], uttrs[1], self.max_seq_length - 3)
            tokens = (
                [self.src_tokenizer.cls_token_id]
                + uttrs[0]
                + [self.src_tokenizer.sep_token_id]
                + uttrs[1]
                + [self.src_tokenizer.sep_token_id]
            )

            if self.model_name_or_path == 'xlm-roberta-base': # roberta는 token_types 사용안함
                token_type = [0] * (len(uttrs[0]) + 2 + len(uttrs[1]) + 1)
            else:
                token_type = [0] * (len(uttrs[0]) + 2) + [1] * (len(uttrs[1]) + 1)
            if len(tokens) < self.max_seq_length:
                gap = self.max_seq_length - len(tokens)
                tokens.extend([self.src_tokenizer.pad_token_id] * gap)
                token_type.extend([0] * gap)
            turns.append(tokens)
            token_types.append(token_type)

            target_ids = []
            gating_id = []
            if turn.label:
                slot_dict = convert_state_dict(turn.label)
            else:
                slot_dict = {}
            for slot_type in self.slot_meta:
                value = slot_dict.get(slot_type, "none")
                
                target_id = self.trg_tokenizer.encode(value, add_special_tokens=False,
                    truncation=True, max_length=self.max_seq_length-1) + [self.trg_tokenizer.sep_token_id]
                target_ids.append(target_id)
                gating_id.append(self.gating2id.get(value, self.gating2id["ptr"]))
            target_ids = self.pad_ids(target_ids, self.trg_tokenizer.pad_token_id)
            labels.append(target_ids)
            gating_ids.append(gating_id)
        num_turn = len(turns)


        if len(turns) < self.max_turn_length:
            gap = self.max_turn_length - len(turns)
            for _ in range(gap):
                dummy_turn = [self.src_tokenizer.pad_token_id] * self.max_seq_length
                dummy_token_type = [0] * self.max_seq_length
                turns.append(dummy_turn)
                token_types.append(dummy_token_type)
                dummy_label = [[self.src_tokenizer.pad_token_id]] * len(self.slot_meta)
                labels.append(dummy_label)
                gating_ids.append([self.gating2id["none"]] * len(self.slot_meta))
        return OpenTurnVocabDSTFeature(
            guid=guid,
            input_ids=turns,
            segment_ids=token_types,
            num_turn=num_turn,
            target_ids=labels,
            gating_ids=gating_ids
        )


    def convert_examples_to_features(self, examples, which=''):
        tdata = tqdm(examples, desc=f'Converting {which} examples to features')
        return list(map(self._convert_example_to_feature, tdata))

    # M, J, G
    # M, J
    def recover_state(self, turn_gate_list, turn_gen_list, num_turn):
        states = []
        for gate_list, gen_list in zip(turn_gate_list[:num_turn], turn_gen_list[:num_turn]):
            state = []
            for slot, gate, value in zip(self.slot_meta, gate_list, gen_list):
                if self.id2gating[gate] == "none":
                    continue

                if self.id2gating[gate] == "dontcare":
                    state.append("%s-%s" % (slot, "dontcare"))
                    continue

                token_id_list = []
                for id_ in value:
                    if id_ in self.trg_tokenizer.all_special_ids:
                        break

                    token_id_list.append(id_)
                value = self.trg_tokenizer.decode(token_id_list, skip_special_tokens=True)

                if value == "none":
                    continue

                state.append("%s-%s" % (slot, value))
            states.append(state)
        return states

    def collate_target_ids(self, batch, pad_token_id):
        max_len = 0
        for b in batch:
            for t in b.target_ids:
                for k in t:
                    max_len = max(max_len, len(k))
        
        new_arrays = []
        for b in batch:
            tmp = []
            for t in b.target_ids:
                tmp.append(self.pad_ids(t, pad_token_id, max_length=max_len))
            new_arrays.append(tmp)
        
        return torch.LongTensor(new_arrays)

    def collate_fn(self, batch):
        guids = [b.guid for b in batch]
        input_ids = torch.LongTensor([b.input_ids for b in batch])
        segment_ids = torch.LongTensor([b.segment_ids for b in batch])
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id)
        target_ids = self.collate_target_ids(batch, self.trg_tokenizer.pad_token_id)
        num_turns = [b.num_turn for b in batch]
        gating_ids = torch.LongTensor([b.gating_ids for b in batch])
        return input_ids, segment_ids, input_masks, target_ids, gating_ids, num_turns, guids
        