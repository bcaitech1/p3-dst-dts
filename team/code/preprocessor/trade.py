import torch
from tqdm.auto import tqdm
import numpy as np

from data_utils import DSTPreprocessor, OpenVocabDSTFeature, convert_state_dict

class TRADEPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        ontology=None,
        max_seq_length=512,
        use_zero_segment_id=True
    ):

        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.gating2id = {"none": 0, "dontcare": 1, "yes": 2, "no": 3, "ptr": 4}
        self.id2gating = {v: k for k, v in self.gating2id.items()}
        self.max_seq_length = max_seq_length
        self.use_zero_segment_id = use_zero_segment_id

    def _convert_example_to_feature(self, example):
        dialogue_context = " [SEP] ".join(example.context_turns + example.current_turn)
        #이전 context에 current_turn 대화를 붙여준다
        input_id = self.src_tokenizer.encode(dialogue_context, add_special_tokens=False)
        input_id = input_id + [self.src_tokenizer.sep_token_id] # seg 만들기 편하게

        # input_id 랑 똑같게 segment_id 생성
        # 0은 SYS 발화, 1은 USR 발화
        # 처음 CLS token은 처음 나오는 값 따라가게
        # 1 2 3 sep 2 3 -> 0 0 0 0 1 1  이렇게 SEP token 까지
        np_input_id = np.array(input_id)
        sep_token_idxs = np.where(np_input_id == self.src_tokenizer.sep_token_id)[0]
        seg_tmp = np.zeros(len(np_input_id), dtype=np.int)
        cur_idx = 1
        while cur_idx < len(sep_token_idxs):
            start = sep_token_idxs[cur_idx-1]
            end = sep_token_idxs[cur_idx]
            seg_tmp[start+1:end+1] = 1
            cur_idx += 2
        segment_id = seg_tmp.tolist()

        max_length = self.max_seq_length - 1
        #max_length보다 길어진다면 그나마 가장 최신 current_turn에 가깝게 끊어준다
        if len(input_id) > max_length: 
            gap = len(input_id) - max_length
            input_id = input_id[gap:]
            segment_id = segment_id[gap:]
        input_id = [self.src_tokenizer.cls_token_id] + input_id
        segment_id = [segment_id[0]] + segment_id

        if self.use_zero_segment_id:
            segment_id = [0] * len(segment_id)

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

    def convert_examples_to_features(self, examples, which=''):
        tdata = tqdm(examples, desc=f'Converting {which} examples to features')
        return list(map(self._convert_example_to_feature, tdata))

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

            #인코딩된 벨류 토큰들을 스페셜토큰을 떼고 다시 subword로 조합해줌
            token_id_list = []
            for id_ in value:
                if id_ in self.trg_tokenizer.all_special_ids:
                    break

                token_id_list.append(id_)
            value = self.trg_tokenizer.decode(token_id_list, skip_special_tokens=True)

            #ptr값으로 존재하는 none이 "no"와 "ne"로 인코딩되므로 합칠 때 none이 나옴
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