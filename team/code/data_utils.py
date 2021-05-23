import dataclasses
import json
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


@dataclass
class OntologyDSTFeature:
    guid: str
    input_ids: List[int]
    segment_ids: List[int]
    num_turn: int
    target_ids: Optional[List[int]]


@dataclass
class OpenVocabDSTFeature:
    guid: str
    input_id: List[int]
    segment_id: List[int]
    gating_id: List[int]
    target_ids: Optional[Union[List[int], List[List[int]]]]


class WOSDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]

def load_dataset(data, dev_split=0.1, given_dev_idx=None,
     filter_old_data=False, dev_has_label=False):
    """ 데이터를 train, dev로 나눔
    제공된 코드에서 이것 저것 추가함

    given_dev_idx: 특정한 dev idx로 dev 나눌때 사용, 리턴하는 dev_idx 기존꺼 리턴
    filter_old_data: 추가 데이터에서 dev에 사용될게 train에 안들어가게 필터 여부
        추가 데이터 형식이 숫자가 아니라는 가정
        이거 사용 안함, 사실 잘 기억 안남
    dev_has_label: dev 데이터도 train 데이터처럼 label 가지게함

    Args:
        data: data
        dev_split (float, optional): 나누는 비율. Defaults to 0.1.
        given_dev_idx (list, optional): dev idx 리스트. Defaults to None.
        filter_old_data (bool, optional): 뭐였지. Defaults to False.
        dev_has_label (bool, optional): dev 데이터 label 여부. Defaults to False.

    Returns:
        tuple: train_data, dev_data, dev_labels, dev_idx
    """
    # no dev
    num_data = len(data)
    num_dev = int(num_data * dev_split)
    if not num_dev:
        return data, [], None, None  # no dev dataset

    # 제공된 dev idx 사용
    if given_dev_idx is None:
        dom_mapper = defaultdict(list)
        for d in data:
            dom_mapper[len(d["domains"])].append(d["dialogue_idx"])

        num_per_domain_trainsition = int(num_dev / 3)
        dev_idx = []
        for v in dom_mapper.values():
            idx = random.sample(v, num_per_domain_trainsition)
            dev_idx.extend(idx)
    else:
        dev_idx = given_dev_idx

    dev_idx_check = set(dev_idx)

    # train, dev 나눔
    train_data, dev_data = [], []
    filter_count = 0
    for d in data:
        if d["dialogue_idx"] in dev_idx_check:
            dev_data.append(d)
        else:
            if not filter_old_data:
                train_data.append(d)
            elif not ('0' <= d['dialogue_idx'][-1] <= '9'):
                should_use = True
                for key in dev_idx_check:
                    if key in d['dialogue_idx']:
                        should_use = False
                        break
                if should_use:
                    train_data.append(d)
                else:
                    filter_count += 1

    if filter_old_data:
        print(f'filter {filter_count} from val')

    dev_labels = {}
    for dialogue in dev_data:
        d_idx = 0
        guid = dialogue["dialogue_idx"]
        for idx, turn in enumerate(dialogue["dialogue"]):
            if turn["role"] != "user":
                continue

            if dev_has_label:
                state = turn["state"]
            else:
                state = turn.pop("state")

            guid_t = f"{guid}-{d_idx}"
            d_idx += 1

            dev_labels[guid_t] = state

    return train_data, dev_data, dev_labels, dev_idx


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def split_slot(dom_slot_value, get_domain_slot=False):
    try:
        dom, slot, value = dom_slot_value.split("-")
    except ValueError:
        tempo = dom_slot_value.split("-")
        if len(tempo) < 2:
            return dom_slot_value, dom_slot_value, dom_slot_value
        dom, slot = tempo[0], tempo[1]
        value = dom_slot_value.replace(f"{dom}-{slot}-", "").strip()

    if get_domain_slot:
        return f"{dom}-{slot}", value
    return dom, slot, value


def build_slot_meta(data):
    slot_meta = []
    for dialog in data:
        for turn in dialog["dialogue"]:
            if not turn.get("state"):
                continue

            for dom_slot_value in turn["state"]:
                domain_slot, _ = split_slot(dom_slot_value, get_domain_slot=True)
                if domain_slot not in slot_meta:
                    slot_meta.append(domain_slot)
    return sorted(slot_meta)


def convert_state_dict(state):
    dic = {}
    for slot in state:
        s, v = split_slot(slot, get_domain_slot=True)
        dic[s] = v
    return dic


@dataclass
class DSTInputExample:
    guid: str
    context_turns: List[str]
    current_turn: List[str]
    label: Optional[List[str]] = None
    before_label: Optional[List[str]] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def get_examples_from_dialogue(dialogue, user_first=False, use_sys_usr_sys=False):
    """ 제공된 거랑 거의 비슷

    use_sys_usr_sys: SYS USR SYS로 사용할지 여부

    Args:
        dialogue: 다이얼로그
        user_first (bool, optional): 유저, 시스템 발화 순서 여부. Defaults to False.
        use_sys_usr_sys (bool, optional): SYS, USR, SYS 사용 여부. Defaults to False.

    Returns:
        examples
    """
    guid = dialogue["dialogue_idx"]
    examples = []
    history = []
    d_idx = 0
    before_state = []
    len_diag = len(dialogue["dialogue"])
    for idx, turn in enumerate(dialogue["dialogue"]):
        if turn["role"] != "user":
            continue

        if idx:
            sys_utter = dialogue["dialogue"][idx - 1]["text"]
        else:
            sys_utter = ""

        if idx + 1 < len_diag:
            next_utter = dialogue["dialogue"][idx + 1]["text"]
        else:
            next_utter = ""

        user_utter = turn["text"]
        state = turn.get("state")
        context = deepcopy(history)
        if use_sys_usr_sys:
            current_turn = [sys_utter, user_utter, next_utter]
        elif user_first:
            current_turn = [user_utter, sys_utter]
        else:
            current_turn = [sys_utter, user_utter]
        examples.append(
            DSTInputExample(
                guid=f"{guid}-{d_idx}",
                context_turns=context,
                current_turn=current_turn,
                label=state,
                before_label=before_state,
            )
        )
        before_state = state
        history.append(sys_utter)
        history.append(user_utter)
        d_idx += 1
    return examples


def get_examples_from_dialogues(data, user_first=False, use_sys_usr_sys=False,
         dialogue_level=False, which=''):
    """ 제공된 거랑 거의 같음

    use_sys_usr_sys: SYS USR SYS로 사용할지 여부
    which: 출력용, 지금 dev인지, train인지 보이게

    Args:
        data: 데이터
        user_first (bool, optional): 유저, 시스템 발화 순서 여부. Defaults to False.
        use_sys_usr_sys (bool, optional): SYS, USR, SYS 사용 여부. Defaults to False.
        dialogue_level (bool, optional): 다이얼로그를 1단 2단 할지. Defaults to False.
        which (str, optional): tqdm 출력용. Defaults to ''.

    Returns:
        examples, dialogue_level에 따라 형식은 조금 다름
    """
    examples = []
    pbar = tqdm(data, desc=f'Getting {which} examples from dialogues')
    for d in pbar:
        example = get_examples_from_dialogue(d, user_first=user_first, 
                use_sys_usr_sys=use_sys_usr_sys)
        if dialogue_level:
            examples.append(example)
        else:
            examples.extend(example)
    return examples


class DSTPreprocessor:
    def __init__(self, slot_meta, src_tokenizer, trg_tokenizer=None, ontology=None):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology

    def pad_ids(self, arrays, pad_idx, max_length=-1):
        if len(arrays) == 0:
            return arrays
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
        return arrays

    def pad_id_of_matrix(self, arrays, padding, max_length=-1, left=False):
        if max_length < 0:
            max_length = max([array.size(-1) for array in arrays])

        new_arrays = []
        for i, array in enumerate(arrays):
            n, l = array.size()
            pad = torch.zeros(n, (max_length - l))
            pad[
                :,
                :,
            ] = padding
            pad = pad.long()
            m = torch.cat([array, pad], -1)
            new_arrays.append(m.unsqueeze(0))

        return torch.cat(new_arrays, 0)

    def _convert_example_to_feature(self):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def recover_state(self):
        raise NotImplementedError
