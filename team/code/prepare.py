from attrdict import AttrDict
from importlib import import_module
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import json
import os
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer
from data_utils import (
    load_dataset, 
    get_examples_from_dialogues, 
    convert_state_dict, 
    DSTInputExample, 
    OpenVocabDSTFeature, 
    DSTPreprocessor, 
    WOSDataset)

from typing import Union

def set_directory(new_dir: str):
    """ eda 저장용 dir 지정하기
    여러 사람 코드 급하게 합치다가 남은거

    Args:
        new_dir (str): eda에 사용될 dir
    """
    global directory # 이거 아마도 eda.py에 선언됨
    directory= new_dir
    if not os.path.exists(directory):
        os.makedirs(directory)

# 특정 slot-meta만 사용하려고 할 때 필요
# 미리 정의된 slot-meta들
# config.yml에 use_domain_slot 옵션이랑 연관 있음
gen_slot_meta = set(['관광-이름', '숙소-예약 기간', '숙소-예약 명수', '숙소-이름', '식당-예약 명수', '식당-이름', '택시-도착지', '택시-출발지', '지하철-도착지', '지하철-출발지'])
time_slot_meta = set(['식당-예약 시간', '지하철-출발 시간', '택시-도착 시간', '택시-출발 시간'])

def get_active_slot_meta(
    args: Union[AttrDict, wandb.Config],
    slot_meta: Union[list, set]
    ) -> tuple[set, set]:
    """ args를 참조해서 현재 사용하는 domain-slot이랑 domain을 리턴함
    Args:
        args (Union[AttrDict, wandb.Config]): use_domain_slot이 있는 args
        slot_meta (Union[list, set]): 전체 domain-slot 정보

    Raises:
        NotImplementedError: 현재 지원되지 않는 domain slot 가져올려고 할 때 발생

    Returns:
        tuple[set, set]: domain-slot, domain 정보
    """
    if args.use_domain_slot == 'gen':
        filter_slot_meta = gen_slot_meta
    elif args.use_domain_slot == 'time':
        filter_slot_meta = time_slot_meta
    elif args.use_domain_slot == 'cat':
        filter_slot_meta = set(slot_meta) - gen_slot_meta
    else:
        raise NotImplementedError(f'not implemented {args.use_domain_slot}')
    filter_domain = set(s.split('-')[0] for s in filter_slot_meta)
    return filter_slot_meta, filter_domain

# 사용 안되는 함수
# def filter_inference(args, data, slot_meta, ontology):
#     if args.use_domain_slot == 'basic':
#         return data, slot_meta, ontology

#     filter_slot_meta, filter_domain = get_active_slot_meta(args, slot_meta)
#     print(f'Inferencing with only {" ".join(filter_slot_meta)}')

#     old_data = data
#     data = []
#     for dial in old_data:
#         if any([x in filter_domain for x in dial['domains']]):
#             new_domains = [x for x in dial['domains'] if x in filter_domain]
#             dial['domains'] = new_domains
#             if len(new_domains) > 0:
#                 data.append(dial)

#     print(f'Filtered {len(old_data)} -> {len(data)}')

#     slot_meta = sorted(list(filter_slot_meta))
#     new_ontology = {}
#     for cur_slot_meta in slot_meta:
#         new_ontology[cur_slot_meta] = ontology[cur_slot_meta]
#     ontology = new_ontology

#     return data, slot_meta, ontology

def get_data(args: Union[AttrDict, wandb.Config]) -> tuple:
    """config에서 data, slot-meta, ontology 정보 가져오기
    -- 영향주는 config 설정 --
    train_from_trained -> 이미 학습된 모델을 사용하는지, 이미 학습된 모델의 slot_meta, ontology 가져옴
    use_convert_ont -> ontology를 바꿀건지 여부(sumbt용, 사용안함)(sumbt가 ontology에서 더 의미를 찾을까봐)
    convert_time -> 시간을 다른 형식으로 바꿔서 사용할건지(xx시 xx분) 이런 형태
    use_domain_slot -> 특정 domain-slot만 사용할지 여부
    use_small_data -> 데이터 100짜리로 줄여서 사용할지 여부(테스트용)

    Args:
        args (Union[AttrDict, wandb.Config]): 설정 다 있는 args

    Returns:
        tuple: config에서 가져온 data, slot-meta, ontology 정보
    """

    print(f'using train: {args.train_file_name}')
    train_data_file = f"{args.data_dir}/{args.train_file_name}"
    data = json.load(open(train_data_file))
    
    # 미리 학습된 모델을 사용 여부에 따라 해당되는 slot-meta, ontology 가져오는 부분
    if 'train_from_trained' not in args:
        args.train_from_trained = None
    if args.train_from_trained is None:
        slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
        ontology = json.load(open(args.ontology_root))
    else:
        slot_meta = json.load(open(f"{args.train_from_trained}/slot_meta.json"))
        ontology = json.load(open(f'{args.train_from_trained}/edit_ontology_metro.json'))

    # ontology 형태 바꾸는 부분
    if args.use_convert_ont:
        if args.convert_time != 'none':
            convert_time_dict = getattr(import_module('change_ont_value'), args.convert_time)
            print(f'Change Time Format: xx:xx -> {convert_time_dict.example}')
            print(f'Change {"  ".join(convert_time_dict.applied)}')
            for cat in convert_time_dict.applied:
                if cat in ontology:
                    ontology[cat] = [convert_time_dict.convert(x) for x in ontology[cat]]
            args.convert_time_dict = convert_time_dict
        else:
            args.convert_time_dict = None
            
    # 모든 domain-slot 사용시
    if args.use_domain_slot == 'basic':
        if args.use_small_data:
            data = data[:100]
        return data, slot_meta, ontology

    filter_slot_meta, filter_domain = get_active_slot_meta(args, slot_meta)

    # 사용하는 domain-slot 정보 없는 다이얼로그 필터링
    print(f'Using only {" ".join(filter_slot_meta)}')
    old_data = data
    data = []
    for dial in old_data:
        if any([x in filter_domain for x in dial['domains']]):
            new_domains = [x for x in dial['domains'] if x in filter_domain]
            dial['domains'] = new_domains
            new_dialogue = []
            is_worthy = False
            for x in dial['dialogue']:
                if 'state' in x:
                    new_state = [st for st in x['state'] if '-'.join(st.split('-')[:2])  in filter_slot_meta]
                    x['state'] = new_state
                    if not is_worthy and len(new_state) > 0:
                        is_worthy = True
                new_dialogue.append(x)
            dial['dialogue'] = new_dialogue

            if is_worthy:
                data.append(dial)
    print(f'Filtered {len(old_data)} -> {len(data)}')
    # sort하는거 중요, slot-domain를 index 기반으로 접근함, slot-domain 일정한 순서 보장 중요
    slot_meta = sorted(list(filter_slot_meta)) 
    new_ontology = {}
    for cur_slot_meta in slot_meta:
        new_ontology[cur_slot_meta] = ontology[cur_slot_meta]
    ontology = new_ontology

    if args.use_small_data:
        data = data[:100]

    return data, slot_meta, ontology

def get_stuff(
    args: Union[AttrDict, wandb.Config],
    train_data: list,
    dev_data: Union[list, None],
    slot_meta,
    ontology
    ) -> tuple:
    """ 주어진 데이터 -> examples -> features 생성해 리턴 + 사용된 tokenizer, processor 리턴
    사용하는 tokenizer, processor는 args에 설정에서 만듬
    dev_data가 None이면 train_data만 features로 만듬

    Args:
        args (Union[AttrDict, wandb.Config]): args
        train_data (list): 학습 데이터
        dev_data (Union[list, None]): validation 데이터, None이면 다 학습 데이터라고 생각함
        slot_meta: 사용하는 slot meta
        ontology: 사용하는 ontology

    Raises:
        NotImplementedError: 지원되지 않는 모델 사용할려고 할 때

    Returns:
        tuple: tokenizer, processor, train_features, dev_features 리턴
    """
    # preprocessor 관련 세팅
    if args.preprocessor == 'TRADEPreprocessor':
        user_first = False
        dialogue_level = False
        processor_kwargs = AttrDict(
            use_zero_segment_id=args.use_zero_segment_id,
        )
    elif args.preprocessor == 'SUMBTPreprocessor':
        user_first = True
        dialogue_level = True
        max_turn = max([len(e['dialogue']) for e in train_data])
        processor_kwargs = AttrDict(
            ontology=ontology,
            max_turn_length=max_turn,
            max_seq_length=args.max_seq_length,
            model_name_or_path=args.model_name_or_path,
            args=args,
        )
    elif args.preprocessor == 'SOMDSTPreprocessor':
        user_first = False
        dialogue_level = False
        processor_kwargs = AttrDict()
    else:
        raise NotImplementedError()

    # data -> examples
    train_examples = get_examples_from_dialogues(
        train_data, user_first=user_first, use_sys_usr_sys=args.use_sys_usr_sys_turn,
             dialogue_level=dialogue_level, which='train'
    )

    if dev_data is not None:
        dev_examples = get_examples_from_dialogues(
            dev_data, user_first=user_first, use_sys_usr_sys=args.use_sys_usr_sys_turn,
            dialogue_level=dialogue_level, which='val'
        )

    # tokenizer, preprocessor 생성
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    processor = getattr(import_module('preprocessor'), args.preprocessor)(
        slot_meta, tokenizer, **processor_kwargs
    )
    args.vocab_size = len(tokenizer)

    if args.preprocessor == 'TRADEPreprocessor':
        args.n_gate = len(processor.gating2id) # gating 갯수 none, dontcare, ptr

    # Extracting Featrues
    # print('Converting examples to features')
    train_features = processor.convert_examples_to_features(train_examples, which='train')
    if dev_data is not None:
        dev_features = processor.convert_examples_to_features(dev_examples, which='val')
    else:
        dev_features = None
    # print('Done converting examples to features')
    
    return tokenizer, processor, train_features, dev_features

# 아마 제공된 코드
def tokenize_ontology(ontology, tokenizer, max_seq_length):
    slot_types = []
    slot_values = []
    for k, v in ontology.items():
        tokens = tokenizer.encode(k)
        if len(tokens) < max_seq_length:
            gap = max_seq_length - len(tokens)
            tokens.extend([tokenizer.pad_token_id] * gap)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        slot_types.append(tokens)
        slot_value = []
        for vv in v:
            tokens = tokenizer.encode(vv)
            if len(tokens) < max_seq_length:
                gap = max_seq_length - len(tokens)
                tokens.extend([tokenizer.pad_token_id] * gap)
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
            slot_value.append(tokens)
        slot_values.append(torch.LongTensor(slot_value))
    return torch.LongTensor(slot_types), slot_values

""" 주어진 데이터 -> examples -> features 생성해 리턴 + 사용된 tokenizer, processor 리턴
    사용하는 tokenizer, processor는 args에 설정에서 만듬
    dev_data가 None이면 train_data만 features로 만듬

    Args:
        args (Union[AttrDict, wandb.Config]): args
        train_data (list): 학습 데이터
        dev_data (Union[list, None]): validation 데이터, None이면 다 학습 데이터라고 생각함
        slot_meta: 사용하는 slot meta
        ontology: 사용하는 ontology

    Raises:
        NotImplementedError: 지원되지 않는 모델 사용할려고 할 때

    Returns:
        tuple: tokenizer, processor, train_features, dev_features 리턴
    """
def get_model(args: Union[AttrDict, wandb.Config], tokenizer, ontology, slot_meta):
    """ args에 설정된 걸로 모델 가져오기

    Args:
        args (Union[AttrDict, wandb.Config]): args
        tokenizer: 사용하는 tokenzier
        ontology: 사용하는 ontology 
        slot_meta: 사용하는 slot-meta

    Raises:
        NotImplementedError: 지원되지 않는 모델 사용할려고 할 때

    Returns:
        model: args으로 만든 모델
    """
    # 각각 모델마다 필요한 것들 설정
    if args.ModelName == 'TRADE':
        tokenized_slot_meta = []
        for slot in slot_meta:
            tokenized_slot_meta.append(
                tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
            )

        model_kwargs = AttrDict(
            slot_meta=tokenized_slot_meta
        )
    elif args.ModelName == 'SUMBT':
        slot_type_ids, slot_values_ids = tokenize_ontology(ontology, tokenizer, args.max_label_length)
        num_labels = [len(s) for s in slot_values_ids]

        model_kwargs = AttrDict(
            num_labels=num_labels,
            device=args.device,
        )
    elif args.ModelName == 'SOM_DST':
        model_kwargs = AttrDict(
            n_op=4,
            n_domain=5,
            update_id=1,
            len_tokenizer=len(tokenizer)
        )
    else:
        raise NotImplementedError()

    pbar = tqdm(desc=f'Making {args.model_class} model -- waiting...', bar_format='{desc} -> {elapsed}')
    model = getattr(import_module('model'), args.model_class)(
        args, **model_kwargs
    )
    pbar.set_description(f'Making {args.model_class} model -- DONE')    
    pbar.close()

    # 각각 모델마다 필요한 후처리
    # if args.ModelName == 'TRADE':
    #     pbar = tqdm(desc='Setting subword embedding -- waiting...', bar_format='{desc} -> {elapsed}')
    #     model.set_subword_embedding(args.model_name_or_path)  # Subword Embedding 초기화    
    #     pbar.set_description('Setting subword embedding -- DONE')
    #     pbar.close()
    if args.ModelName == 'SUMBT':
        print('Initializing slot value lookup --------------')
        model.initialize_slot_value_lookup(slot_values_ids, slot_type_ids)  # Tokenized Ontology의 Pre-encoding using BERT_SV        
        print('Finished initializing slot value lookup -----')

    return model