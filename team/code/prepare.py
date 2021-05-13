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

def set_directory(new_dir):
    global directory
    directory= new_dir
    if not os.path.exists(directory):
        os.makedirs(directory)

gen_slot_meta = set(['관광-이름', '숙소-예약 기간', '숙소-예약 명수', '숙소-이름', '식당-예약 명수', '식당-이름', '택시-도착지', '택시-출발지', '지하철-도착지', '지하철-출발지'])
time_slot_meta = set(['식당-예약 시간', '지하철-출발 시간', '택시-도착 시간', '택시-출발 시간'])

def get_active_slot_meta(args, slot_meta):
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

def filter_inference(args, data, slot_meta, ontology):
    if args.use_domain_slot == 'basic':
        return data, slot_meta, ontology

    filter_slot_meta, filter_domain = get_active_slot_meta(args, slot_meta)
    print(f'Inferencing with only {" ".join(filter_slot_meta)}')

    old_data = data
    data = []
    for dial in old_data:
        if any([x in filter_domain for x in dial['domains']]):
            new_domains = [x for x in dial['domains'] if x in filter_domain]
            dial['domains'] = new_domains
            if len(new_domains) > 0:
                data.append(dial)

    print(f'Filtered {len(old_data)} -> {len(data)}')

    slot_meta = sorted(list(filter_slot_meta))
    new_ontology = {}
    for cur_slot_meta in slot_meta:
        new_ontology[cur_slot_meta] = ontology[cur_slot_meta]
    ontology = new_ontology

    return data, slot_meta, ontology

def get_data(args):
    train_data_file = f"{args.data_dir}/train_dials.json"
    data = json.load(open(train_data_file))
        
    if args.train_from_trained is None:
        slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
        ontology = json.load(open(args.ontology_root))
    else:
        slot_meta = json.load(open(f"{args.train_from_trained}/slot_meta.json"))
        ontology = json.load(open(f'{args.train_from_trained}/edit_ontology_metro.json'))

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
            
    if args.use_domain_slot == 'basic':
        if args.use_small_data:
            data = data[:100]
        return data, slot_meta, ontology

    filter_slot_meta, filter_domain = get_active_slot_meta(args, slot_meta)

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
    slot_meta = sorted(list(filter_slot_meta))
    new_ontology = {}
    for cur_slot_meta in slot_meta:
        new_ontology[cur_slot_meta] = ontology[cur_slot_meta]
    ontology = new_ontology

    if args.use_small_data:
        data = data[:100]

    return data, slot_meta, ontology

def get_stuff(args, train_data, dev_data, slot_meta, ontology):
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
    else:
        raise NotImplementedError()

    train_examples = get_examples_from_dialogues(
        train_data, user_first=user_first, use_sys_usr_sys=args.use_sys_usr_sys_turn,
             dialogue_level=dialogue_level, which='train'
    )

    if dev_data is not None:
        dev_examples = get_examples_from_dialogues(
            dev_data, user_first=user_first, use_sys_usr_sys=args.use_sys_usr_sys_turn,
            dialogue_level=dialogue_level, which='val'
        )

    # Define Preprocessor
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

def get_model(args, tokenizer, ontology, slot_meta):
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
    else:
        raise NotImplementedError()

    pbar = tqdm(desc=f'Making {args.model_class} model -- waiting...', bar_format='{desc} -> {elapsed}')
    model = getattr(import_module('model'), args.model_class)(
        args, **model_kwargs
    )
    pbar.set_description(f'Making {args.model_class} model -- DONE')    
    pbar.close()

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